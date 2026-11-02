"""Integrity, replay and optimizer-continuation checks for one finite budget test."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def sha(path):
    with Path(path).open('rb') as f:
        h = hashlib.sha256()
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


class BudgetHooks:
    def __init__(self, here, protocol):
        self.here, self.p = here, protocol
        self.origin = here.parent/'runs/bridge-demo806-fit-208207'
        self.ns = None

    def schedule(self, old_batches, report):
        import torch
        assert sha(self.origin/'report.json') == self.p['baseline_report_sha256']
        self.old = read(self.origin/'report.json')
        assert self.old['execution_checks_passed'] and self.old['optimization_operations'] == 8192
        assert sha(self.here/'schedule.npz') == self.p['schedule_sha256']
        with np.load(self.here/'schedule.npz', allow_pickle=False) as z:
            batches = z['batches'].copy()
        assert batches.shape == (16170, 8)
        assert np.array_equal(batches[:8192], old_batches.numpy())
        assert hashlib.sha256(batches.tobytes()).hexdigest() == self.p['schedule_tensor_sha256']
        for key in ('input_identity_sha256', 'token_cache_key_sha256', 'initial_shared_sha256',
                    'augmented_initial_sha256', 'torch_version', 'sampling_steps', 'sampling_seeds',
                    'input_row_checks', 'episode_payload_checks'):
            assert report[key] == self.old[key], key
        assert report['full_token_cache_reused'] and report['new_token_rows_encoded'] == report['old_token_rows_reencoded'] == 0
        logs = []
        for name in ('common_updates.jsonl', 'reduced_updates.jsonl'):
            logs.extend(json.loads(line) for line in (self.origin/name).read_text().splitlines())
        assert len(logs) == 8192 and [r['update'] for r in logs] == list(range(1, 8193))
        assert all(np.isfinite(r['loss']) and r['learning_rate'] == (1e-4 if r['update'] <= 4096 else 1e-5) for r in logs)
        assert logs[0]['loss'] == self.old['common_first_loss'] and logs[4095]['loss'] == self.old['common_last_loss']
        assert logs[-1]['loss'] == self.old['runs'][0]['loss_last']
        self.old_logs = logs
        report['budget'] = dict(protocol_sha256=sha(self.here/'protocol.json'),
            baseline_report_sha256=sha(self.origin/'report.json'),
            baseline_log_sha256={name: sha(self.origin/name) for name in ('common_updates.jsonl', 'reduced_updates.jsonl')},
            schedule_sha256=sha(self.here/'schedule.npz'), prefix8192_schedule_exact=True,
            endpoint_updates=16170, prefix_updates=8192, extension_updates=7978,
            original_learning_rate_switch=4096, extension_learning_rate=1e-5,
            additional_data=False, parameters_reinitialized_before_extension=False,
            optimizer_reset_before_extension=False, reserved25_scored=False,
            prefix8192_reconstruction_required=True,
            global_old_control_retrained_flag_refers_to405_reference=True)
        return torch.from_numpy(batches)

    def _frozen(self, x):
        import torch
        digest, prefixes = self.ns['digest'], self.ns['PREFIXES']
        assert x['clip_hash'] == [digest(x['model'].vision_encoder), digest(x['model'].text_encoder)]
        assert all(torch.equal(x['model'].state_dict()[n].cpu(), v) for n, v in x['initial'].items() if not n.startswith(prefixes))

    def reference_replay(self, x, label):
        import torch
        model, r, args = x['model'], x['r'], x['args']
        ids = list(range(3624, 3628))
        with self.ns['observation_guard'](model):
            ctx = torch.cat([x['context'](torch.tensor(ids[lo:lo+2], device='cuda')) for lo in range(0, 4, 2)])
            truth = x['targets'][ids].cpu().numpy()
            pred, _ = self.ns['evaluate'](model, ctx, x['current'][ids], truth, [1101, 1102, 1103], batch=2)
        with np.load(self.origin/'reduced_u8192_development.npz', allow_pickle=False) as old:
            assert np.array_equal(truth, old['targets'][:4])
            context_difference = float(np.max(np.abs(ctx.cpu().numpy()-old['contexts'][:4])))
            prediction_difference = float(np.max(np.abs(pred-old['predictions'][:, :4])))
        np.savez_compressed(args.output/(label+'_replay_check.npz'), contexts=ctx.cpu().numpy(), predictions=pred, targets=truth)
        proof = dict(context_max_abs=context_difference, prediction_max_abs=prediction_difference,
                     passed=context_difference <= self.p['tolerances']['context'] and prediction_difference <= self.p['tolerances']['prediction'])
        r['budget'][label+'_replay'] = proof
        print('BUDGET_REPLAY', label, json.dumps(proof), flush=True)
        assert proof['passed'], proof

    def save_state(self, x, name, update):
        import torch
        path = x['args'].output/name
        assert not path.exists()
        payload = dict(update=update, config=x['cfg'], input_protocol=x['p'], budget_protocol=self.p,
                       stage=x['args'].stage, actual_optimizer_operations=len(x['step_seconds']),
                       trainable_state_dict=self.ns['trainable_copy'](x['model']),
                       optimizer_state_dict=self.ns['cpu_copy'](x['optimizer'].state_dict()),
                       torch_cpu_rng=torch.get_rng_state(), torch_cuda_rng=(torch.cuda.get_rng_state() if torch.cuda.is_available() else None),
                       numpy_rng=np.random.get_state(), budget_protocol_sha256=sha(self.here/'protocol.json'))
        parameter_hash = self.ns['state_sha'](payload['trainable_state_dict'])
        optimizer_hash = self.ns['state_sha'](payload['optimizer_state_dict'])
        temporary = path.with_suffix('.partial')
        assert not temporary.exists()
        torch.save(payload, temporary)
        assert temporary.stat().st_size < 2*1024**3
        loaded = torch.load(temporary, map_location='cpu', weights_only=False)
        assert self.ns['state_sha'](loaded['trainable_state_dict']) == parameter_hash
        assert self.ns['state_sha'](loaded['optimizer_state_dict']) == optimizer_hash
        temporary.replace(path)
        x['r'].setdefault('optimizer_checkpoints', []).append(dict(path=name, update=update, bytes=path.stat().st_size,
            sha256=sha(path), parameters_sha256=parameter_hash, optimizer_sha256=optimizer_hash, roundtrip_exact=True))
        return path

    def preflight(self, x):
        import torch
        started = time.monotonic()
        model, optimizer, r = x['model'], x['optimizer'], x['r']
        before = self.ns['trainable_copy'](model)
        initial_hash = self.ns['state_sha'](before)
        checkpoint = self.origin/'final_trainable.pt'
        assert sha(checkpoint) == self.p['baseline_checkpoint_sha256']
        value = torch.load(checkpoint, map_location='cpu', weights_only=False)
        assert 'optimizer_state_dict' not in value, 'If optimizer exists, reconsider replay instead of duplicating work'
        assert set(value['trainable_state_dict']) == set(before)
        model.load_state_dict(value['trainable_state_dict'], strict=False)
        self.reference_replay(x, 'frozen208207')
        model.load_state_dict(before, strict=False)
        assert self.ns['state_sha'](self.ns['trainable_copy'](model)) == initial_hash
        del before, value
        for group in optimizer.param_groups:
            group['lr'] = 1e-5
        x['advance']('probe_warm', 8189, 8192)
        path = self.save_state(x, 'probe_optimizer.pt', 8192)
        first = x['advance']('probe_continue', 8193, 8196)
        first_parameters = self.ns['state_sha'](self.ns['trainable_copy'](model))
        first_optimizer = self.ns['state_sha'](optimizer.state_dict())
        loaded = torch.load(path, map_location='cpu', weights_only=False)
        model.load_state_dict(loaded['trainable_state_dict'], strict=False)
        optimizer.load_state_dict(loaded['optimizer_state_dict'])
        torch.set_rng_state(loaded['torch_cpu_rng']); torch.cuda.set_rng_state(loaded['torch_cuda_rng'])
        np.random.set_state(loaded['numpy_rng'])
        assert self.ns['state_sha'](self.ns['trainable_copy'](model)) == r['optimizer_checkpoints'][0]['parameters_sha256']
        assert self.ns['state_sha'](optimizer.state_dict()) == r['optimizer_checkpoints'][0]['optimizer_sha256']
        del loaded
        repeated = x['advance']('probe_restored', 8193, 8196)
        assert first == repeated
        assert self.ns['state_sha'](self.ns['trainable_copy'](model)) == first_parameters
        assert self.ns['state_sha'](optimizer.state_dict()) == first_optimizer
        # Only the just-created probe file is removed; original checkpoints remain untouched.
        path.unlink()
        r['optimizer_checkpoints'][0]['probe_deleted_after_verified_reload'] = True
        self._frozen(x)
        warm = float(np.mean(x['step_seconds'][1:]))
        estimate = 600 + 1.2*(self.old['elapsed_seconds'] + 7978*warm)
        r.update(execution_checks_passed=True, clip_unchanged=True, other_frozen_unchanged=True,
                 optimization_operations=len(x['step_seconds']), development_scored=False,
                 warm_update_seconds=warm, peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                 estimated_formal_seconds=estimate, trainable_parameters=sum(v.numel() for v in x['trainables']),
                 scientific_verdict='Operational preflight only; no new candidate generalization verdict.')
        r['budget'].update(optimizer_reload_continuation_exact=True, probe_updates=[8189, 8196],
                           actual_prefix8192_replay=False, preflight_seconds=time.monotonic()-started,
                           new_candidate_development_scored=False)
        assert r['optimization_operations'] == 12 and r['peak_reserved_gib'] < 12 and estimate < 12600
        print('BUDGET PREFLIGHT PASSED', x['args'].output/'report.json', flush=True)

    def require_preflight(self, report):
        path = Path(__import__('os').environ['BUDGET_PREFLIGHT_JOB'])
        # Job ID only, never a user-controlled arbitrary path.
        assert str(path).isdigit()
        directory = self.here.parent/('runs/bridge-budget-preflight-'+str(path))
        prior = self.validate(directory)
        for key in ('input_identity_sha256', 'token_cache_key_sha256', 'initial_shared_sha256',
                    'augmented_initial_sha256', 'torch_version', 'input_row_checks', 'episode_payload_checks'):
            assert report[key] == prior[key], key
        assert report['budget']['baseline_log_sha256'] == prior['budget']['baseline_log_sha256']

    def boundary(self, x):
        import torch
        r = x['r']
        actual = []
        for name in ('common_updates.jsonl', 'reduced_updates.jsonl'):
            actual.extend(json.loads(line) for line in (x['args'].output/name).read_text().splitlines())
        assert len(actual) == 8192
        assert all(a['update'] == b['update'] and a['learning_rate'] == b['learning_rate'] for a, b in zip(actual, self.old_logs))
        loss_delta = max(abs(a['loss']-b['loss']) for a, b in zip(actual, self.old_logs))
        proof = dict(loss_max_abs=loss_delta, passed=False)
        r['budget']['prefix8192_replay'] = proof
        assert loss_delta <= self.p['tolerances']['loss'], proof
        checkpoint = self.origin/'final_trainable.pt'
        assert sha(checkpoint) == self.p['baseline_checkpoint_sha256']
        old = torch.load(checkpoint, map_location='cpu', weights_only=False)['trainable_state_dict']
        now = self.ns['trainable_copy'](x['model'])
        assert old.keys() == now.keys()
        delta = max(float(torch.max(torch.abs(now[k]-old[k]))) for k in old)
        proof.update(parameter_max_abs=delta, parameters_exact=all(torch.equal(now[k], old[k]) for k in old))
        del old, now
        assert delta <= self.p['tolerances']['parameters'], proof
        self.reference_replay(x, 'prefix8192')
        proof['passed'] = True
        self.save_state(x, 'training_state_u08192.pt', 8192)
        print('BUDGET PREFIX8192 VERIFIED; continuing7978 updates', json.dumps(proof), flush=True)

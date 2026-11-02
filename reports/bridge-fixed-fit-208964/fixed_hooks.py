"""Keep train-mode contexts and global clipping; mask only context Adam updates."""
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np


def read(p): return json.loads(Path(p).read_text())
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(8*1024**2), b''): h.update(block)
    return h.hexdigest()


class FixedHooks:
    def __init__(self, here, protocol):
        self.here, self.p = here, protocol
        self.origin = here.parent/'runs/bridge-budget-fit-208586'
        self.ns = None
        self.block = False
        self.first_decoder = None

    @staticmethod
    def context_name(name):
        return name.startswith(('adapter.', 'history_projection.', 'state_projection.'))

    def mask_updates(self, model):
        # Called AFTER original global gradient clipping, BEFORE original Adam.step.
        # Keep forward modes/dropout, RNG draws and clipping geometry unchanged.
        # grad=None makes Adam skip both parameter and moment/step updates.
        if self.block:
            for name, value in model.named_parameters():
                if self.context_name(name): value.grad = None

    def after_step(self, model, update):
        if self.first_decoder is None:
            self.first_decoder = self.ns['state_sha']({n: p.detach() for n, p in model.named_parameters()
                                                       if n.startswith('diffusion_decoder.')})

    def context_hash(self, model):
        return self.ns['state_sha']({n: p.detach() for n, p in model.state_dict().items() if self.context_name(n)})

    def context_adam_hash(self, model, optimizer):
        return self.ns['state_sha']({n: optimizer.state[p] for n, p in model.named_parameters() if self.context_name(n)})

    def restore(self, x):
        import torch
        value = torch.load(self.origin/'training_state_u08192.pt', map_location='cpu', weights_only=False)
        assert value['update'] == 8192 and value['actual_optimizer_operations'] == 8192
        assert value['budget_protocol_sha256'] == self.p['original_budget_protocol_sha256']
        assert self.ns['state_sha'](value['trainable_state_dict']) == self.p['prefix_parameters_sha256']
        assert self.ns['state_sha'](value['optimizer_state_dict']) == self.p['prefix_optimizer_sha256']
        assert set(value['trainable_state_dict']) == set(self.ns['trainable_copy'](x['model']))
        x['model'].load_state_dict(value['trainable_state_dict'], strict=False)
        x['optimizer'].load_state_dict(value['optimizer_state_dict'])
        torch.set_rng_state(value['torch_cpu_rng']); torch.cuda.set_rng_state(value['torch_cuda_rng'])
        np.random.set_state(value['numpy_rng'])
        assert self.ns['state_sha'](self.ns['trainable_copy'](x['model'])) == self.p['prefix_parameters_sha256']
        assert self.ns['state_sha'](x['optimizer'].state_dict()) == self.p['prefix_optimizer_sha256']
        assert len(x['optimizer'].param_groups) == 1 and x['optimizer'].param_groups[0]['lr'] == 1e-5
        self.first_decoder = None
        del value

    def assert_frozen(self, x, context_before, adam_before):
        import torch
        model = x['model']
        assert self.context_hash(model) == context_before, 'Context parameters/buffers changed'
        assert self.context_adam_hash(model, x['optimizer']) == adam_before, 'Masked Adam state changed'
        assert x['clip_hash'] == [self.ns['digest'](model.vision_encoder), self.ns['digest'](model.text_encoder)]
        assert all(torch.equal(model.state_dict()[n].cpu(), v) for n, v in x['initial'].items()
                   if not n.startswith(self.ns['PREFIXES']))

    def replay(self, x):
        import torch
        ids = list(range(3624, 3628))
        with self.ns['observation_guard'](x['model']):
            ctx = torch.cat([x['context'](torch.tensor(ids[i:i+2], device='cuda')) for i in range(0, 4, 2)])
            truth = x['targets'][ids].cpu().numpy()
            pred, _ = self.ns['evaluate'](x['model'], ctx, x['current'][ids], truth, [1101, 1102, 1103], batch=2)
        origin = self.here.parent/'runs/bridge-demo806-fit-208207/reduced_u8192_development.npz'
        assert sha(origin) == self.p['old_development_sha256']
        with np.load(origin, allow_pickle=False) as z:
            assert np.array_equal(truth, z['targets'][:4])
            c = float(np.max(abs(ctx.cpu().numpy()-z['contexts'][:4])))
            d = float(np.max(abs(pred-z['predictions'][:, :4])))
        proof = dict(context_max_abs=c, prediction_max_abs=d,
                     passed=c <= self.p['tolerances']['context'] and d <= self.p['tolerances']['prediction'])
        x['r']['fixed_context']['prefix_replay'] = proof
        np.savez_compressed(x['args'].output/'prefix_replay.npz', contexts=ctx.cpu().numpy(), predictions=pred, targets=truth)
        print('FIXED_PREFIX_REPLAY', json.dumps(proof), flush=True)
        assert proof['passed'], proof

    def verify_origin(self, x):
        import torch
        assert sha(self.origin/'report.json') == self.p['joint_report_sha256']
        old = read(self.origin/'report.json')
        assert old['execution_checks_passed'] and old['optimization_operations'] == 16170
        for key in ('input_identity_sha256', 'token_cache_key_sha256', 'initial_shared_sha256',
                    'augmented_initial_sha256', 'torch_version', 'input_row_checks', 'episode_payload_checks'):
            assert x['r'][key] == old[key], key
        assert x['r']['full_token_cache_reused'] and x['r']['new_token_rows_encoded'] == x['r']['old_token_rows_reencoded'] == 0
        assert sha(self.origin/'training_state_u08192.pt') == self.p['prefix_file_sha256']
        schedule = self.here.parent/'bridge_demo806_budget_v1/schedule.npz'
        assert sha(schedule) == self.p['schedule_sha256']
        with np.load(schedule, allow_pickle=False) as z: b = z['batches'].copy()
        assert b.shape == (16170, 8) and np.array_equal(b[:8192], x['batches'].numpy())
        assert hashlib.sha256(b.tobytes()).hexdigest() == self.p['schedule_tensor_sha256']
        # advance closes over original batches; replace its captured tensor, not locals().
        cell = dict(zip(x['advance'].__code__.co_freevars, x['advance'].__closure__))['batches']
        cell.cell_contents = torch.from_numpy(b)
        log = self.origin/'extension_updates.jsonl'
        assert sha(log) == self.p['joint_extension_log_sha256']
        self.reference_logs = [json.loads(v) for v in log.read_text().splitlines()]
        assert [v['update'] for v in self.reference_logs] == list(range(8193, 16171))
        assert all(v['learning_rate'] == 1e-5 for v in self.reference_logs)

    def compare_logs(self, directory, label):
        actual = [json.loads(v) for v in (directory/(label+'_updates.jsonl')).read_text().splitlines()]
        expected = self.reference_logs[:len(actual)]
        differences = {}
        for field in ('loss', 'global_unclipped_gradient_norm'):
            differences[field] = max(abs(a[field]-b[field]) for a, b in zip(actual, expected))
        differences['module_gradients'] = max(abs(a['unclipped_module_gradient_norms'][k]-b['unclipped_module_gradient_norms'][k])
            for a, b in zip(actual, expected) for k in a['unclipped_module_gradient_norms'])
        assert all(v <= self.p['tolerances']['loss'] for v in differences.values()), differences
        assert all(a['update'] == b['update'] and a['learning_rate'] == b['learning_rate'] for a, b in zip(actual, expected))
        return differences

    def preflight(self, x):
        model, optimizer, r = x['model'], x['optimizer'], x['r']
        self.block = False
        control = x['advance']('joint_replay', 8193, 8196)
        differences = self.compare_logs(x['args'].output, 'joint_replay')
        first = self.first_decoder
        joint_context = self.context_hash(model)
        self.restore(x)
        before, adam_before = self.context_hash(model), self.context_adam_hash(model, optimizer)
        assert joint_context != before
        self.block = True
        masked = x['advance']('masked_probe', 8193, 8196)
        assert masked[0] == control[0] and self.first_decoder == first
        self.assert_frozen(x, before, adam_before)
        final = self.ns['state_sha'](self.ns['trainable_copy'](model))
        final_adam = self.ns['state_sha'](optimizer.state_dict())
        assert final != self.p['prefix_parameters_sha256']
        self.restore(x)
        repeated = x['advance']('masked_restored', 8193, 8196)
        assert repeated == masked
        assert self.ns['state_sha'](self.ns['trainable_copy'](model)) == final
        assert self.ns['state_sha'](optimizer.state_dict()) == final_adam
        self.assert_frozen(x, before, adam_before)
        warm = float(np.mean(x['step_seconds'][1:]))
        estimate = 900 + 1.3*(7978*warm + self.p['evaluation_allowance_seconds'])
        r['fixed_context'].update(joint_first4_replay_max_difference=differences,
            first_masked_decoder_update_matches_joint=True, full_restored_masked_continuation_exact=True,
            context_parameters_and_adam_unchanged=True, prefix_retrained=False, new_candidate_development_scored=False)
        r.update(execution_checks_passed=True, optimization_operations=12, development_scored=False,
                 clip_unchanged=True, other_frozen_unchanged=True, warm_update_seconds=warm,
                 peak_reserved_gib=__import__('torch').cuda.max_memory_reserved()/2**30,
                 estimated_formal_seconds=estimate)
        assert r['peak_reserved_gib'] < 12 and estimate < 14400
        print('FIXED CONTEXT PREFLIGHT PASSED', x['args'].output/'report.json', flush=True)

    def fit(self, x):
        import torch
        from validate_preflight import validate
        job = os.environ['FIXED_PREFLIGHT_JOB']; assert job.isdigit()
        pre = validate(self.here.parent/('runs/bridge-fixed-preflight-'+job))
        assert pre['input_identity_sha256'] == x['r']['input_identity_sha256']
        model, optimizer, r = x['model'], x['optimizer'], x['r']
        before, adam_before = self.context_hash(model), self.context_adam_hash(model, optimizer)
        self.block = True
        losses = x['advance']('fixed_context', 8193, 16170)
        assert len(losses) == 7978
        self.assert_frozen(x, before, adam_before)
        assert self.ns['state_sha'](self.ns['trainable_copy'](model)) != self.p['prefix_parameters_sha256']
        train, _ = x['check']('fixed_context', 16170, 'train')
        dev, _ = x['check']('fixed_context', 16170, 'development')
        # Frozen encoder eval contexts must equal archived8192 contexts on ALL rows.
        for split in ('train', 'development'):
            origin = self.here.parent/f'runs/bridge-demo806-fit-208207/reduced_u8192_{split}.npz'
            assert sha(origin) == self.p['old_arrays_sha256'][split]
            with np.load(origin, allow_pickle=False) as a, np.load(x['args'].output/f'fixed_context_u16170_{split}.npz', allow_pickle=False) as b:
                assert np.array_equal(a['targets'], b['targets']) and np.array_equal(a['row_indices'], b['row_indices'])
                delta = float(np.max(abs(a['contexts']-b['contexts'])))
                assert delta <= self.p['tolerances']['context'], (split, delta)
            r['fixed_context'].setdefault('all_rows_context_replay_max_abs', {})[split] = delta
        self.assert_frozen(x, before, adam_before)
        payload = dict(config=x['cfg'], trainable_state_dict=self.ns['trainable_copy'](model),
            protocol=x['p'], update=16170, resumed_from_update=8192, fixed_context_protocol=self.p,
            original_state_protocol=read(self.here.parent/'bridge_state_conditioning_v1/protocol.json'))
        path = x['args'].output/'final_trainable.pt'
        torch.save(payload, path)
        loaded = torch.load(path, map_location='cpu', weights_only=False)
        assert self.ns['state_sha'](loaded['trainable_state_dict']) == self.ns['state_sha'](payload['trainable_state_dict'])
        del loaded
        r['checkpoints'] = [dict(path=path.name, sha256=sha(path), bytes=path.stat().st_size, roundtrip_exact=True)]
        payload.update(optimizer_state_dict=self.ns['cpu_copy'](optimizer.state_dict()),
            torch_cpu_rng=torch.get_rng_state(), torch_cuda_rng=torch.cuda.get_rng_state(), numpy_rng=np.random.get_state())
        state_path = x['args'].output/'training_state_u16170.pt'
        torch.save(payload, state_path)
        assert state_path.stat().st_size < 2*1024**3
        loaded = torch.load(state_path, map_location='cpu', weights_only=False)
        assert self.ns['state_sha'](loaded['optimizer_state_dict']) == self.ns['state_sha'](payload['optimizer_state_dict'])
        assert self.ns['state_sha'](loaded['trainable_state_dict']) == self.ns['state_sha'](payload['trainable_state_dict'])
        del loaded, payload
        r['optimizer_checkpoint'] = dict(path=state_path.name, sha256=sha(state_path), bytes=state_path.stat().st_size, roundtrip_exact=True)
        r['runs'] = [dict(arm='fixed_context', final_train=train, final_development=dev, loss_first=losses[0], loss_last=losses[-1])]
        r['fixed_context'].update(context_parameters_and_adam_unchanged=True, prefix_retrained=False)
        r.update(execution_checks_passed=True, optimization_operations=7978, final_update=16170,
                 resumed_from_update=8192, development_scored=True, clip_unchanged=True, other_frozen_unchanged=True,
                 peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                 warm_update_seconds=float(np.mean(x['step_seconds'][1:])))
        print('FIXED CONTEXT FIT EXECUTION COMPLETE; inspect generalization metrics', x['args'].output/'report.json', flush=True)

    def run(self, x):
        self.verify_origin(x)
        r = x['r']
        r['fixed_context'] = dict(protocol_sha256=sha(self.here/'protocol.json'),
            source_sha256={name: sha(self.here/name) for name in self.p['local_package_sha256']},
            effective_runner_sha256=self.p['effective_runner_sha256'],
            prefix_parameters_sha256=self.p['prefix_parameters_sha256'], prefix_optimizer_sha256=self.p['prefix_optimizer_sha256'],
            training_context_dropout_preserved=True, global_clipping_before_update_mask=True,
            shadow_context_gradients_computed=True, context_update_mask_only=True)
        self.restore(x)
        self.replay(x)
        r.update(updated_parameters=sum(v.numel() for n, v in x['model'].named_parameters() if n.startswith('diffusion_decoder.')),
                 scientific_verdict='Execution success is not a generalization improvement verdict.')
        assert r['updated_parameters'] == 77670023
        if x['args'].stage == 'preflight': self.preflight(x)
        else: self.fit(x)

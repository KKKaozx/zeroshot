"""One larger training pool, matched task slots and frozen DDPM recipe."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
CODE = OVERRIDE_CODE
BASE = CODE.parent / 'bridge_growth128_v1'
for p in [CODE.parent / 'bridge_state_conditioning_v1', CODE.parent / 'bridge_inner_selection_v1', BASE]:
    sys.path.insert(0, str(p))
os.environ.update(USE_TF='0', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TF_CPP_MIN_LOG_LEVEL='2')
import numpy as np
import torch
from transformers import CLIPTokenizer
from models import RobotAdapterModel
from check_bridge_batch_control import metrics, tensor_sha, sha
from replay_and_fit_bridge_positive import digest, evaluate, seed
from trace_bridge_batch_training import observation_guard
from fit_state_arm import PREFIXES, trainable_copy, gradient_norms, cpu_copy, state_sha
from state_input_contract import add_zero_port
from expanded_inputs import load_inputs, tokens, old_replay_guard

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ['data-root', 'data-reference', 'clip', 'old-cache', 'cache', 'output', 'control-report']:
        ap.add_argument('--' + name, type=Path, required=True)
    ap.add_argument('--stage', choices=['preflight', 'fit'], default='preflight')
    ap.add_argument('--preflight-reference', type=Path)
    ap.add_argument('--local-bounded', action='store_true')
    args = ap.parse_args()
    assert args.cache.is_file(), 'Existing806 token cache required; no re-encoding'
    assert not (args.local_bounded and args.stage == 'fit')
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    p = json.loads((CODE / 'protocol.json').read_text())
    r = dict(stage=args.stage, local_bounded=args.local_bounded, execution_checks_passed=False, protocol_sha256=sha(CODE / 'protocol.json'), entry_sha256=sha(CODE / 'run_training_coverage.py'), input_helper_sha256=sha(CODE / 'expanded_inputs.py'), downloads=False, simulation_executed=False, gripper_trained=False, reserved_test_targets_read=False, old_control_retrained=False, runs=[], checkpoints=[])

    def persist():
        r['elapsed_seconds'] = time.monotonic() - started
        (args.output / 'report.json').write_text(json.dumps(r, indent=2) + '\n')
    (args.output / 'runner_snapshot.py').write_bytes(Path(__file__).read_bytes())
    try:
        for (path, h) in p['inherited_sha256'].items():
            assert sha(CODE.parent / path) == h, path
        for (name, expected) in [('rows.json', p['rows_sha256']), ('schedule.npz', p['schedule_file_sha256']), ('episode_reference.json', p['episode_reference_sha256'])]:
            assert sha(CODE / name) == expected
        assert sha(args.control_report) == p['control_report_sha256']
        control = json.loads(args.control_report.read_text())
        rows_json = json.loads((CODE / 'rows.json').read_text())
        rows = rows_json['train'] + rows_json['development']
        with np.load(CODE / 'schedule.npz') as z:
            batches = torch.from_numpy(z['batches'].copy())
        assert tensor_sha(batches) == p['schedule_tensor_sha256'] and batches.shape == (8192, 8)
        assert torch.cuda.is_available()
        torch.set_num_threads(4)
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.cuda.reset_peak_memory_stats()
        (items, features, old_items, checks, retained_indices) = load_inputs(args, CODE, p, rows, batches.numpy(), r)
        r['state_statistics_changed'] = False
        if not args.local_bounded:
            assert r['full_input_contract_passed'] and r['retained_data_checks'] == control['data_checks']
            assert r['retained_state_features_sha256'] == control['state_features_sha256']
        if args.stage == 'fit':
            assert args.preflight_reference
            pre = json.loads((args.preflight_reference / 'report.json').read_text())
            assert pre['stage'] == 'preflight' and (not pre['local_bounded']) and pre['execution_checks_passed']
            assert pre['branch_restore_passed'] and pre['full_input_contract_passed']
            for k in ['protocol_sha256', 'entry_sha256', 'input_helper_sha256', 'input_identity_sha256']:
                assert pre[k] == r[k], k
            assert pre['peak_reserved_gib'] < 12.0 and pre['estimated_formal_seconds'] < 12600
        cfg = copy.deepcopy(p['config'])
        cfg['model']['name'] = str(args.clip)
        seed(p['recipe']['initialization_seed'])
        model = RobotAdapterModel(cfg).cuda().eval()
        tokenizer = CLIPTokenizer.from_pretrained(str(args.clip), local_files_only=True)
        (visual, language, mask) = tokens(args, CODE, p, model, tokenizer, items, old_items, checks, retained_indices, r, digest)
        model.vision_encoder.cpu()
        model.text_encoder.cpu()
        torch.cuda.empty_cache()
        clip_hash = [digest(model.vision_encoder), digest(model.text_encoder)]
        model.history_projection = torch.nn.Linear(1024, 1024, bias=False).cuda()
        torch.nn.init.zeros_(model.history_projection.weight)
        for (n, v) in model.named_parameters():
            v.requires_grad_(n.startswith(PREFIXES))
        initial = {n: v.detach().cpu().clone() for (n, v) in model.state_dict().items() if not n.startswith(('vision_encoder.', 'text_encoder.'))}
        initial_hash = hashlib.sha256(''.join((n + tensor_sha(v) for (n, v) in sorted(initial.items()))).encode()).hexdigest()
        add_zero_port(model)
        initial['state_projection.weight'] = model.state_projection.weight.detach().cpu().clone()
        model.state_projection.weight.requires_grad_(True)
        initial_augmented = state_sha(trainable_copy(model))
        if not args.local_bounded:
            assert initial_hash == control['initial_shared_sha256'] and initial_augmented == control['augmented_initial_sha256']
            assert str(torch.__version__) == control['torch_version']
        r.update(initialization_seed=p['recipe']['initialization_seed'], initial_shared_sha256=initial_hash, augmented_initial_sha256=initial_augmented, torch_version=str(torch.__version__), gpu=torch.cuda.get_device_name(), sampling_steps=100, sampling_seeds=[1101, 1102, 1103])
        trainables = [v for v in model.parameters() if v.requires_grad]
        assert all((n.startswith(PREFIXES) == v.requires_grad for (n, v) in model.named_parameters()))
        optimizer = torch.optim.Adam(trainables, lr=0.0001)
        targets = torch.zeros((3664, 16, 8), device='cuda')
        current = torch.zeros((3664, 1), device='cuda')
        for i in r['verified_row_indices']:
            targets[i] = items[i][3].cuda()
            current[i] = items[i][2].cuda()
        state = torch.from_numpy(features).cuda()
        valid = set(r['verified_row_indices'])

        def context(indices):
            assert set(indices.detach().cpu().tolist()) <= valid
            now = model.adapter(visual[indices], language[indices], mask[indices])
            repeated = model.adapter(visual[indices], language[indices], mask[indices])
            return now + model.history_projection(repeated - now) + model.state_projection(state[indices])
        check_seconds = []
        step_seconds = []

        def check(arm, update, split):
            ids = ([0, 32, 64, 96] if args.stage == 'preflight' else list(range(3624))) if split == 'train' else list(range(3624, 3664))
            begin = time.monotonic()
            with observation_guard(model):
                ctx = torch.cat([context(torch.tensor(ids[i:i + 2], device='cuda')) for i in range(0, len(ids), 2)])
                truth = targets[ids].cpu().numpy()
                (pred, fit) = evaluate(model, ctx, current[ids], truth, [1101, 1102, 1103], batch=2)
                m = metrics(pred, truth, [rows[i] for i in ids])
                m['fit_gate'] = fit
            check_seconds.append((time.monotonic() - begin) / len(ids))
            np.savez_compressed(args.output / f'{arm}_u{update:04d}_{split}.npz', predictions=pred, targets=truth, contexts=ctx.cpu().numpy(), row_indices=np.array(ids))
            print('COVERAGE_METRIC', arm, update, split, 'position_cm', m['path_position_cm'], 'rotation_deg', m['path_rotation_deg'], flush=True)
            return (m, pred)

        def advance(arm, first, last):
            losses = []
            with (args.output / f'{arm}_updates.jsonl').open('w') as log:
                for update in range(first, last + 1):
                    begin = time.monotonic()
                    model.train()
                    ids = batches[update - 1].cuda()
                    seed(430000 + update)
                    ctx = context(ids)
                    seed(420000 + update)
                    (guess, truth, _) = model.diffusion_loss(targets[ids], ctx, current[ids])
                    loss = torch.nn.functional.mse_loss(guess, truth)
                    assert torch.isfinite(loss)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    grads = gradient_norms(model)
                    assert all((v.grad is None for v in model.vision_encoder.parameters()))
                    assert all((v.grad is None for v in model.text_encoder.parameters()))
                    norm = torch.nn.utils.clip_grad_norm_(trainables, 1.0, error_if_nonfinite=True)
                    FIXED.mask_updates(model)
                    optimizer.step()
                    FIXED.after_step(model, update)
                    torch.cuda.synchronize()
                    step_seconds.append(time.monotonic() - begin)
                    value = float(loss)
                    losses.append(value)
                    log.write(json.dumps(dict(update=update, learning_rate=optimizer.param_groups[0]['lr'], loss=value, global_unclipped_gradient_norm=float(norm), unclipped_module_gradient_norms=grads)) + '\n')
                    if update == first or update % 256 == 0:
                        log.flush()
                        print('COVERAGE_UPDATE', arm, update, '/', last, 'loss', value, flush=True)
            return losses
        FIXED.run(locals())
        return
    except Exception as error:
        r['error'] = repr(error)
        raise
    finally:
        persist()
        print('TRAINING_COVERAGE_REPORT', args.output / 'report.json', flush=True)
if __name__ == '__main__':
    main()

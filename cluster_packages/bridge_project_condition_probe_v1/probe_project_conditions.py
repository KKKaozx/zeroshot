"""Frozen 202278 real-state context-to-action diagnostic, native DDPM100 only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.update(USE_TF='0', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
import numpy as np
import torch
from probe_helpers import STEPS, sha, rng_state, restore_rng, rng_equal, replace_rows, sample_with_trace, pose_metrics


def tensor_sha(t):
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def state_digest(model):
    return hashlib.sha256(''.join(n + tensor_sha(v) for n, v in sorted(model.state_dict().items())).encode()).hexdigest()


def seed(n):
    np.random.seed(n)
    torch.manual_seed(n)
    torch.cuda.manual_seed_all(n)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    package = Path(__file__).resolve().parent
    protocol = json.loads((package / 'protocol.json').read_text())
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    report = dict(stage='frozen_project_context_probe', execution_passed=False,
                  updates=0, trained=False, robot_controlled=False,
                  protocol_sha256=sha(package / 'protocol.json'), error=None)
    started = time.monotonic()
    try:
        assert torch.cuda.is_available()
        torch.set_num_threads(4)
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        for name, expected in protocol['required_file_sha256'].items():
            assert sha(args.project / name) == expected, 'Changed frozen input: ' + name
        source = args.project / 'bridge_growth128_v1'
        sys.path.insert(0, str(source))
        from models import RobotAdapterModel
        run = args.project / protocol['run_directory']
        payload = torch.load(run / protocol['checkpoint_name'], map_location='cpu', weights_only=False)
        assert payload['stage'] == 'late_lr_trainable_only' and payload['state_arm'] == 'real'
        assert payload['update'] == 8192 and payload['initialization_seed'] == 43
        assert payload['optimizer_state_dict'] is None
        for name, expected in payload['source_sha256'].items():
            assert sha(source / name) == expected
        assert payload['config'] == protocol['runtime_config']
        assert payload['manifest_sha256'] == protocol['manifest_sha256']
        assert torch.__version__ == protocol['torch_version']
        seed(43)
        model = RobotAdapterModel(payload['config']).cuda().eval()
        # Recreate only the original modules/weights. Cached contexts already
        # include Adapter, zero history contribution, and real-state port.
        # Never run CLIP, Adapter, or state_projection on these contexts again.
        model.vision_encoder.cpu()
        model.text_encoder.cpu()
        model.history_projection = torch.nn.Linear(1024, 1024, bias=False).cuda()
        torch.nn.init.zeros_(model.history_projection.weight)
        initial = {n: v for n, v in model.state_dict().items()
                   if not n.startswith(('vision_encoder.', 'text_encoder.'))}
        initial_hash = hashlib.sha256(''.join(n + tensor_sha(v) for n, v in sorted(initial.items())).encode()).hexdigest()
        assert initial_hash == payload['frozen_nonclip_initial_sha256'], 'Frozen initialization differs'
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            model.state_projection = torch.nn.Linear(9, 1024, bias=False).cuda()
            torch.nn.init.zeros_(model.state_projection.weight)
        prefixes = ('adapter.', 'diffusion_decoder.', 'history_projection.', 'state_projection.')
        expected_keys = {n for n, p in model.named_parameters() if n.startswith(prefixes)}
        assert set(payload['trainable_state_dict']) == expected_keys
        incompat = model.load_state_dict(payload['trainable_state_dict'], strict=False)
        assert not incompat.unexpected_keys and not any(n in expected_keys for n in incompat.missing_keys)
        for n, v in payload['trainable_state_dict'].items():
            assert torch.equal(model.state_dict()[n].cpu(), v)
        del payload, initial
        model.requires_grad_(False)
        model.eval()
        assert model.num_diffusion_steps == 100 and model.diffusion_prediction_type == 'sample'
        assert model.clip_denoised and model.max_normalized_position == 3.0
        before = state_digest(model)
        device = torch.device('cuda', torch.cuda.current_device())
        caches = []
        for split, count, offset in [('train', 128, 0), ('development', 40, 128)]:
            with np.load(run / ('reduced_u8192_' + split + '.npz'), allow_pickle=False) as z:
                cache = {k: z[k].copy() for k in z.files}
            assert cache['contexts'].shape == (count, 1024)
            assert cache['targets'].shape == (count, 16, 8)
            assert np.array_equal(cache['row_indices'], np.arange(offset, offset + count))
            caches.append(cache)
        train, dev = caches
        all_context = torch.from_numpy(np.concatenate([train['contexts'], dev['contexts']])).to(device)
        with np.load(package / 'currents.npz') as z:
            current_array = z['currents'].copy()
        assert current_array.shape == (168, 1)
        all_current = torch.from_numpy(current_array).to(device)
        for i, row in enumerate(protocol['current_checks']):
            assert hashlib.sha256(current_array[i].tobytes()).hexdigest() == row
        windows = np.array([p['window'] for p in protocol['pairs']])
        donors = np.array([p['donor_window'] for p in protocol['pairs']])
        assert windows.min() >= 128 and windows.max() < 168 and donors.max() < 128
        original, donor_final, original_raw, donor_raw, original_clean, donor_clean, full = [], [], [], [], [], [], []
        hook_max, replay_max, untouched_max, calls = 0., 0., 0., 0
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            # Complete the original40-window gate for all3 draws BEFORE any
            # intervention. A failure stops this job and preserves a failure report.
            for seed_index, sample_seed in enumerate(protocol['sampling_seeds']):
                seed(sample_seed)
                rows = [model.sample(all_context[start:start + 2], all_current[start:start + 2]).cpu().numpy()
                        for start in range(128, 168, 2)]
                calls += 20
                delta = float(np.max(np.abs(np.concatenate(rows) - dev['predictions'][seed_index])))
                replay_max = max(replay_max, delta)
                assert delta <= protocol['replay_tolerance'], 'Original40 predictions did not replay; intervention stopped'
                print('PROJECT_ORIGINAL_REPLAY_GATE', sample_seed, '40/40', 'max', delta, flush=True)
            for sample_seed in protocol['sampling_seeds']:
                seed(sample_seed)
                baseline_rows = []
                of = np.empty((9, 16, 8), dtype=np.float32)
                df = np.empty_like(of)
                oraw = np.empty((5, 9, 16, 7), dtype=np.float32)
                draw = np.empty_like(oraw); oc = np.empty_like(oraw); dc = np.empty_like(oraw)
                for start in range(128, 168, 2):
                    context, current = all_context[start:start + 2], all_current[start:start + 2]
                    before_rng = rng_state(device)
                    baseline = model.sample(context, current)
                    after_rng = rng_state(device)
                    calls += 1
                    baseline_np = baseline.cpu().numpy()
                    baseline_rows.append(baseline_np.copy())
                    selected = [j for j, w in enumerate(windows) if start <= w < start + 2]
                    if not selected:
                        continue
                    slots = [int(windows[j] - start) for j in selected]
                    donor_indices = [int(donors[j]) for j in selected]
                    restore_rng(before_rng, device)
                    tracked, ot = sample_with_trace(model, context, current)
                    calls += 1
                    hook_max = max(hook_max, float((tracked - baseline).abs().max()))
                    assert torch.equal(tracked, baseline), 'Trace observer changes original sampler'
                    assert rng_equal(rng_state(device), after_rng), 'Trace observer consumes RNG'
                    swapped_c, swapped_g = replace_rows(context, current, slots, donor_indices, all_context, all_current)
                    restore_rng(before_rng, device)
                    swapped, dt = sample_with_trace(model, swapped_c, swapped_g)
                    calls += 1
                    assert np.array_equal(ot['initial_noise'], dt['initial_noise'])
                    assert rng_equal(rng_state(device), after_rng), 'Paired paths use different random streams'
                    restore_rng(after_rng, device)
                    untouched = [i for i in range(2) if i not in slots]
                    if untouched:
                        untouched_max = max(untouched_max, float((swapped[untouched] - baseline[untouched]).abs().max()))
                        assert untouched_max <= protocol['replay_tolerance']
                    for j, slot in zip(selected, slots):
                        of[j] = baseline_np[slot]; df[j] = swapped[slot].cpu().numpy()
                        for k, step in enumerate(STEPS):
                            oraw[k, j] = ot[step]['raw'][slot]; draw[k, j] = dt[step]['raw'][slot]
                            oc[k, j] = ot[step]['clean'][slot]; dc[k, j] = dt[step]['clean'][slot]
                replay = np.concatenate(baseline_rows)
                expected = dev['predictions'][len(full)]
                delta = float(np.max(np.abs(replay - expected)))
                replay_max = max(replay_max, delta)
                assert delta <= protocol['replay_tolerance'], 'Original40 predictions did not replay'
                full.append(replay); original.append(of); donor_final.append(df)
                original_raw.append(oraw); donor_raw.append(draw)
                original_clean.append(oc); donor_clean.append(dc)
                print('PROJECT_CONDITION_REPLAY', sample_seed, '40/40', 'max', delta, flush=True)
        arrays = dict(full40_replay=np.stack(full), original=np.stack(original), donor=np.stack(donor_final),
                      original_raw_x0=np.stack(original_raw), donor_raw_x0=np.stack(donor_raw),
                      original_clean_x0=np.stack(original_clean), donor_clean_x0=np.stack(donor_clean),
                      original_targets=dev['targets'][windows - 128], donor_targets=train['targets'][donors],
                      windows=windows, donors=donors, trace_steps=np.array(STEPS),
                      original_contexts=all_context[windows].cpu().numpy(), donor_contexts=all_context[donors].cpu().numpy())
        report.update(replay_max_absolute_difference=replay_max, observer_max_absolute_difference=hook_max,
                      untouched_max_absolute_difference=untouched_max, paired_rng_and_initial_noise_equal=True,
                      frozen_initialization_verified=True, trainable_state_loaded_exactly=True,
                      original_final=pose_metrics(arrays['original'], arrays['original_targets']),
                      donor_final=pose_metrics(arrays['donor'], arrays['donor_targets']),
                      paired_final_change=pose_metrics(arrays['original'], arrays['donor']),
                      expert_pair_difference=pose_metrics(arrays['original_targets'], arrays['donor_targets']), stages={})
        for k, step in enumerate(STEPS):
            row = {}
            for kind in ('raw', 'clean'):
                a, b = arrays['original_' + kind + '_x0'][:, k], arrays['donor_' + kind + '_x0'][:, k]
                row[kind] = dict(original=pose_metrics(a, arrays['original_targets']),
                                 donor=pose_metrics(b, arrays['donor_targets']), paired_change=pose_metrics(a, b))
            report['stages'][str(step)] = row
            print('PROJECT_CONDITION_STAGE', step, json.dumps(row['clean']), flush=True)
        report['window_metrics'] = []
        for j, pair in enumerate(protocol['pairs']):
            report['window_metrics'].append(dict(pair, original=pose_metrics(arrays['original'][:, j], arrays['original_targets'][j]),
                donor=pose_metrics(arrays['donor'][:, j], arrays['donor_targets'][j]),
                paired_change=pose_metrics(arrays['original'][:, j], arrays['donor'][:, j]),
                expert_difference=pose_metrics(arrays['original_targets'][j], arrays['donor_targets'][j])))
        assert state_digest(model) == before, 'Frozen weights/buffers changed'
        assert calls == protocol['expected_sampler_calls']
        report.update(parameters_and_buffers_unchanged=True, execution_passed=True,
                      torch_version=torch.__version__, gpu=torch.cuda.get_device_name(),
                      native_sampler_calls=calls, peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                      scope='Saved fused-context boundary only; no separate image/state or Adapter causality claim. Intermediate x0 is an estimate, not executed action. Donor changes observation and truth, not a generalization fix.')
        np.savez_compressed(args.output / 'predictions.npz', **arrays)
        assert (args.output / 'predictions.npz').stat().st_size < protocol['output_limit_mib'] * 1024**2
    except Exception as error:
        report['error'] = repr(error)
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic() - started
        (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        print('FROZEN PROJECT CONDITION REPORT', args.output / 'report.json', flush=True)


if __name__ == '__main__':
    main()

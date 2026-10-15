"""Independent saved-array audit of frozen202278 context diagnostic204017.

CPU only; no checkpoint, CLIP or action sampling; no optimization.
"""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'reports/bridge-project-condition-204017'
PACK = ROOT / 'cluster_packages/bridge_project_condition_probe_v1'
OUT = ROOT / 'reports/bridge_project_condition_analysis_204017'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def errors(pred, target):
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    distance = np.sqrt(np.square(p[..., :3] - t[..., :3]).sum(-1)) * 10
    q, tq = p[..., 3:7].copy(), t[..., 3:7].copy()
    length = np.sqrt(np.square(q).sum(-1, keepdims=True))
    bad = length[..., 0] <= 1e-6
    q /= np.maximum(length, 1e-8); q[bad] = [0, 0, 0, 1]
    tq /= np.maximum(np.sqrt(np.square(tq).sum(-1, keepdims=True)), 1e-8)
    angle = np.degrees(2 * np.arccos(np.clip(np.abs((q * tq).sum(-1)), 0, 1)))
    return distance, angle, int(bad.sum())


def metrics(pred, target):
    p, r, bad = errors(pred, target)
    return dict(position_cm=float(p.mean()), rotation_deg=float(r.mean()),
                endpoint_position_cm=float(p[..., -1].mean()), endpoint_rotation_deg=float(r[..., -1].mean()),
                degenerate_quaternions=bad)


def equal_metrics(a, b):
    assert a.keys() == b.keys()
    for k in a:
        assert abs(a[k] - b[k]) < 1e-8, (k, a[k], b[k])


def main():
    OUT.mkdir(exist_ok=True)
    r, protocol = read(DATA / 'report.json'), read(PACK / 'protocol.json')
    assert r['execution_passed'] and r['updates'] == 0 and not r['trained'] and not r['robot_controlled']
    assert r['error'] is None and r['protocol_sha256'] == sha(PACK / 'protocol.json')
    for k in ['paired_rng_and_initial_noise_equal', 'frozen_initialization_verified',
              'trainable_state_loaded_exactly', 'parameters_and_buffers_unchanged']:
        assert r[k]
    assert r['native_sampler_calls'] == protocol['expected_sampler_calls'] == 150
    assert r['observer_max_absolute_difference'] == r['untouched_max_absolute_difference'] == 0
    assert r['peak_reserved_gib'] < 20
    for n, expected in protocol['required_file_sha256'].items():
        if n.startswith('bridge_growth128_v1/'):
            file = ROOT / 'cluster_packages' / n
        elif n.endswith('.pt'):
            continue  # Retained on cluster; file/hash verification from GPU report.
        else:
            file = ROOT / 'reports' / Path(n).relative_to('runs')
        assert sha(file) == expected
    with np.load(DATA / 'predictions.npz', allow_pickle=False) as z:
        arrays = {n: z[n].copy() for n in z.files}
    names = {'full40_replay', 'original', 'donor', 'original_raw_x0', 'donor_raw_x0',
             'original_clean_x0', 'donor_clean_x0', 'original_targets', 'donor_targets',
             'windows', 'donors', 'trace_steps', 'original_contexts', 'donor_contexts'}
    assert set(arrays) == names and all(np.isfinite(v).all() for v in arrays.values())
    windows = np.array([p['window'] for p in protocol['pairs']])
    donors = np.array([p['donor_window'] for p in protocol['pairs']])
    assert np.array_equal(arrays['windows'], windows) and np.array_equal(arrays['donors'], donors)
    assert np.array_equal(arrays['trace_steps'], protocol['trace_steps'])
    assert arrays['full40_replay'].shape == (3, 40, 16, 8)
    assert all(arrays[n].shape == (3, 9, 16, 8) for n in ['original', 'donor'])
    assert all(arrays[n].shape == (3, 5, 9, 16, 7) for n in names if n.endswith('_x0'))
    caches = []
    run = ROOT / 'reports/bridge-state-fit-202278/real'
    for split, count, offset in [('train', 128, 0), ('development', 40, 128)]:
        with np.load(run / ('reduced_u8192_' + split + '.npz')) as z:
            caches.append({k: z[k].copy() for k in ['predictions', 'contexts', 'targets']})
            assert np.array_equal(z['row_indices'], np.arange(offset, offset + count))
    train, dev = caches
    replay = float(np.max(np.abs(arrays['full40_replay'] - dev['predictions'])))
    assert replay == r['replay_max_absolute_difference'] == 0
    assert np.array_equal(arrays['original'], arrays['full40_replay'][:, windows - 128])
    assert np.array_equal(arrays['original_targets'], dev['targets'][windows - 128])
    assert np.array_equal(arrays['donor_targets'], train['targets'][donors])
    assert np.array_equal(arrays['original_contexts'], dev['contexts'][windows - 128])
    assert np.array_equal(arrays['donor_contexts'], train['contexts'][donors])
    summaries = {}
    for key, pred, target in [('original_final', 'original', 'original_targets'), ('donor_final', 'donor', 'donor_targets'),
                              ('paired_final_change', 'original', 'donor'), ('expert_pair_difference', 'original_targets', 'donor_targets')]:
        summaries[key] = metrics(arrays[pred], arrays[target]); equal_metrics(summaries[key], r[key])
    for k, step in enumerate(arrays['trace_steps']):
        for kind in ['raw', 'clean']:
            a, b = arrays['original_' + kind + '_x0'][:, k], arrays['donor_' + kind + '_x0'][:, k]
            got = dict(original=metrics(a, arrays['original_targets']), donor=metrics(b, arrays['donor_targets']), paired_change=metrics(a, b))
            for arm in got:
                equal_metrics(got[arm], r['stages'][str(step)][kind][arm])
    rows = read(ROOT / 'cluster_packages/bridge_growth128_v1/manifest.json')
    rows = rows['train'] + rows['development']
    per_window = []
    for j, pair in enumerate(protocol['pairs']):
        saved = r['window_metrics'][j]
        assert all(saved[k] == pair[k] for k in pair)
        item = dict(pair, episode=rows[pair['window']]['shard'] + '::' + str(rows[pair['window']]['record_index']),
                    input_index=rows[pair['window']]['start_index'], clean_stages={}, raw_stages={})
        for kind in ['raw', 'clean']:
            for k, step in enumerate(arrays['trace_steps']):
                item[kind + '_stages'][str(step)] = dict(original=metrics(arrays['original_' + kind + '_x0'][:, k, j], arrays['original_targets'][j]),
                                                       donor=metrics(arrays['donor_' + kind + '_x0'][:, k, j], arrays['donor_targets'][j]))
        for field, pred, target in [('original', 'original', 'original_targets'), ('donor', 'donor', 'donor_targets'),
                                   ('paired_change', 'original', 'donor')]:
            target_row = arrays[target][:, j] if arrays[target].ndim == 4 else arrays[target][j]
            item[field] = metrics(arrays[pred][:, j], target_row); equal_metrics(item[field], saved[field])
        item['expert_difference'] = metrics(arrays['original_targets'][j], arrays['donor_targets'][j])
        equal_metrics(item['expert_difference'], saved['expert_difference'])
        c, d = arrays['original_contexts'][j].astype(float), arrays['donor_contexts'][j].astype(float)
        item['context_cosine'] = float(c @ d / np.linalg.norm(c) / np.linalg.norm(d))
        item['context_exact_equal'] = bool(np.array_equal(c, d))
        item['first_to_final_error_change_cm'] = item['original']['position_cm'] - item['clean_stages']['99']['original']['position_cm']
        item['first_error_fraction_of_final'] = item['clean_stages']['99']['original']['position_cm'] / item['original']['position_cm']
        per_window.append(item)
    episode_groups = []
    for episode in dict.fromkeys(row['episode'] for row in per_window):
        ix = [j for j, row in enumerate(per_window) if row['episode'] == episode]
        original_targets = arrays['original_targets'][ix]
        static = np.zeros_like(original_targets); static[..., 6] = 1
        episode_groups.append(dict(episode=episode, windows=windows[ix].tolist(),
            first=metrics(arrays['original_clean_x0'][:, 0, ix], original_targets),
            final=metrics(arrays['original'][:, ix], original_targets), static=metrics(static, original_targets)))
    first = errors(arrays['original_clean_x0'][:, 0], arrays['original_targets'])[0].mean(-1)
    last = errors(arrays['original'], arrays['original_targets'])[0].mean(-1)
    train_stage = errors(arrays['donor_clean_x0'], arrays['donor_targets'])[0].mean(-1)
    # Raw/clean differences can include native float32 conversion and clipping;
    # absent intermediate noisy inputs, do not reconstruct the transformation.
    transformations = {str(step): dict(
        original_max_raw_clean_xyz_difference_cm=float(np.max(np.abs(arrays['original_raw_x0'][:, k, ..., :3] - arrays['original_clean_x0'][:, k, ..., :3])) * 10),
        original_raw_xyz_outside_limit=int((np.abs(arrays['original_raw_x0'][:, k, ..., :3]) > 3).sum()),
        original_raw_quaternion_components_outside_limit=int((np.abs(arrays['original_raw_x0'][:, k, ..., 3:]) > 1).sum()))
        for k, step in enumerate(arrays['trace_steps'])}
    result = dict(verified=True, independent_cpu_recomputation=True, trained=False, model_loaded=False,
        execution_report_checks_passed=True, replay_max_absolute_difference=replay, metrics_match_report=True,
        original_targets_and_contexts_exact=True, donor_targets_and_contexts_exact=True,
        protocol_sha256=sha(PACK / 'protocol.json'), artifact_sha256={n: sha(DATA / n) for n in ['report.json', 'predictions.npz']},
        artifact_bytes=sum((DATA / n).stat().st_size for n in ['report.json', 'predictions.npz']),
        stage_metrics=r['stages'], summaries=summaries, per_window=per_window, per_episode=episode_groups,
        original_first_error_range_cm=[float(first.mean(0).min()), float(first.mean(0).max())],
        original_final_error_range_cm=[float(last.mean(0).min()), float(last.mean(0).max())],
        windows_first_error_at_least_90_percent_of_final=int((first.mean(0) / last.mean(0) >= .9).sum()),
        max_absolute_per_window_first_to_final_error_change_cm=float(np.abs(last.mean(0) - first.mean(0)).max()),
        donor_largest_window_draw_stage_error_mm=float(train_stage.max() * 10),
        donor_final_all_draws_windows_under_5mm=bool((errors(arrays['donor'], arrays['donor_targets'])[0].mean(-1) <= .5).all()),
        per_draw_first_error_cm=first.mean(1).tolist(), per_draw_final_error_cm=last.mean(1).tolist(),
        raw_clean_transform_diagnostics=transformations,
        protection_limits='RNG/initial noise, unused-row and parameter/buffer protection verified from GPU report plus frozen code/CPU preparation; no unused-row intervention outputs or intermediate noisy states retained for separate reconstruction',
        scientific_limits='Nine overlapping windows from two held-out broccoli episodes; seven unique train donors. No untouched test, task-success verdict or uniquely identified faulty layer.')
    (OUT / 'analysis.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k not in ['stage_metrics', 'per_window', 'artifact_sha256', 'summaries', 'raw_clean_transform_diagnostics']}, indent=2))
    print('WINDOW SUMMARY')
    for row in per_window:
        print(row['window'], 'first', round(row['clean_stages']['99']['original']['position_cm'], 3),
              'final', round(row['original']['position_cm'], 3), 'change', round(row['first_to_final_error_change_cm'], 3))


if __name__ == '__main__':
    main()

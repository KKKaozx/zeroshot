"""CPU-only audit of saved208056 artifacts; never imports the model or runs inference."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'reports/bridge-heldout-fixed-208056'
BASE = ROOT / 'cluster_packages/bridge_heldout_evaluation_v1'
REPAIR = ROOT / 'cluster_packages/bridge_heldout_context_repair_v1'
OUT = ROOT / 'reports/bridge_heldout_analysis_208056'
NUMERIC_DIFFERENCES = []


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def errors(pred, target):
    # One XYZ label unit represents0.1m, or10cm. Quaternions are xyzw.
    a = np.asarray(pred, dtype=np.float64)
    b = np.asarray(target, dtype=np.float64)
    assert np.isfinite(a).all() and np.isfinite(b).all()
    position = np.sqrt(np.sum((a[..., :3] - b[..., :3]) ** 2, axis=-1)) * 10
    qa, qb = a[..., 3:7], b[..., 3:7]
    na, nb = np.linalg.norm(qa, axis=-1), np.linalg.norm(qb, axis=-1)
    assert na.min() > 1e-8 and nb.min() > 1e-8
    dot = np.clip(np.abs(np.sum(qa * qb, axis=-1) / (na * nb)), 0, 1)
    # atan2 is an independent equivalent of the runner's acos implementation.
    rotation = np.rad2deg(2 * np.arctan2(np.sqrt(np.maximum(0, 1-dot*dot)), dot))
    return position, rotation


def summarize(pred, target, rows):
    position, rotation = errors(pred, target)

    def mean(indices):
        p, r = position[:, indices], rotation[:, indices]
        return dict(path_position_cm=float(p.mean()), path_rotation_deg=float(r.mean()),
                    endpoint_position_cm=float(p[..., -1].mean()),
                    endpoint_rotation_deg=float(r[..., -1].mean()))

    episodes = {}
    for i, row in enumerate(rows):
        episodes.setdefault(row['shard']+'::'+str(row['record_index']), []).append(i)
    per_episode = [dict(episode=key, task=rows[ix[0]]['task'], windows=len(ix), **mean(ix))
                   for key, ix in episodes.items()]
    per_task = []
    for task in sorted({row['task'] for row in rows}):
        ix = [i for i, row in enumerate(rows) if row['task'] == task]
        per_task.append(dict(task=task, windows=len(ix), **mean(ix)))
    metrics = ('path_position_cm', 'path_rotation_deg', 'endpoint_position_cm', 'endpoint_rotation_deg')
    return dict(windows=len(rows), episodes=len(episodes),
                window_weighted=mean(list(range(len(rows)))),
                episode_equal={k: float(np.mean([e[k] for e in per_episode])) for k in metrics},
                per_episode=per_episode, per_task=per_task,
                per_target=[dict(target=k+1, position_cm=float(position[..., k].mean()),
                                 rotation_deg=float(rotation[..., k].mean())) for k in range(16)])


def compare(actual, expected, location=''):
    """Verify all saved summary fields against arrays, including identity/order."""
    if isinstance(actual, dict):
        for key, value in actual.items():
            compare(value, expected[key], location+'/'+key)
    elif isinstance(actual, list):
        assert len(actual) == len(expected), location
        for i, (a, b) in enumerate(zip(actual, expected)):
            compare(a, b, location+'/'+str(i))
    elif isinstance(actual, float):
        NUMERIC_DIFFERENCES.append(abs(actual-expected))
        assert abs(actual-expected) <= 1e-8, (location, actual, expected)
    else:
        assert actual == expected, (location, actual, expected)


def main():
    report = read(SOURCE/'report.json')
    protocol = read(BASE/'protocol.json')
    selection = read(BASE/'selection.json')
    repair = read(SOURCE/'repair_entry.json')
    for name in ('protocol', 'assets'):
        assert sha(SOURCE/(name+'_snapshot.json')) == sha(BASE/(name+'.json')) == report[name+'_sha256']
    assert sha(BASE/'selection.json') == protocol['selection_sha256']
    assert sha(SOURCE/'runner_snapshot.py') == sha(BASE/'run_evaluation.py') == report['entry_sha256']
    assert sha(SOURCE/'repair_runner_snapshot.py') == sha(REPAIR/'run_fixed.py') == repair['repair_entry_sha256']
    assert sha(SOURCE/'repair_contract_snapshot.json') == sha(REPAIR/'repair_contract.json') == repair['repair_contract_sha256']
    assert repair['execution_passed'] and repair['diagnostic_evidence']['verified']
    assert not any(repair[k] for k in ('new_training', 'original_protocol_changed', 'numeric_tolerance_changed'))
    for key in ('execution_checks_passed', 'input_contract_passed', 'clip_unchanged',
                'other_frozen_unchanged', 'input_state_statistics_from_original_train_only',
                'test_predictions_generated', 'test_set_consumed'):
        assert report[key], key
    for key in ('training', 'optimizer_created', 'gripper_trained', 'parameters_updated',
                'state_statistics_changed', 'downloads', 'simulation'):
        assert report[key] is False, key
    assert len(report['input_checks']) == 64 and report['raw_payloads_checked'] == 14
    for check in report['input_checks']:
        assert check['raw_exact']
        assert max(check['max_absolute_differences'].values()) <= 1e-6
    rows = selection['rows']
    assert len(rows) == 60
    with np.load(BASE/'input_reference.npz', allow_pickle=False) as z:
        local_truth = z['targets'][4:].copy()
        replay_truth = z['targets'][:4].copy()
    with np.load(SOURCE/'coverage_test_predictions.npz', allow_pickle=False) as z:
        truth = z['targets'].copy()
    target_delta = float(np.max(np.abs(truth.astype(float)-local_truth.astype(float))))
    assert target_delta <= 1e-6
    identity = np.zeros_like(truth)
    identity[..., 6] = 1
    summaries = {'static': summarize(identity[None], truth, rows)}
    compare(summaries['static'], report['static'], 'static')
    arrays = {}
    replay_differences = {}
    for model in ('coverage', 'rotation_weight'):
        with np.load(SOURCE/(model+'_test_predictions.npz'), allow_pickle=False) as z:
            pred = z['predictions'].copy()
            assert pred.shape == (3, 60, 16, 8)
            assert z['contexts'].shape == (60, 1024) and np.isfinite(z['contexts']).all()
            assert np.array_equal(z['row_indices'], np.arange(60))
            assert np.array_equal(z['targets'], truth)
            # Recompute against the exact targets used by the cluster as well.
            target = z['targets'].copy()
        arrays[model] = pred
        summaries[model] = summarize(pred, target, rows)
        compare(summaries[model], report['models'][model], model)
        with np.load(SOURCE/(model+'_replay_check.npz'), allow_pickle=False) as a, np.load(BASE/(model+'_replay.npz'), allow_pickle=False) as b:
            delta = {k: float(np.max(np.abs(a[k].astype(float)-b[k].astype(float)))) for k in a.files}
            assert all(v == 0 for v in delta.values()), (model, delta)
            assert np.max(np.abs(a['targets']-replay_truth)) <= 1e-6
            replay_differences[model] = delta
        m = summaries[model]
        advantage = all(m[g][k] < summaries['static'][g][k] for g in ('episode_equal', 'window_weighted')
                        for k in ('path_position_cm', 'path_rotation_deg'))
        assert advantage == report['models'][model]['descriptive_mean_advantage_over_static']

    episode_rows, task_rows = [], []
    counts = {}
    for model in ('coverage', 'rotation_weight'):
        counts[model] = {}
        for group, dest, identity_key in [('per_episode', episode_rows, 'episode'), ('per_task', task_rows, 'task')]:
            wins = {k: 0 for k in ('path_position_cm', 'path_rotation_deg', 'endpoint_position_cm', 'endpoint_rotation_deg')}
            both = 0
            for model_row, static_row in zip(summaries[model][group], summaries['static'][group]):
                assert model_row[identity_key] == static_row[identity_key]
                row = dict(model=model, **model_row)
                for metric in wins:
                    row['static_'+metric] = static_row[metric]
                    row['delta_'+metric] = model_row[metric]-static_row[metric]
                    wins[metric] += int(model_row[metric] < static_row[metric])
                both += int(row['delta_path_position_cm'] < 0 and row['delta_path_rotation_deg'] < 0)
                dest.append(row)
            counts[model][group] = dict(total=len(summaries[model][group]), wins=wins, both_path_metrics_better=both)

    OUT.mkdir(exist_ok=True)
    for filename, data in [('episodes.csv', episode_rows), ('tasks.csv', task_rows)]:
        with (OUT/filename).open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0]))
            writer.writeheader(); writer.writerows(data)
    result = dict(execution_checks_passed=True, job='208056', gpu_used=False, new_inference=False,
                  training=False, source_report_sha256=sha(SOURCE/'report.json'),
                  source_files={p.name: dict(bytes=p.stat().st_size, sha256=sha(p)) for p in sorted(SOURCE.iterdir()) if p.is_file()},
                  snapshots_match_local_frozen_sources=True, all_saved_metrics_recomputed=True,
                  metric_max_absolute_difference=max(NUMERIC_DIFFERENCES),
                  cluster_vs_local_target_max_absolute_difference=target_delta,
                  metrics_use_exact_cluster_saved_targets=True, sampling_seeds=protocol['sampling']['seeds'],
                  replay_differences=replay_differences, summaries=summaries, paired_counts=counts,
                  diagnostic_evidence_scope='Downloaded repair entry records cluster verification of16 old diagnostic arrays; those16 arrays were not separately downloaded or independently recomputed locally.',
                  interpretation='Descriptive same-task independent13episode performance; no significance, unseen-task, grasp-success, or test-driven tuning claim.')
    (OUT/'analysis.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(verified=True, paired_counts=counts,
                         means={k: {g:v[g] for g in ('episode_equal','window_weighted')} for k,v in summaries.items()},
                         tasks=task_rows), indent=2))


if __name__ == '__main__':
    main()

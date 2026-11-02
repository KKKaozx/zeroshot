"""CPU-only verification of saved 208207 arrays; no inference or model imports."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from analyze_bridge_heldout_208056 import errors, summarize

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / 'cluster_packages/bridge_demonstration_expansion_v1'
SOURCE = ROOT / 'reports/bridge-demo806-fit-208207'
OLD = ROOT / 'reports/bridge-coverage-rep-fit-207558'
OUT = ROOT / 'reports/bridge_demo806_analysis_208207'
METRICS = ('path_position_cm', 'path_rotation_deg', 'endpoint_position_cm', 'endpoint_rotation_deg')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path, count, first):
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k].copy() for k in z.files}
    assert arrays['predictions'].shape == (3, count, 16, 8)
    assert arrays['targets'].shape == (count, 16, 8)
    assert arrays['contexts'].shape == (count, 1024)
    assert np.array_equal(arrays['row_indices'], np.arange(first, first+count))
    assert all(np.isfinite(a).all() for a in arrays.values())
    return arrays


def static(truth):
    identity = np.zeros_like(truth)
    identity[..., 6] = 1
    return identity[None]


def main():
    r, p = read(SOURCE/'report.json'), read(PACK/'protocol.json')
    old_r = read(OLD/'report.json')
    rows = read(PACK/'rows.json')
    old_rows = read(ROOT/'cluster_packages/bridge_coverage_replication_v1/rows.json')
    contract = read(PACK/'adaptation_contract.json')
    assert len(rows['train']) == 3624 and len(rows['development']) == 40
    assert rows['train'][:1836] == old_rows['train'] and rows['development'] == old_rows['development']
    assert sha(OLD/'report.json') == p['baseline_report_sha256']
    for key, path in [('protocol_sha256', PACK/'protocol.json'),
                      ('entry_sha256', PACK/'run_training_coverage.py'),
                      ('input_helper_sha256', PACK/'expanded_inputs.py')]:
        assert r[key] == sha(path), key
    for snapshot, original, key in [
        ('runner_snapshot.py', PACK/'run_training_coverage.py', 'entry_sha256'),
        ('wrapper_snapshot.py', PACK/'run_expansion.py', 'wrapper_sha256'),
        ('adaptation_contract_snapshot.json', PACK/'adaptation_contract.json', 'contract_sha256')]:
        expected = r[key] if key == 'entry_sha256' else r['adaptation'][key]
        assert sha(SOURCE/snapshot) == sha(original) == expected
    assert sha(SOURCE/'effective_runner_snapshot.py') == contract['effective_runner_sha256'] == r['adaptation']['effective_runner_sha256']
    for file, expected in p['inherited_sha256'].items():
        assert sha(PACK.parent/file) == expected, file
    for filename, key in [('rows.json', 'rows_sha256'), ('schedule.npz', 'schedule_file_sha256'),
                          ('episode_reference.json', 'episode_reference_sha256')]:
        assert sha(PACK/filename) == p[key]
    assert r['stage'] == 'fit' and not r['local_bounded']
    for key in ('execution_checks_passed', 'full_input_contract_passed', 'retained_numeric_contract_passed',
                'old_1876_input_contract_exact', 'original_1876_token_cache_reused', 'retained_visual_tokens_exact',
                'retained_language_tokens_exact', 'full_token_cache_reused', 'clip_unchanged', 'other_frozen_unchanged', 'development_scored'):
        assert r[key], key
    for key in ('downloads', 'simulation_executed', 'gripper_trained', 'reserved_test_targets_read',
                'old_control_retrained', 'state_statistics_changed'):
        assert r[key] is False, key
    assert r['optimization_operations'] == r['final_update'] == 8192 and r['branch_update'] == 4096
    assert r['initialization_seed'] == 42 and r['sampling_seeds'] == [1101, 1102, 1103] and r['sampling_steps'] == 100
    assert r['initial_shared_sha256'] == p['expected_initial_shared_sha256'] == old_r['initial_shared_sha256']
    assert r['augmented_initial_sha256'] == p['expected_augmented_initial_sha256'] == old_r['augmented_initial_sha256']
    assert r['torch_version'] == old_r['torch_version'] == '2.5.1+cu118'
    assert r['trainable_parameters'] == old_r['trainable_parameters']
    assert r['verified_row_indices'] == list(range(3664)) and len(r['input_row_checks']) == 3664
    assert r['active_token_rows'] == 3664 and r['old_token_rows_reencoded'] == r['new_token_rows_encoded'] == 0
    mapping = list(range(1836)) + list(range(3624, 3664))
    assert r['old_1876_to_new_row_indices'] == mapping
    assert r['old_input_identity_sha256'] == old_r['input_identity_sha256']
    assert r['old_token_cache_key_sha256'] == old_r['token_cache_key_sha256']
    for old_i, new_i in enumerate(mapping):
        a, b = dict(old_r['input_row_checks'][old_i]), dict(r['input_row_checks'][new_i])
        a.pop('index'); b.pop('index'); assert a == b, (old_i, new_i)
    reference = read(PACK/'episode_reference.json')['episodes']
    identity = lambda x: (x['shard'], x['record_index'])
    assert len(r['episode_payload_checks']) == len(reference) == 816
    assert {identity(x): x['payload_sha256'] for x in reference} == {identity(x): x['payload_sha256'] for x in r['episode_payload_checks']}
    with np.load(PACK/'schedule.npz') as z, np.load(PACK.parent/'bridge_coverage_replication_v1/schedule.npz') as b:
        batches, old_batches = z['batches'], b['batches']
        tasks = np.array([x['task'] for x in rows['train']])
        old_tasks = np.array([x['task'] for x in old_rows['train']])
        assert batches.shape == (8192, 8) and np.array_equal(tasks[batches], old_tasks[old_batches])
        assert all(len(set(batch)) == 8 for batch in batches)
        assert np.bincount(batches.ravel(), minlength=3624).min() > 0
    summaries, arrays, comparisons = {}, {}, []
    for label, directory, split_rows, first in [
        ('405_train', OLD, old_rows['train'], 0), ('405_dev', OLD, old_rows['development'], 1836),
        ('806_train', SOURCE, rows['train'], 0), ('806_dev', SOURCE, rows['development'], 3624)]:
        split = 'development' if label.endswith('dev') else 'train'
        a = load(directory/f'reduced_u8192_{split}.npz', len(split_rows), first)
        arrays[label] = a
        summaries[label] = summarize(a['predictions'], a['targets'], split_rows)
        if label.startswith('806'):
            for local_i, target in enumerate(a['targets']):
                assert hashlib.sha256(target.tobytes()).hexdigest() == r['input_row_checks'][first+local_i]['target_sha256']
            saved = r['runs'][0]['final_'+split]
            summary = summaries[label]
            static_summary = summarize(static(a['targets']), a['targets'], split_rows)
            for key in METRICS:
                comparisons.append((key, abs(summary['window_weighted'][key]-saved[key])))
                comparisons.append((key, abs(static_summary['window_weighted'][key]-saved['static_'+key])))
            position, rotation = errors(a['predictions'], a['targets'])
            for values, key, saved_key in [(position, 'path_position_cm', 'seed_path_position_cm'),
                                            (rotation, 'path_rotation_deg', 'seed_path_rotation_deg')]:
                comparisons.extend((key, abs(float(v)-s)) for v, s in zip(values.mean(axis=(1, 2)), saved[saved_key]))
            for computed, recorded in zip(summary['per_episode'], saved['per_episode']):
                assert all(computed[k] == recorded[k] for k in ('episode', 'task', 'windows'))
                for key in METRICS:
                    comparisons.append((key, abs(computed[key]-recorded[key])))
            for key, saved_key in [('path_position_cm', 'episode_mean_path_position_cm'),
                                   ('path_rotation_deg', 'episode_mean_path_rotation_deg')]:
                comparisons.append((key, abs(summary['episode_equal'][key]-saved[saved_key])))
    max_difference = {kind: max(v for k, v in comparisons if kind in k) for kind in ('position', 'rotation')}
    # Independent float64 recomputation vs original float32 kernels; not a replay tolerance change.
    assert max_difference['position'] < 1e-4 and max_difference['rotation'] < 1e-3, max_difference
    assert np.array_equal(arrays['405_dev']['targets'], arrays['806_dev']['targets'])
    assert np.array_equal(arrays['405_train']['targets'], arrays['806_train']['targets'][:1836])
    summaries['806_original405'] = summarize(arrays['806_train']['predictions'][:, :1836], arrays['806_train']['targets'][:1836], old_rows['train'])
    summaries['806_added401'] = summarize(arrays['806_train']['predictions'][:, 1836:], arrays['806_train']['targets'][1836:], rows['train'][1836:])
    summaries['static_dev'] = summarize(static(arrays['806_dev']['targets']), arrays['806_dev']['targets'], rows['development'])
    paired, counts = {}, {}
    for group in ('per_episode', 'per_task'):
        paired[group] = []
        for new, old, hold in zip(summaries['806_dev'][group], summaries['405_dev'][group], summaries['static_dev'][group]):
            id_key = 'episode' if group == 'per_episode' else 'task'
            assert new[id_key] == old[id_key] == hold[id_key]
            rec = {id_key: new[id_key], 'task': new['task']} if id_key == 'episode' else {'task': new['task']}
            rec['windows'] = new['windows']
            for key in METRICS:
                rec.update({f'405_{key}': old[key], f'806_{key}': new[key], f'static_{key}': hold[key], f'delta_{key}': new[key]-old[key]})
            paired[group].append(rec)
        counts[group] = {'total': len(paired[group]), **{
            f'{key}_better_405': sum(x[f'delta_{key}'] < 0 for x in paired[group]) for key in METRICS}, **{
            f'{key}_better_static': sum(x[f'806_{key}'] < x[f'static_{key}'] for x in paired[group]) for key in METRICS}}
    gates = {f'{group}_{key}': summaries['806_dev'][group][key] <= summaries['405_dev'][group][key]
             for group in ('window_weighted', 'episode_equal') for key in METRICS[:2]}
    strict_improvement = any(summaries['806_dev'][g][k] < summaries['405_dev'][g][k]
                             for g in ('window_weighted', 'episode_equal') for k in METRICS[:2])
    passed = all(gates.values()) and strict_improvement
    result = dict(verified=True, job_id='208207', new_inference=False, training=False, gpu_used=False,
                  downloaded_files={f.name: {'bytes': f.stat().st_size, 'sha256': sha(f)} for f in sorted(SOURCE.iterdir()) if f.is_file()},
                  baseline_report_sha256=sha(OLD/'report.json'), protocol_sha256=sha(PACK/'protocol.json'),
                  max_float64_vs_report_difference=max_difference, saved_old_targets_exact=True,
                  snapshots_and_report_contracts_verified=True, saved_metrics_recomputed=True,
                  checkpoint_metadata=r['checkpoints'], checkpoint_downloaded=False,
                  runtime_integrity_evidence_scope='Frozen parameters, full cache identity and checkpoint roundtrip rely on cluster runtime assertions; no local model execution or checkpoint hash verification.',
                  summaries=summaries, paired_counts=counts, gates=gates, screen_passed=passed,
                  peak_reserved_gib=r['peak_reserved_gib'], elapsed_seconds=r['elapsed_seconds'],
                  next_action='Close fixed8192-budget expansion as mixed tradeoff; preserve405 baseline; no automatic extra training or reserved25 assessment.',
                  limitations=['One initialization; diffusion sampling seeds are not training replications.',
                               'Repeated10-episode development set; no new independent test result for806 model.',
                               'Fixed budget lowers average exposure per window; cannot identify data amount alone as cause.',
                               'Gripper frozen; offline measured pose only, not grasp success.'])
    OUT.mkdir(exist_ok=True)
    for group, filename in [('per_episode', 'episodes.csv'), ('per_task', 'tasks.csv')]:
        with (OUT/filename).open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(paired[group][0])); writer.writeheader(); writer.writerows(paired[group])
    (OUT/'analysis.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(verified=True, screen_passed=passed, gates=gates, metric_difference=max_difference,
                         means={k: {g: v[g] for g in ('window_weighted', 'episode_equal')} for k, v in summaries.items()},
                         counts=counts, tasks=paired['per_task']), indent=2))


if __name__ == '__main__':
    main()

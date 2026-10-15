"""Independent CPU audit and preregistered verdict for history pair204258.

No model/checkpoint load, sampling or optimization; both arms must be retained.
"""
from collections import defaultdict
import csv
import json
from pathlib import Path
import numpy as np
from analyze_bridge_growth64_200600 import sha, summarize
from analyze_bridge_training_trace_200039 import FIELDS

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'reports/bridge-history-fit-204258'
PACK = ROOT / 'cluster_packages/bridge_history_generalization_v1'
BASE = ROOT / 'cluster_packages/bridge_growth128_v1'
OLD = ROOT / 'reports/bridge-state-fit-202278/real'
OUT = ROOT / 'reports/bridge_history_generalization_analysis_204258'


def read(p): return json.loads(p.read_text(encoding='utf-8'))


def write_csv(name, rows):
    with (OUT / name).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def main():
    OUT.mkdir(exist_ok=True)
    protocol, manifest, pair = read(PACK / 'protocol.json'), read(BASE / 'manifest.json'), read(DATA / 'report.json')
    assert pair['passed'] and pair['stage'] == 'fit' and pair['trained'] and pair['error'] is None
    assert pair['protocol_sha256'] == sha(PACK / 'protocol.json')
    assert protocol['rows'] == manifest['train'] + manifest['development']
    assert pair['initial_conditions_and_predictions_exact'] and pair['real_history_projection_gradient_exists']
    assert all(sha(BASE / n) == h for n, h in manifest['source_sha256'].items())
    ntrain, ndev = len(manifest['train']), len(manifest['development'])
    assert (ntrain, ndev) == (128, 40)
    ident = lambda rows: {(r['shard'], r['record_index']) for r in rows}
    assert len(ident(manifest['train'])) == 64 and len(ident(manifest['development'])) == 10
    assert not ident(manifest['train']) & ident(manifest['development'])
    expected_files = {'report.json'}
    for arm in ['repeat_current', 'real_previous']:
        expected_files.update(arm + '/' + n for n in ['report.json', 'plan.json', 'runner_snapshot.py',
            'common_updates.jsonl', 'reduced_updates.jsonl', 'common_u0000_train.npz', 'reduced_u8192_train.npz', 'reduced_u8192_development.npz'])
    assert {p.relative_to(DATA).as_posix() for p in DATA.rglob('*') if p.is_file()} == expected_files
    hashes = {name: sha(DATA / name) for name in sorted(expected_files)}
    arrays, reports, results, log_rows, table_windows = {}, {}, {}, {}, []
    deltas = dict.fromkeys(FIELDS, 0.)
    def compare(actual, expected):
        for k in FIELDS:
            d = abs(actual[k] - expected[k]); deltas[k] = max(deltas[k], d)
            assert d < (.02 if 'rotation' in k else 1e-4), (k, actual[k], expected[k])
    input_audit = read(PACK / 'history_inputs.json')
    for arm in ['repeat_current', 'real_previous']:
        directory = DATA / arm; r = read(directory / 'report.json'); reports[arm] = r
        assert r['history_arm'] == arm and r['state_arm'] == 'real' and r['stage'] == 'fit'
        assert sha(directory / 'runner_snapshot.py') == sha(PACK / 'fit_history_arm.py')
        plan = read(directory / 'plan.json'); assert plan['manifest'] == manifest
        assert r['manifest_sha256'] == sha(BASE / 'manifest.json') and r['source_sha256'] == manifest['source_sha256']
        assert r['initialization_seed'] == 43 and r['branch_update'] == 4096 and r['final_update'] == 8192
        assert r['sampling_steps'] == 100 and r['sampling_seeds'] == [1101, 1102, 1103]
        for k in ['execution_checks_passed', 'input_contract_passed', 'clip_unchanged', 'other_frozen_unchanged']: assert r[k]
        for k in ['gripper_trained', 'simulation_executed', 'downloads', 'reserved_test_targets_read']: assert not r[k]
        state = r['state_checks']; assert state['raw_current_pose_exact'] and state['train_only_statistics']
        assert not state['target_or_future_state_used_as_context']
        history = r['history_input_checks']; assert history['passed'] and not history['future_input_used'] and not history['cross_episode_input_used']
        assert len(history['lineage']) == 168 and (history['real_history_train'], history['real_history_development']) == (64, 30)
        for j, (line, audited) in enumerate(zip(history['lineage'], input_audit['rows'])):
            assert line['window'] == j and line['current_index'] == audited['start_index'] and line['previous_index'] == max(0, audited['start_index'] - 1)
            assert line['has_real_history'] == (line['current_index'] > 0)
            assert line['encoded_sha256'] == audited['previous_encoded_sha256'] and line['rgb_sha256'] == audited['previous_rgb_sha256']
        assert all(x['raw_identity_exact'] and all(v['max_abs'] <= v['atol'] for v in x['differences'].values()) for x in r['input_numeric_comparisons'])
        assert len(r['data_checks']) == 168 and r['data_checks'] == read(OLD / 'report.json')['data_checks']
        assert r['state_protocol_sha256'] == sha(ROOT / 'cluster_packages/bridge_state_conditioning_v1/protocol.json')
        assert len(r['runs']) == 1; run = r['runs'][0]
        assert run['arm'] == 'reduced' and run['learning_rate'] == 1e-5 and run['actual_draw_slots_per_window'] == [256] * 128
        assert run['history'][-1]['train'] == run['final_train']
        logs = []
        for segment, lo, hi, lr in [('common', 1, 4096, 1e-4), ('reduced', 4097, 8192, 1e-5)]:
            records = [json.loads(s) for s in (directory / (segment + '_updates.jsonl')).read_text().splitlines()]
            assert [x['update'] for x in records] == list(range(lo, hi + 1))
            assert all(x['learning_rate'] == lr and np.isfinite(x['loss']) and np.isfinite(x['global_unclipped_gradient_norm']) for x in records)
            for x in records:
                assert set(x['unclipped_module_gradient_norms']) == {'adapter.', 'diffusion_decoder.', 'history_projection.', 'state_projection.'}
                assert all(np.isfinite(v) and v >= 0 for v in x['unclipped_module_gradient_norms'].values())
            assert records[0]['loss'] == (r['warmup_first_loss'] if segment == 'common' else run['loss_first'])
            if segment == 'reduced':
                assert records[-1]['loss'] == run['loss_last']
                assert float(np.mean([x['loss'] for x in records])) == run['mean_loss']
            logs.extend(records)
        log_rows[arm] = logs
        assert max(x['unclipped_module_gradient_norms']['history_projection.'] for x in logs) > 0
        # No checkpoint download: roundtrip/frozen protections come from the
        #execution report and preserved source, rather than a local weight audit.
        assert len(r['checkpoints']) == 1 and r['checkpoints'][0]['roundtrip_exact'] and not r['checkpoints'][0]['includes_optimizer']
        for segment, update, split, recorded in [('common', 0, 'train', r['common_history'][0]['train']),
                ('reduced', 8192, 'train', run['final_train']), ('reduced', 8192, 'development', run['final_development'])]:
            with np.load(directory / f'{segment}_u{update:04d}_{split}.npz', allow_pickle=False) as z:
                a = {n: z[n].copy() for n in z.files}
            assert set(a) == {'predictions', 'targets', 'contexts', 'row_indices'}
            rows = manifest[split]; n = len(rows)
            assert a['predictions'].shape == (3, n, 16, 8) and a['targets'].shape == (n, 16, 8)
            assert a['contexts'].shape == (n, 1024) and all(np.isfinite(v).all() for v in a.values())
            assert a['row_indices'].tolist() == list(range(n)) if split == 'train' else a['row_indices'].tolist() == list(range(128, 168))
            with np.load(OLD / f'reduced_u8192_{split}.npz') as old:
                assert np.array_equal(a['targets'], old['targets'])
            m, good, gate, pos, rot = summarize(a['predictions'], a['targets']); compare(m, recorded)
            static = np.zeros_like(a['targets']); static[..., 6] = 1
            sm, _, _, _, _ = summarize(static[None], a['targets']); compare(sm, {k: recorded['static_' + k] for k in FIELDS})
            if split == 'train':
                assert gate == recorded['fit_gate']['pose_fit_passed'] and int(good.sum()) == recorded['fit_gate']['windows_all_three_seeds_pass']
            if update == 8192:
                key = arm + '/' + split; arrays[key] = a
                results[key] = dict(metrics=m, static=sm, pose_fit_passed=gate, passing_windows=int(good.sum()), windows=n)
                by_episode = defaultdict(list)
                for j, row in enumerate(rows):
                    by_episode[row['shard'] + '::' + str(row['record_index'])].append(j)
                    table_windows.append(dict(arm=arm, split=split, window=j, task=row['task'], episode=row['shard'] + '::' + str(row['record_index']),
                        real_history_available=row['start_index'] > 0, path_position_cm=float(pos[:, j].mean()), path_rotation_deg=float(rot[:, j].mean())))
                recorded_episodes = {e['episode']: e for e in recorded['per_episode']}
                assert set(by_episode) == set(recorded_episodes)
                for episode, j in by_episode.items():
                    em = summarize(a['predictions'][:, j], a['targets'][j])[0]; compare(em, recorded_episodes[episode])
            else: arrays[arm + '/initial'] = a
        compare(results[arm + '/development']['metrics'], pair['all40_final_control' if arm == 'repeat_current' else 'all40_final_history'])
    a, b = reports['repeat_current'], reports['real_previous']
    assert a['initial_shared_sha256'] == b['initial_shared_sha256'] and a['augmented_initial_sha256'] == b['augmented_initial_sha256']
    assert a['history_input_checks'] == b['history_input_checks'] and a['data_checks'] == b['data_checks']
    assert read(DATA / 'repeat_current/plan.json') == read(DATA / 'real_previous/plan.json')
    assert all(np.array_equal(arrays['repeat_current/initial'][k], arrays['real_previous/initial'][k]) for k in arrays['repeat_current/initial'])
    for split in ['train', 'development']:
        assert np.array_equal(arrays['repeat_current/' + split]['targets'], arrays['real_previous/' + split]['targets'])
    original_replay = {}
    for split in ['initial', 'train', 'development']:
        filename = 'common_u0000_train.npz' if split == 'initial' else 'reduced_u8192_' + split + '.npz'
        with np.load(OLD / filename) as old:
            differences = {k: float(np.max(np.abs(arrays['repeat_current/' + split][k] - old[k]))) for k in old.files}
        assert all(v == 0 for v in differences.values()); original_replay[split] = differences
    original_log = [json.loads(s) for n in ['common_updates.jsonl', 'reduced_updates.jsonl'] for s in (OLD / n).read_text().splitlines()]
    control_loss_difference = max(abs(x['loss'] - y['loss']) for x, y in zip(original_log, log_rows['repeat_current']))
    assert control_loss_difference == pair['first16_control_loss_max_difference'] == 0 and len(original_log) == 8192
    subset_table, episode_table, task_table = [], [], []
    dev_rows = manifest['development']
    groups = [('all40', list(range(40))), ('real_history30', [i for i, r in enumerate(dev_rows) if r['start_index'] > 0]),
              ('boundary10', [i for i, r in enumerate(dev_rows) if r['start_index'] == 0])]
    for label, indices in groups:
        item = dict(group=label, windows=len(indices))
        for arm in ['repeat_current', 'real_previous']:
            data = arrays[arm + '/development']; item[arm] = summarize(data['predictions'][:, indices], data['targets'][indices])[0]
        item['history_minus_control'] = {k: item['real_previous'][k] - item['repeat_current'][k] for k in FIELDS}
        subset_table.append(item)
    for kind, output in [('episode', episode_table), ('task', task_table)]:
        grouped = defaultdict(list)
        for i, row in enumerate(dev_rows):
            grouped[row['shard'] + '::' + str(row['record_index']) if kind == 'episode' else row['task']].append(i)
        for label, indices in grouped.items():
            item = {kind: label, 'task': dev_rows[indices[0]]['task'], 'windows': len(indices)}
            for arm in ['repeat_current', 'real_previous']:
                data = arrays[arm + '/development']; item[arm] = summarize(data['predictions'][:, indices], data['targets'][indices])[0]
            item['history_minus_control'] = {k: item['real_previous'][k] - item['repeat_current'][k] for k in FIELDS}
            output.append(item)
    position_improvements = sum(x['history_minus_control']['path_position_cm'] < 0 for x in episode_table)
    rotation_improvements = sum(x['history_minus_control']['path_rotation_deg'] < 0 for x in episode_table)
    control, history = results['repeat_current/development']['metrics'], results['real_previous/development']['metrics']
    reduction = (control['path_position_cm'] - history['path_position_cm']) / control['path_position_cm']
    conditions = dict(position_reduction_at_least_10_percent=reduction >= .10,
        rotation_not_worse=history['path_rotation_deg'] <= control['path_rotation_deg'],
        at_least_6_episode_position_improvements=position_improvements >= 6,
        both_training_gates_passed=all(results[arm + '/train']['pose_fit_passed'] for arm in ['repeat_current', 'real_previous']),
        control_replay_passed=True)
    candidate = all(conditions.values())
    static = results['real_previous/development']['static']
    development_pass = history['path_position_cm'] < static['path_position_cm'] and history['path_rotation_deg'] < static['path_rotation_deg']
    assert pair['both_training_gates_passed'] == conditions['both_training_gates_passed']
    result = dict(verified=True, artifacts=len(expected_files), artifact_bytes=sum((DATA / f).stat().st_size for f in expected_files),
        artifact_sha256=hashes, protocol_sha256=sha(PACK / 'protocol.json'), results=results,
        maximum_metric_recomputation_differences=deltas, original_control_replay=original_replay,
        all8192_control_losses_max_difference=control_loss_difference,
        position_reduction_fraction=reduction, position_worsening_percent=-100 * reduction,
        history_minus_control={k: history[k] - control[k] for k in FIELDS}, candidate_conditions=conditions,
        candidate_passed=candidate, development_beats_static=bool(development_pass),
        position_improved_episodes=position_improvements, rotation_improved_episodes=rotation_improvements,
        per_episode=episode_table, per_task=task_table, subgroups=subset_table,
        checkpoint_loaded=False, training_executed=False, new_model_inference=False,
        verdict='Close the current history configuration; no follow-up seed or parameter sweep' if not candidate else 'Candidate pending the one allowed paired initialization replication',
        limitations='One training initialization, ten reused development episodes with overlapping windows; frozen-weight and source protections partly rely on GPU report. Does not prove history generally useless or uniquely identify a faulty module.')
    (OUT / 'analysis.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    write_csv('per_window.csv', table_windows)
    flat = lambda rows: [{**{k:v for k,v in row.items() if not isinstance(v,dict)},
        **{group+'_'+k:v for group in ['repeat_current','real_previous','history_minus_control'] for k,v in row[group].items()}} for row in rows]
    write_csv('per_episode.csv', flat(episode_table)); write_csv('per_task.csv', flat(task_table)); write_csv('subgroups.csv', flat(subset_table))
    print(json.dumps({k:v for k,v in result.items() if k not in ['artifact_sha256','original_control_replay','maximum_metric_recomputation_differences','per_episode','per_task']}, indent=2))
    print('EPISODES', [(x['episode'], round(x['history_minus_control']['path_position_cm'],3)) for x in episode_table])


if __name__ == '__main__': main()

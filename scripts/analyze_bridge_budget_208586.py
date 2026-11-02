"""CPU-only audit of the finite 806-pool optimizer-budget experiment."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

from analyze_bridge_demo806_208207 import load, static
from analyze_bridge_heldout_208056 import errors, summarize

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'reports/bridge-budget-fit-208586'
BASELINE = ROOT/'reports/bridge-demo806-fit-208207'
PACK = ROOT/'cluster_packages/bridge_demo806_budget_v1'
INPUT = ROOT/'cluster_packages/bridge_demonstration_expansion_v1'
OUT = ROOT/'reports/bridge_budget_analysis_208586'
METRICS = ('path_position_cm', 'path_rotation_deg', 'endpoint_position_cm', 'endpoint_rotation_deg')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    r, p, previous = read(SOURCE/'report.json'), read(PACK/'protocol.json'), read(BASELINE/'report.json')
    baseline_analysis = read(ROOT/'reports/bridge_demo806_analysis_208207/analysis.json')
    assert baseline_analysis['verified']
    assert sha(BASELINE/'report.json') == p['baseline_report_sha256'] == r['budget']['baseline_report_sha256']
    assert sha(PACK/'protocol.json') == r['budget']['protocol_sha256']
    for name in ('run_budget.py', 'budget_hooks.py', 'protocol.json'):
        assert sha(PACK/name) == sha(SOURCE/('budget_'+name)) == r['budget_source_sha256'][name], name
    assert sha(SOURCE/'runner_snapshot.py') == previous['entry_sha256']
    assert sha(SOURCE/'budget_effective_runner_snapshot.py') == p['budget_effective_runner_sha256'] == r['budget_effective_runner_sha256']
    spec = importlib.util.spec_from_file_location('audit_budget_source_factory', PACK/'run_budget.py')
    factory = importlib.util.module_from_spec(spec); spec.loader.exec_module(factory)
    effective = factory.adapted_source((BASELINE/'effective_runner_snapshot.py').read_text())
    assert hashlib.sha256(effective.encode()).hexdigest() == p['budget_effective_runner_sha256']
    assert r['stage'] == 'fit' and not r['local_bounded']
    for key in ('execution_checks_passed', 'full_input_contract_passed', 'retained_numeric_contract_passed',
                'old_1876_input_contract_exact', 'original_1876_token_cache_reused', 'retained_visual_tokens_exact',
                'retained_language_tokens_exact', 'full_token_cache_reused', 'clip_unchanged', 'other_frozen_unchanged', 'development_scored'):
        assert r[key], key
    for key in ('downloads', 'simulation_executed', 'gripper_trained', 'reserved_test_targets_read',
                'old_control_retrained', 'state_statistics_changed'):
        assert r[key] is False, key
    assert r['optimization_operations'] == r['final_update'] == 16170 and r['branch_update'] == 4096
    for key in ('input_row_checks', 'episode_payload_checks', 'input_identity_sha256', 'token_cache_key_sha256',
                'initial_shared_sha256', 'augmented_initial_sha256', 'torch_version', 'initialization_seed',
                'sampling_steps', 'sampling_seeds', 'trainable_parameters', 'verified_row_indices'):
        assert r[key] == previous[key], key
    assert r['active_token_rows'] == 3664 and r['old_token_rows_reencoded'] == r['new_token_rows_encoded'] == 0
    assert sha(PACK/'schedule.npz') == p['schedule_sha256'] == r['budget']['schedule_sha256']
    with np.load(PACK/'schedule.npz') as z, np.load(INPUT/'schedule.npz') as old:
        batches = z['batches']
        assert batches.shape == (16170, 8) and np.array_equal(batches[:8192], old['batches'])
        assert hashlib.sha256(batches.tobytes()).hexdigest() == p['schedule_tensor_sha256']
        assert all(len(set(batch)) == 8 for batch in batches)
    # Original logs were not separately downloaded, but their frozen runtime digests
    # match the full downloaded reconstructed logs byte for byte, not just means.
    log_proof = {}
    all_logs = []
    for name, first, last in [('common_updates.jsonl', 1, 4096),
                              ('reduced_updates.jsonl', 4097, 8192),
                              ('extension_updates.jsonl', 8193, 16170)]:
        entries = [json.loads(line) for line in (SOURCE/name).read_text().splitlines()]
        assert [x['update'] for x in entries] == list(range(first, last+1))
        for entry in entries:
            assert entry['learning_rate'] == (1e-4 if entry['update'] <= 4096 else 1e-5)
            assert np.isfinite(entry['loss']) and np.isfinite(entry['global_unclipped_gradient_norm'])
            assert all(np.isfinite(v) for v in entry['unclipped_module_gradient_norms'].values())
        if first <= 8192:
            exact = sha(SOURCE/name) == r['budget']['baseline_log_sha256'][name]
            assert exact
            log_proof[name] = dict(original_digest_match=True, sha256=sha(SOURCE/name), rows=len(entries))
        all_logs.extend(entries)
    assert len(all_logs) == 16170
    assert all_logs[0]['loss'] == r['common_first_loss'] and all_logs[4095]['loss'] == r['common_last_loss']
    assert all_logs[-1]['loss'] == r['runs'][0]['loss_last']
    assert sha(BASELINE/'reduced_u8192_development.npz') == p['baseline_development_npz_sha256']
    with np.load(SOURCE/'prefix8192_replay_check.npz') as replay, np.load(BASELINE/'reduced_u8192_development.npz') as old:
        replay_differences = {
            'contexts': float(np.max(np.abs(replay['contexts'].astype(float)-old['contexts'][:4]))),
            'predictions': float(np.max(np.abs(replay['predictions'].astype(float)-old['predictions'][:, :4]))),
            'targets': float(np.max(np.abs(replay['targets'].astype(float)-old['targets'][:4])))}
        assert all(v == 0 for v in replay_differences.values())
    for key in ('additional_data', 'parameters_reinitialized_before_extension', 'optimizer_reset_before_extension', 'reserved25_scored'):
        assert r['budget'][key] is False
    assert r['budget']['prefix8192_replay']['passed']
    assert [x['update'] for x in r['optimizer_checkpoints']] == [8192, 16170]
    assert all(x['roundtrip_exact'] and x['bytes'] < 2*1024**3 for x in r['optimizer_checkpoints'])
    rows = read(INPUT/'rows.json')
    summaries, differences, arrays = {}, [], {}
    for split, first in [('train', 0), ('development', 3624)]:
        a = load(SOURCE/f'reduced_u16170_{split}.npz', len(rows[split]), first)
        arrays[split] = a
        with np.load(BASELINE/f'reduced_u8192_{split}.npz') as old:
            assert np.array_equal(a['targets'], old['targets'])
        for index, target in enumerate(a['targets']):
            assert hashlib.sha256(target.tobytes()).hexdigest() == r['input_row_checks'][first+index]['target_sha256']
        computed = summarize(a['predictions'], a['targets'], rows[split])
        hold = summarize(static(a['targets']), a['targets'], rows[split])
        summaries[split] = computed
        saved = r['runs'][0]['final_'+split]
        for metric in METRICS:
            differences.append((metric, abs(computed['window_weighted'][metric]-saved[metric])))
            differences.append((metric, abs(hold['window_weighted'][metric]-saved['static_'+metric])))
        assert len(computed['per_episode']) == len(saved['per_episode'])
        for actual, recorded in zip(computed['per_episode'], saved['per_episode']):
            assert all(actual[k] == recorded[k] for k in ('episode', 'task', 'windows'))
            differences.extend((key, abs(actual[key]-recorded[key])) for key in METRICS)
        for key, recorded_key in [('path_position_cm', 'episode_mean_path_position_cm'), ('path_rotation_deg', 'episode_mean_path_rotation_deg')]:
            differences.append((key, abs(computed['episode_equal'][key]-saved[recorded_key])))
        position, rotation = errors(a['predictions'], a['targets'])
        for values, key, recorded_key in [(position, 'path_position_cm', 'seed_path_position_cm'), (rotation, 'path_rotation_deg', 'seed_path_rotation_deg')]:
            differences.extend((key, abs(float(v)-s)) for v, s in zip(values.mean(axis=(1, 2)), saved[recorded_key]))
    max_delta = {kind: max(v for k, v in differences if kind in k) for kind in ('position', 'rotation')}
    assert max_delta['position'] < 1e-4 and max_delta['rotation'] < 1e-3
    summaries['train_original405'] = summarize(arrays['train']['predictions'][:, :1836], arrays['train']['targets'][:1836], rows['train'][:1836])
    summaries['train_added401'] = summarize(arrays['train']['predictions'][:, 1836:], arrays['train']['targets'][1836:], rows['train'][1836:])
    paired, counts = {}, {}
    for group in ('per_episode', 'per_task'):
        paired[group] = []
        for new, old, hold in zip(summaries['development'][group], baseline_analysis['summaries']['806_dev'][group], baseline_analysis['summaries']['static_dev'][group]):
            key = 'episode' if group == 'per_episode' else 'task'
            assert new[key] == old[key] == hold[key]
            row = {key: new[key], 'task': new['task'], 'windows': new['windows']}
            for metric in METRICS:
                row.update({f'u8192_{metric}': old[metric], f'u16170_{metric}': new[metric],
                            f'static_{metric}': hold[metric], f'delta_{metric}': new[metric]-old[metric]})
            paired[group].append(row)
        counts[group] = dict(total=len(paired[group]), **{f'{metric}_better_u8192': sum(x[f'delta_{metric}'] < 0 for x in paired[group]) for metric in METRICS})
    gates = {f'{group}_{metric}': summaries['development'][group][metric] <= baseline_analysis['summaries']['806_dev'][group][metric]
             for group in ('window_weighted', 'episode_equal') for metric in METRICS[:2]}
    strict = any(summaries['development'][g][m] < baseline_analysis['summaries']['806_dev'][g][m]
                 for g in ('window_weighted', 'episode_equal') for m in METRICS[:2])
    screen_passed = all(gates.values()) and strict
    log_means = [dict(first=first, last=last, mean_loss=float(np.mean([x['loss'] for x in all_logs[first-1:last]])))
                 for first, last in [(1, 4096), (4097, 8192), (8193, 12288), (12289, 16170)]]
    result = dict(verified=True, job_id='208586', new_inference=False, training=False, gpu_used=False,
                  downloaded_files={f.name: dict(bytes=f.stat().st_size, sha256=sha(f)) for f in sorted(SOURCE.iterdir()) if f.is_file()},
                  snapshots_and_input_contracts_verified=True, baseline_log_proof=log_proof,
                  prefix4_replay_differences=replay_differences, all_targets_match8192_exact=True,
                  all_saved_pose_metrics_recomputed=True, max_float64_vs_report_difference=max_delta,
                  budget_report_recording_issue='reference_replay overwrote budget.prefix8192_replay with its context/prediction subrecord. Loss/parameter max fields are not retained in JSON. Loss exactness is independently recovered from byte-identical logs; parameter tolerance was enforced by the frozen source before extension, but its numeric max is not locally recomputed without the remote checkpoints. Do not claim a local parameter-equality audit.',
                  parameter_check_evidence='Frozen source enforces <=1e-5 before extension; optimizer-checkpoint metadata is runtime evidence only, PT files not downloaded.',
                  optimizer_checkpoints=r['optimizer_checkpoints'], final_model_checkpoint=r['checkpoints'], checkpoint_downloaded=False,
                  summaries=summaries, baseline_summaries={k: baseline_analysis['summaries'][k] for k in ('806_dev', 'static_dev')},
                  paired_counts=counts, log_block_means=log_means, gates=gates, screen_passed=screen_passed,
                  peak_reserved_gib=r['peak_reserved_gib'], elapsed_seconds=r['elapsed_seconds'],
                  next_action='Close this finite low-LR budget branch; preserve405 reference; no automatic further optimization or reserved25 scoring.',
                  limitations=['One training initialization; three diffusion seeds are not training replications.',
                               'Repeated10-episode development set, no independent test claim for806 model.',
                               'Rules out improvement from this continuation, not all training budgets or learning-rate schedules.',
                               'Frozen-parameter/cache/checkpoint guarantees use runtime evidence; no local model re-execution.',
                               'Offline measured pose with gripper frozen; no physical grasp claim.'])
    OUT.mkdir(exist_ok=True)
    for group, filename in [('per_episode', 'episodes.csv'), ('per_task', 'tasks.csv')]:
        with (OUT/filename).open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=list(paired[group][0])); writer.writeheader(); writer.writerows(paired[group])
    (OUT/'analysis.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(verified=True, screen_passed=screen_passed, gates=gates, replay_differences=replay_differences,
                         original_logs_byte_exact=True, metric_difference=max_delta,
                         means={k: {g: v[g] for g in ('window_weighted', 'episode_equal')} for k, v in summaries.items()},
                         counts=counts, tasks=paired['per_task'], log_means=log_means), indent=2))


if __name__ == '__main__':
    main()

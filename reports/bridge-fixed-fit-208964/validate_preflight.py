"""Read-only admission gate; operational success is not a generalization result."""
import argparse
from pathlib import Path
from fixed_hooks import read, sha

HERE = Path(__file__).resolve().parent


def validate(directory):
    r, p = read(directory/'report.json'), read(HERE/'protocol.json')
    assert r['stage'] == 'preflight' and not r['local_bounded'] and r['execution_checks_passed']
    assert r['optimization_operations'] == 12 and not r['development_scored'] and not r['runs'] and not r['checkpoints']
    f = r['fixed_context']
    assert f['protocol_sha256'] == sha(HERE/'protocol.json') == sha(directory/'protocol.json')
    assert sha(directory/'effective_runner_snapshot.py') == p['effective_runner_sha256'] == f['effective_runner_sha256']
    for name, expected in p['local_package_sha256'].items():
        assert sha(HERE/name) == sha(directory/name) == f['source_sha256'][name] == expected
    for key in ('full_input_contract_passed', 'retained_numeric_contract_passed', 'full_token_cache_reused',
                'clip_unchanged', 'other_frozen_unchanged'):
        assert r[key], key
    for key in ('first_masked_decoder_update_matches_joint', 'full_restored_masked_continuation_exact',
                'context_parameters_and_adam_unchanged', 'training_context_dropout_preserved',
                'global_clipping_before_update_mask', 'shadow_context_gradients_computed', 'context_update_mask_only'):
        assert f[key], key
    assert f['prefix_replay']['passed']
    assert not f['prefix_retrained'] and not f['new_candidate_development_scored']
    assert all(v <= p['tolerances']['loss'] for v in f['joint_first4_replay_max_difference'].values())
    for key in ('downloads', 'gripper_trained', 'simulation_executed', 'reserved_test_targets_read', 'state_statistics_changed'):
        assert not r[key], key
    assert len(r['input_row_checks']) == r['active_token_rows'] == 3664
    assert len(r['episode_payload_checks']) == 816 and r['verified_row_indices'] == list(range(3664))
    assert r['new_token_rows_encoded'] == r['old_token_rows_reencoded'] == 0
    assert r['torch_version'] == '2.5.1+cu118' and r['initialization_seed'] == 42
    assert r['peak_reserved_gib'] < 12 and r['estimated_formal_seconds'] < 14400
    return r


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('directory', type=Path); args = ap.parse_args()
    r = validate(args.directory)
    print('COMPLETE FIXED-CONTEXT PREFLIGHT VERIFIED; no generalization verdict')
    print({key: r[key] for key in ('peak_reserved_gib', 'warm_update_seconds', 'estimated_formal_seconds')})

"""Accept only a complete cluster optimizer-continuation preflight."""
import argparse
import json
from pathlib import Path

from budget_hooks import read, sha

HERE = Path(__file__).resolve().parent


def validate(directory):
    r, p = read(directory/'report.json'), read(HERE/'protocol.json')
    assert r['stage'] == 'preflight' and not r['local_bounded'] and r['execution_checks_passed']
    assert r['optimization_operations'] == 12 and not r['development_scored']
    assert not r['runs'] and not r['checkpoints']
    for name in ('run_budget.py', 'budget_hooks.py', 'protocol.json'):
        assert sha(HERE/name) == sha(directory/('budget_'+name)) == r['budget_source_sha256'][name]
    assert sha(directory/'budget_effective_runner_snapshot.py') == p['budget_effective_runner_sha256'] == r['budget_effective_runner_sha256']
    assert r['protocol_sha256'] == sha(HERE.parent/'bridge_demonstration_expansion_v1/protocol.json')
    assert len(r['input_row_checks']) == r['active_token_rows'] == 3664
    assert len(r['episode_payload_checks']) == 816 and r['verified_row_indices'] == list(range(3664))
    for key in ('full_input_contract_passed', 'retained_numeric_contract_passed', 'full_token_cache_reused',
                'original_1876_token_cache_reused', 'retained_visual_tokens_exact', 'retained_language_tokens_exact',
                'clip_unchanged', 'other_frozen_unchanged'):
        assert r[key], key
    assert r['new_token_rows_encoded'] == r['old_token_rows_reencoded'] == 0
    for key in ('downloads', 'simulation_executed', 'gripper_trained', 'reserved_test_targets_read',
                'old_control_retrained', 'state_statistics_changed'):
        assert r[key] is False, key
    assert r['torch_version'] == '2.5.1+cu118' and r['initialization_seed'] == 42
    b = r['budget']
    assert b['protocol_sha256'] == sha(HERE/'protocol.json') and b['prefix8192_schedule_exact']
    assert b['frozen208207_replay']['passed'] and b['optimizer_reload_continuation_exact']
    assert b['baseline_report_sha256'] == p['baseline_report_sha256']
    assert not b['actual_prefix8192_replay'] and not b['new_candidate_development_scored']
    assert len(r['optimizer_checkpoints']) == 1 and r['optimizer_checkpoints'][0]['roundtrip_exact']
    assert r['optimizer_checkpoints'][0]['probe_deleted_after_verified_reload']
    assert r['peak_reserved_gib'] < 12 and r['estimated_formal_seconds'] < 12600
    return r


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('directory', type=Path); args = ap.parse_args()
    r = validate(args.directory)
    print('COMPLETE BUDGET PREFLIGHT VERIFIED; prefix8192 replay still required in formal run')
    print(json.dumps(dict(peak_reserved_gib=r['peak_reserved_gib'], warm_update_seconds=r['warm_update_seconds'],
                          estimated_formal_hours=r['estimated_formal_seconds']/3600), indent=2))


if __name__ == '__main__':
    main()

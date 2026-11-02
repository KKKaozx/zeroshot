"""Validate complete806-pool preflight before its single fixed-budget formal run."""
import argparse
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('directory',type=Path);args=ap.parse_args()
    r=json.loads((args.directory/'report.json').read_text());p=json.loads((HERE/'protocol.json').read_text())
    assert r['stage']=='preflight' and not r['local_bounded'] and r['execution_checks_passed']
    assert r['protocol_sha256']==sha(HERE/'protocol.json')
    assert r['entry_sha256']==sha(HERE/'run_training_coverage.py')
    assert r['input_helper_sha256']==sha(HERE/'expanded_inputs.py')
    assert r['adaptation']['wrapper_sha256']==sha(HERE/'run_expansion.py')
    assert r['adaptation']['contract_sha256']==sha(HERE/'adaptation_contract.json')
    assert r['adaptation']['effective_runner_sha256']==sha(args.directory/'effective_runner_snapshot.py')
    assert r['branch_restore_passed'] and r['optimization_operations']==16
    assert r['old4_frozen_replay']['passed']
    assert r['old4_frozen_replay']['context_max_abs']<=1e-4 and r['old4_frozen_replay']['prediction_max_abs']<=1e-4
    assert len(r['input_row_checks'])==3664 and len(r['episode_payload_checks'])==816
    assert r['verified_row_indices']==list(range(3664)) and r['active_token_rows']==3664
    for key in ('full_input_contract_passed','retained_numeric_contract_passed','old_1876_input_contract_exact',
                'original_1876_token_cache_reused','retained_visual_tokens_exact','retained_language_tokens_exact',
                'clip_unchanged','other_frozen_unchanged'):
        assert r[key],key
    assert r['old_token_rows_reencoded']==0 and r['new_token_rows_encoded'] in (0,1788)
    for key in ('gripper_trained','downloads','simulation_executed','reserved_test_targets_read',
                'old_control_retrained','development_scored','state_statistics_changed'):
        assert not r[key],key
    assert r['initialization_seed']==42 and r['torch_version']=='2.5.1+cu118'
    assert r['peak_reserved_gib']<12 and r['estimated_formal_seconds']<12600
    assert (HERE/'full_tokens.pt').is_file() and (HERE/'full_tokens.pt').stat().st_size<5*1024**3
    assert not r['checkpoints'] and not any('final_development' in a for a in r['runs'])
    assert r['initial_shared_sha256']==p['expected_initial_shared_sha256']
    assert r['augmented_initial_sha256']==p['expected_augmented_initial_sha256']
    print('COMPLETE806 PREFLIGHT VERIFIED; no generalization verdict')
    print(json.dumps(dict(peak_reserved_gib=r['peak_reserved_gib'],warm_update_seconds=r['warm_update_seconds'],
        estimated_formal_hours=r['estimated_formal_seconds']/3600,cache_gib=(HERE/'full_tokens.pt').stat().st_size/2**30),indent=2))


if __name__=='__main__':main()

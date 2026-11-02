"""Only a complete cluster preflight may authorize this fixed formal run."""
import argparse
import hashlib
import json
from pathlib import Path
CODE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('directory',type=Path)
    ap.add_argument('--control',type=Path,default=Path('/projects/Zeroshot/runs/bridge-state-fit-202278/real/report.json'))
    ap.add_argument('--cache',type=Path,default=CODE/'full_tokens.pt')
    args=ap.parse_args();r=read(args.directory/'report.json');p=read(CODE/'protocol.json');old=read(args.control)
    assert sha(args.control)==p['control_report_sha256']
    assert r['stage']=='preflight' and not r['local_bounded'] and r['execution_checks_passed']
    assert r['branch_restore_passed'] and r['optimization_operations']==16
    assert r['full_input_contract_passed'] and len(r['input_row_checks'])==1876 and len(r['episode_payload_checks'])==415
    assert r['verified_row_indices']==list(range(1876)) and r['active_token_rows']==1876
    assert r['retained_data_checks']==old['data_checks'] and r['retained_state_checks']==old['state_checks']
    assert r['retained_state_features_sha256']==old['state_features_sha256']
    assert r['initial_shared_sha256']==old['initial_shared_sha256']
    assert r['augmented_initial_sha256']==old['augmented_initial_sha256'] and r['torch_version']==old['torch_version']
    assert r['protocol_sha256']==sha(CODE/'protocol.json')
    assert r['entry_sha256']==sha(CODE/'run_training_coverage.py')
    assert r['input_helper_sha256']==sha(CODE/'coverage_inputs.py')
    for path,h in p['inherited_sha256'].items():assert sha(CODE.parent/path)==h
    for k in ['clip_unchanged','other_frozen_unchanged','original_token_cache_reused','retained_visual_tokens_exact','retained_language_tokens_exact']:
        assert r[k],k
    assert not any(r[k] for k in ['gripper_trained','downloads','simulation_executed','reserved_test_targets_read',
                                  'old_control_retrained','development_scored','state_statistics_changed'])
    assert r['peak_reserved_gib']<7.5 and r['estimated_formal_seconds']<9000
    assert args.cache.is_file() and args.cache.stat().st_size<3*1024**3
    assert not r['checkpoints'] and not any('final_development' in v for v in r['runs'])
    print('COMPLETE TRAINING-COVERAGE PREFLIGHT VERIFIED; no generalization verdict')
    print(json.dumps(dict(peak_reserved_gib=r['peak_reserved_gib'],warm_update_seconds=r['warm_update_seconds'],
        estimated_formal_hours=r['estimated_formal_seconds']/3600,cache_gib=args.cache.stat().st_size/2**30),indent=2))


if __name__=='__main__':main()

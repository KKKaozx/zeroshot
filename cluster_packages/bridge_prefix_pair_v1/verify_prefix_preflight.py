"""Read and validate a COMPLETE cluster operational report; never submits jobs."""
import argparse
import hashlib
import json
from pathlib import Path
CODE=Path(__file__).resolve().parent


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('directory',type=Path);args=ap.parse_args()
    r=json.loads((args.directory/'report.json').read_text())
    p=json.loads((CODE/'protocol.json').read_text())
    assert r['stage']=='preflight' and r['execution_checks_passed']
    assert r['protocol_sha256']==sha(CODE/'protocol.json') and r['package_sums_sha256']==sha(CODE/'SHA256SUMS')
    for name in ('formal_training_executed','development_scored','final25_payloads_read','gripper_trained','downloads','final_weights_saved'):
        assert not r[name],name
    assert r['optimization_operations']==16 and r['clip_unchanged'] and r['gripper_unchanged']
    assert r['actual207558_checkpoint_sha256']==p['initialization']['sha256']
    assert r['input_checks']['counts']==dict(train_delta=1558,validation=50,train_episodes=405,validation_episodes=10)
    assert r['trained207558_replay']['passed']
    assert r['trained207558_replay']['context_max_abs']<=1e-4
    assert r['trained207558_replay']['prediction_max_abs']<=1e-4
    assert [v['arm'] for v in r['runs']]==['control','candidate']
    for arm in r['runs']:
        assert arm['logical_optimizer_updates']==len(arm['losses'])==8
        for flag in ('restore_weights_and_Adam_exact','restored_next_prediction_and_loss_exact',
            'request_projection_changed','base_trainables_changed','temporary_checkpoint_removed'):assert arm[flag],flag
    assert r['runs'][0]['losses'][0]==r['runs'][1]['losses'][0]
    assert r['peak_reserved_gib']<=7.5 and r['estimated_paired_formal_seconds']<=7200
    assert (CODE/'delta_visual_tokens.pt').is_file()
    print('PREFIX PREFLIGHT VERIFIED; no generalization verdict or automatic formal submission')
    print(json.dumps({k:r[k] for k in ('peak_reserved_gib','warm_update_seconds','estimated_paired_formal_seconds')},indent=2))


if __name__=='__main__':main()

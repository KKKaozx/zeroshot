"""Read an engineering receipt only; no automatic formal release/submission."""
import argparse
import json
from pathlib import Path
from pose_features import HERE,read,sha


def verify(path):
    r=read(path/'report.json')
    assert r['engineering_passed'] and r['protocol_sha256']==sha(HERE/'protocol.json')
    assert r['physical_optimizer_calls']==6 and r['logical_steps_per_group']==2
    assert r['paired_initial_loss_base_gradients_and_noise_exact'] and r['source_checkpoint_size_mtime_unchanged']
    assert r['formal_training_submitted'] is False and r['formal_code_released'] is False
    c,s=[r['groups'][role] for role in ['mask_control','real_pose']]
    assert c['zero_endpoint_replay']['passed'] and c['zero_endpoint_replay']['archived_x0_predictions_exact']
    for group in [c,s]:
        assert group['passed'] and group['restore_second_step_exact'] and group['restore_base_port_adam_plan_schedule_rng_exact']
        assert group['optimizer_calls']==3 and group['logical_new_updates']==2 and group['completed_update']==8194
        assert group['new_port_parameter_count']==18432 and group['disk_full_checkpoints_written']==0
        assert group['frozen_digest_unchanged'] and not group['trained_weights_retained_for_formal_use']
    return {'engineering_verified':True,'source_update':8192,'physical_disposable_optimizer_calls':6,
        'same_initial_core_loss_gradients_noise':True,'full_branch_and_core_adam_rng_restore_exact':True,
        'peak_reserved_gib':{role:g['peak_reserved_gib'] for role,g in r['groups'].items()},
        'estimated_pair_update_minutes_excluding_replay_eval_io':r['estimated_pair_update_seconds_excluding_replay_eval_io']/60,
        'formal_training_submitted':False,'generalization_verdict':None,'automatic_following_action':False}


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('output',type=Path)
    print('CURRENT POSE PREFLIGHT RECEIPT VERIFIED; engineering only')
    print(json.dumps(verify(ap.parse_args().output),indent=2))

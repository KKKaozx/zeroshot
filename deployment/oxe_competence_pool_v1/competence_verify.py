"""Verify complete saved endpoint and all paired primitive traces, without inference."""
import argparse
import json
from pathlib import Path
import numpy as np
import competence_bind as binding


def verify_outputs(root,output):
    pq=binding.parent(root);pq.attach(root)
    from sampling_verify import verify_training
    from pair_steps import check_paired_stochastic
    from competence_metrics import summarize
    from run_competence import make_contract
    p=binding.read(binding.HERE/'protocol.json');r=binding.read(output/'report.json');c=r['contract']
    assert r['execution_complete'] and c==make_contract(p,binding.read(binding.HERE/'source_control_binding.json'))
    assert r['source_replay_exact'] and r['all_actual_noise_t_and_rng_matched'] and r['actual_matched_steps']==p['updates_each']
    assert r['new_control_training_updates']==0 and r['control_checkpoint_unchanged']
    assert r['automatic_following_action'] is False and r['main_model_replaced'] is False and r['same_training_allocation'] is False
    guard=binding.read(output/'zero_update_guard.json')
    assert guard==r['zero_update_guard'] and guard['passed'] and guard['optimizer_updates']==0
    assert guard['old_control_loss_gradient_noise_rng_replay_exact'] and guard['new_pool_noise_t_rng_matched']
    saved,old=binding.control(root);plans=binding.load_plans()
    state=root/'runs/oxe-sampling-pair-233531/replacement/latest_state.pt'
    assert [state.stat().st_size,state.stat().st_mtime_ns]==c['source_binding']['control_checkpoint_size_mtime']
    for role,record in guard['first_batch'].items():
        assert role in ['replacement','moderate_pool'] and record['token_shape']==[8,77]
        check_paired_stochastic(record['measured'],record['noise'],saved['trace']['history'][0],saved['trace']['noise'][0])
        if role=='replacement':
            assert record['noise']==saved['trace']['noise'][0]
            assert all(record['measured'][k]==saved['trace']['history'][0][k] for k in ['loss','pose_loss','gripper_loss','gradient_norm'])
    assert set(guard['first_batch'])=={'replacement','moderate_pool'}
    candidate=verify_training(output/'moderate_pool','moderate_pool',c,plans)
    assert candidate['receipt']==r['training'] and r['control_training_receipt']==saved['receipt']
    assert guard['initial_identity']==candidate['receipt']['initial_identity']==saved['receipt']['initial_identity']
    assert r['training_job_ids']==candidate['receipt']['training_job_ids'] and r['old_control_training_job_ids']==old['training_job_ids']
    for row,event,oldrow,oldevent in zip(candidate['trace']['history'],candidate['trace']['noise'],saved['trace']['history'],saved['trace']['noise']):
        check_paired_stochastic(row,event,oldrow,oldevent)
    selection=binding.read(binding.HERE/'selection.json')
    with np.load(output/'predictions_and_targets.npz') as z:
        target=z['targets'];mask=z['masks'];outputs={k:z[k] for k in ['moderate_pool','replacement','source232587','static']}
        assert target.shape==mask.shape==(120,16,8) and all(v.shape==(3,120,16,8) and np.isfinite(v).all() for v in outputs.values())
        with np.load(root/'runs/oxe-sampling-pair-233531/predictions_and_targets.npz') as oldz:
            assert all(np.array_equal(z[k],oldz[k]) for k in ['targets','masks','replacement','source232587','static'])
        with np.load(output/'moderate_pool/predictions.npz') as arm:
            assert np.array_equal(arm['predictions'],outputs['moderate_pool']) and np.array_equal(arm['targets'],target) and np.array_equal(arm['masks'],mask)
        assert summarize(outputs,target,mask,selection)==r['metrics']
    return {'output_verified':True,'new_training_updates':p['updates_each'],'control_retrained':False,
        'source_replay_exact':True,'all_actual_primitive_noise_t_and_rng_matched':True,
        'train32_anchor_results':r['metrics']['by_source']['train'],
        'train32_anchor_motion':r['metrics']['motion']['train'],
        'all_source_training_anchor_basic_passed':r['metrics']['all_source_training_anchor_basic_passed'],
        'limited_training_anchor_support':r['metrics']['limited_training_anchor_support'],
        'scope':r['metrics']['scope'],'automatic_following_action':False}


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path('/projects/Zeroshot'));ap.add_argument('output',type=Path)
    a=ap.parse_args();result=verify_outputs(a.root.resolve(),a.output)
    print('MODERATE POOL OUTPUT VERIFIED; training anchors only')
    print(json.dumps(result,indent=2))

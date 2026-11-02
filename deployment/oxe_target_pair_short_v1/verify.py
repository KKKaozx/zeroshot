"""Check actual prefix outputs against the fixed schedule; no model construction."""
import argparse
import copy
import json
from pathlib import Path
import sys
import numpy as np
from protocols import HERE,read,sha


def arm(path,target,*,checkpoint=True):
    receipt=read(path/'arm_receipt.json');report=read(path/'step_report.json')
    pair=read(HERE/'pair_protocol.json');reference=read(HERE/'reference_contract.json')
    assert receipt['passed'] and receipt['target']==target and receipt['prefix_updates']==8192
    assert receipt['original_lr_horizon']==65536 and receipt['pair_protocol_sha256']==sha(HERE/'pair_protocol.json')
    assert receipt['gpu']=='NVIDIA A40' and receipt['full_optimizer_state_retained'] and receipt['not_full_horizon_training']
    assert report['completed_updates']==8192 and report['total_updates']==65536 and report['complete'] is False
    assert report['diagnostic_prefix_completed'] and report['frozen_state_unchanged'] and report['validation']==[]
    assert report['config']['model']['diffusion_prediction_type']==target
    state=receipt['hook_state']
    assert state==report['diagnostic_target_state'] and state['target']==target
    assert state['completed_updates']==state['captured_batches']==8192
    assert state['initial_learned_sha256']==pair['first_initial_learned_sha256']
    assert state['first_reference_replay']['passed'] and state['first_reference_replay']['optimizer_steps']==0
    expected=read(HERE/'reference_first_batch.json')
    assert state['first_reference_replay']['actual']=={k:expected[k] for k in ['loss','pose_loss','gripper_loss','gradient_norm']}
    assert len(receipt['frozen_restore_events'])==1
    event=receipt['frozen_restore_events'][0]
    assert event['frozen_sha256_after']==pair['frozen_sha256'] and not event['identity_tolerance_used']
    assert not event['learned_parameters_written'] and not event['encoder_parameters_written']
    with np.load(HERE/'reference_prefix.npz') as z:
        assert len(report['updates'])==8192
        for i,row in enumerate(report['updates']):
            assert row['update']==i+1 and row['indices']==z['batches'][i,:int(z['lengths'][i])].tolist()
            assert row['lr']==z['lrs'][i].tolist() and row['valid_dimensions']==z['valid_dimensions'][i].tolist()
            assert np.isfinite([row[k] for k in ['loss','pose_loss','gripper_loss','gradient_norm']]).all()
    if checkpoint:
        sys.path.insert(0,str(HERE/'code'))
        import torch
        from completed_policy_export import contract_sha,model_sha
        payload=torch.load(path/'latest_step.pt',map_location='cpu',weights_only=False,mmap=True)
        assert payload['format']=='fixed-update-training-state-v1'
        c=payload['contract'];assert payload['contract_sha256']==contract_sha(c)==receipt['checkpoint_contract_sha256']
        assert payload['plan']['completed_updates']==8192 and payload['schedule']['total_updates']==65536
        assert payload['extra']['updates']==report['updates'] and payload['extra']['diagnostic_target_state']==state
        assert payload['extra']['validation']==[] and payload['optimizer']['state'] and payload['rng']
        assert model_sha(payload['model'])==receipt['final_learned_sha256']
        canonical=copy.deepcopy(c);d=canonical.pop('diagnostic_target_pair')
        assert d['target']==target and d['short_updates']==8192 and d['lr_horizon']==65536
        assert d['pair_protocol_sha256']==sha(HERE/'pair_protocol.json')
        assert d['hook_source_sha256']==sha(HERE/'hook.py') and d['protocol_loader_sha256']==sha(HERE/'protocols.py')
        assert d['initial_learned_sha256']==pair['first_initial_learned_sha256']
        assert d['loop_source_sha256']==sha(path/'effective_runner.py')
        assert d['diffusion_loss_source_sha256']==sha(path/'effective_diffusion_loss.py')
        canonical['arguments']['diffusion_prediction_type']='epsilon';canonical['config']['model']['diffusion_prediction_type']='epsilon'
        canonical['offline_training_protocol']=c['offline_training_protocol']['source_reference_admission']
        assert canonical==reference
    return receipt


def verify_pair(output,*,checkpoint=True):
    roles={t:arm(output/t,t,checkpoint=checkpoint) for t in ['epsilon','sample']}
    old,new=roles.values()
    assert old['slurm_job_id']==new['slurm_job_id'] and old['node']==new['node']
    assert old['hook_state']['stream_sha256']==new['hook_state']['stream_sha256'],'Timestep/noise/noisy-action streams differ'
    return {'passed':True,'prefix_updates_each':8192,'original_lr_horizon':65536,
        'same_allocation':True,'same_initial_learned_sha256':old['hook_state']['initial_learned_sha256'],
        'identical_timestep_noise_noisy_action_streams':True,'arms':roles,
        'scope':'Training receipt and prefix identity only; generated metrics still required',
        'automatic_following_action':False}


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('output',type=Path);ap.add_argument('--with-evaluation',action='store_true')
    a=ap.parse_args();r=verify_pair(a.output)
    if a.with_evaluation:
        from evaluate import verify_evaluation
        r['evaluation']=verify_evaluation(a.output)
    print('MAIN TARGET SHORT PAIR VERIFIED; diagnostic only')
    print(json.dumps(r,indent=2))

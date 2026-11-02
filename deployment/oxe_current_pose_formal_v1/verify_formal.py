"""Verify saved continuation/evaluation; no new model inference or optimization."""
import argparse
import json
from pathlib import Path
import numpy as np
from binding import HERE,read,sha,attach,admit


def arm(path,role):
    import torch
    from formal_state import FORMAT
    from completed_policy_export import contract_sha,model_sha
    from pose_runtime import digest
    r=read(path/'arm_receipt.json');p=torch.load(path/'latest_state.pt',map_location='cpu',weights_only=False,mmap=True)
    assert r['passed'] and r['role']==role and r['completed_update']==10240 and r['additional_updates']==2048
    assert r['formal_contract_sha256']==sha(HERE/'formal_contract.json')
    assert r['frozen_state_sha256']==read(HERE/'formal_contract.json')['frozen_sha256']
    assert p['format']==FORMAT and p['contract_sha256']==contract_sha(p['contract'])==r['contract_sha256']
    c=p['contract'];assert c['role']==role and c['format']=='current-pose-bounded-formal-augmentation-v1'
    assert c['formal_contract_sha256']==sha(HERE/'formal_contract.json') and c['logical_steps']==2048
    assert c['source_contract_sha256']==read(HERE/'formal_contract.json')['source_contract_sha256']
    assert c['source_learned_sha256']==read(HERE/'formal_contract.json')['source_learned_sha256']
    assert c['base_config_unchanged']['model']['diffusion_prediction_type']=='sample'
    assert p['plan']['completed_updates']==10240 and p['schedule']['total_updates']==65536
    assert model_sha(p['base'])==r['base_learned_sha256'] and digest(p['pose_port'])==r['port_sha256']
    assert set(p['pose_port'])=={'weight'} and tuple(p['pose_port']['weight'].shape)==(1024,18)
    if role=='mask_control':assert torch.count_nonzero(p['pose_port']['weight'][:,:9])==0
    else:assert torch.count_nonzero(p['pose_port']['weight'][:,:9])>0
    assert len(p['optimizer']['param_groups'])==3 and p['optimizer']['state'] and p['rng']
    extra=p['extra'];assert len(extra['rows'])==len(extra['noise_stream'])==2048
    assert extra['initial']==r['initial'] and extra['training_jobs']==r['training_jobs']
    assert r['first_two_updates_match_preflight'] and r['initial']['source_learned_sha256']==c['source_learned_sha256']
    with np.load(Path(__import__('pose_features').HERE)/'reference_suffix.npz') as z:
        for i,row in enumerate(extra['rows']):
            assert row['update']==8193+i and row['indices']==z['batches'][i].tolist()
            assert row['lr_used']==z['lrs'][i].tolist()+[float(z['lrs'][i,0])]
            assert row['valid_dimensions']==z['valid_dimensions'][i].tolist()
            assert np.isfinite([row[k] for k in ['loss','pose_loss','gripper_loss','gradient_norm']]).all()
            assert extra['noise_stream'][i]['step']==row['update']
    return r,extra


def verify_training(output):
    c,ce=arm(output/'mask_control','mask_control');r,re=arm(output/'real_pose','real_pose')
    for key in ['source_optimizer_sha256','source_rng_sha256','initial_port_sha256','source_learned_sha256']:assert c['initial'][key]==r['initial'][key]
    assert ce['noise_stream']==re['noise_stream'] and ce['rows'][0]['base_gradient_sha256_before_clipping']==re['rows'][0]['base_gradient_sha256_before_clipping']
    for key in ['loss','pose_loss','gripper_loss']:assert ce['rows'][0][key]==re['rows'][0][key]
    assert c['initial']['zero_endpoint_replay']['passed']
    jobs=sorted(set(c['training_jobs']+r['training_jobs']))
    return {'passed':True,'source_update':8192,'completed_update':10240,'additional_updates_each':2048,
        'initial_weights_adam_rng_and_training_noise_matched':True,'all_updates_saved':4096,
        'same_training_allocation':len(jobs)==1,'training_job_ids':jobs,'formal_contract_sha256':sha(HERE/'formal_contract.json'),
        'scope':'Training execution checks only; generated evaluation still required'}


def verify_evaluation(output):
    from pair_metrics import summarize
    selection=read(Path(__import__('pose_features').HERE).parent/'oxe_target_pair_short_v1/selection.json')
    p=output/'evaluation';r=read(p/'report.json')
    assert r['execution_complete'] and r['optimizer_updates']==0
    assert r['formal_contract_sha256']==sha(HERE/'formal_contract.json')
    assert r['selection_sha256']==sha(Path(__import__('pose_features').HERE).parent/'oxe_target_pair_short_v1/selection.json')
    assert r['final_frozen_sha256']==read(HERE/'formal_contract.json')['frozen_sha256']
    assert r['generation_steps']==100 and r['sampling_seeds']==[1101,1102,1103] and r['paired_sampling_rng_exact']
    with np.load(p/'predictions_and_targets.npz') as z:
        targets=z['targets'];masks=z['masks'];outputs={k:z[k] for k in ['mask_control','real_pose','unchanged231297','static']}
        assert targets.shape==masks.shape==(120,16,8)
        assert all(v.shape==(3,120,16,8) and np.isfinite(v).all() for v in outputs.values())
        static=np.zeros_like(targets);static[...,6]=1;assert np.array_equal(outputs['static'],np.repeat(static[None],3,0))
        assert summarize(outputs,targets,masks,selection)==r['metrics']
    for role in ['mask_control','real_pose']:
        receipt=read(output/role/'arm_receipt.json')
        assert r['checkpoint_metadata'][role]['base_learned_sha256']==receipt['base_learned_sha256']
        assert r['checkpoint_metadata'][role]['port_sha256']==receipt['port_sha256']
    return {'verified':True,'primary_bridge64':{name:{k:v for k,v in group.items() if k!='per_window'} for name,group in r['metrics']['primary_bridge64'].items()},
        'paired_comparisons':r['metrics']['paired_comparisons'],'gates':r['metrics']['gates'],
        'candidate_for_next_qualification':r['metrics']['candidate_for_next_qualification'],
        'decision':r['metrics']['decision'],'scope':r['metrics']['scope'],'automatic_following_action':False}


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path('/projects/Zeroshot'));ap.add_argument('output',type=Path);ap.add_argument('--with-evaluation',action='store_true')
    a=ap.parse_args();admit(a.root.resolve(),space=False);attach(a.root.resolve());result=verify_training(a.output)
    if a.with_evaluation:result['evaluation']=verify_evaluation(a.output)
    print('CURRENT POSE FORMAL OUTPUT VERIFIED; fixed development diagnostic only')
    print(json.dumps(result,indent=2))

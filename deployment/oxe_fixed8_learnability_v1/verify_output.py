"""Recompute saved fixed8 metrics without inference or training."""
import argparse
import json
from pathlib import Path
import numpy as np
from qualification import HERE,read,sha,admit,attach


def verify(root,output):
    import torch
    from engine import metrics,decision
    from completed_policy_export import model_sha
    from pose_runtime import digest
    p=read(HERE/'protocol.json');r=read(output/'report.json')
    assert r['execution_complete'] and r['additional_optimizer_calls']==p['updates']==512
    assert r['source_update']==10240 and r['endpoint_update']==10752
    assert r['protocol_sha256']==sha(HERE/'protocol.json') and r['source_prediction_replay_exact']
    assert r['source_file_size_mtime_unchanged'] and r['frozen_sha256']==p['source_identity']['frozen_state_sha256']
    assert r['cached_training_guard']['passed'] and r['cached_training_guard']['optimizer_updates']==0
    assert r['cached_training_guard']['native_and_cached_loss_all_gradients_noise_rng_exact']
    assert r['source_binding']==read(HERE/'source_binding.json')
    with np.load(output/'predictions_and_targets.npz') as z,np.load(HERE/'numeric_reference.npz') as ref:
        assert z['targets'].shape==z['masks'].shape==(8,16,8)
        assert np.array_equal(z['targets'],ref['targets']) and np.array_equal(z['masks'],ref['masks'])
        outputs={k:z[k] for k in ['before','after','static']}
        assert all(v.shape==(3,8,16,8) and np.isfinite(v).all() for v in outputs.values())
        with np.load(root/'runs/oxe-pose-pair-232587/evaluation/predictions_and_targets.npz') as old:
            assert np.array_equal(outputs['before'],old['real_pose'][:,:8]) and np.array_equal(outputs['static'],old['static'][:,:8])
        assert metrics(outputs,z['targets'],z['masks'])==r['metrics']
        assert decision(r['metrics'],p['fit_gate'])==r['fit_decision']
    selected=read(HERE/'selection.json');indices=[v['window_index'] for v in selected['rows']]
    assert len(r['history'])==len(r['noise_stream'])==512
    with np.load(HERE/'reference_lrs.npz') as z:
        for i,(row,noise) in enumerate(zip(r['history'],r['noise_stream'])):
            assert row['diagnostic_update']==i+1 and row['source_schedule_update']==10241+i
            assert row['indices']==indices and row['lr_used']==z['lrs'][i].tolist()
            assert noise['step']==10241+i and np.isfinite([row[k] for k in ['loss','pose_loss','gripper_loss','gradient_norm']]).all()
    policy=torch.load(output/'diagnostic_policy.pt',map_location='cpu',weights_only=False,mmap=True)
    assert policy['format']=='current-oxe-fixed8-diagnostic-policy-v1' and policy['protocol_sha256']==sha(HERE/'protocol.json')
    assert policy['diagnostic_updates']==512 and policy['source_update']==10240
    assert policy['source_contract_sha256']==p['source_identity']['contract_sha256']
    assert model_sha(policy['base'])==r['final_base_sha256'] and digest(policy['pose_port'])==r['final_port_sha256']
    return {'output_verified':True,'updates':512,'fixed_training_windows':8,'additional_exposures_per_window':512,
        'metrics':{name:{k:v for k,v in m.items() if k!='per_window'} for name,m in r['metrics'].items()},
        'fit_decision':r['fit_decision'],'scope':r['scope'],'automatic_following_action':False}


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path('/projects/Zeroshot'));ap.add_argument('output',type=Path)
    a=ap.parse_args();root=a.root.resolve();admit(root,space=False);attach(root)
    result=verify(root,a.output)
    print('OXE FIXED8 OUTPUT VERIFIED; repeated training-window diagnostic only')
    print(json.dumps(result,indent=2))

"""Independent endpoint readback; small arrays can also be checked locally."""
import argparse
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE/'code'),str(HERE)]
import numpy as np
from bcap_runtime import read,sha
from bcap_core import summarize


def same(a,b):
    if isinstance(a,dict):
        assert isinstance(b,dict) and set(a)==set(b)
        for k in a:same(a[k],b[k])
    elif isinstance(a,list):
        assert isinstance(b,list) and len(a)==len(b)
        for x,y in zip(a,b):same(x,y)
    elif isinstance(a,float):assert np.isclose(a,b,atol=1e-10,rtol=0),(a,b)
    else:assert a==b,(a,b)


def verify(out,full=True):
    out=Path(out);r=read(out/'report.json');selection=read(HERE/'selection.json')
    assert r['execution_complete'] and r['completed_updates']==65536 and r['training_seed']==42
    assert r['protocol_sha256']==sha(HERE/'protocol.json') and not r['automatic_following_action']
    init=read(out/'initialization.json');guard=read(out/'zero_update_guard.json')
    assert init['empty_adam'] and init['pose_port_zero'] and not init['source_learned_checkpoints_read']
    assert init['identity']==r['initial_identity'] and guard['passed'] and guard['optimizer_updates']==0
    with np.load(out/'predictions_and_targets.npz',allow_pickle=False) as z:
        target=z['targets'];mask=z['masks'];assert target.shape==mask.shape==(192,16,8)
        with np.load(HERE/'evaluation_reference.npz',allow_pickle=False) as ref:
            assert np.array_equal(target,ref['targets']) and np.array_equal(mask,ref['masks'])
        outputs={k:z[k] for k in ['before','after','static']}
        assert all(v.shape==(3,192,16,8) and np.isfinite(v).all() for v in outputs.values())
        static=np.zeros_like(target);static[...,6]=1
        assert np.array_equal(outputs['static'],np.repeat(static[None],3,axis=0))
        same(summarize(outputs,target,mask,selection),r['metrics'])
        with np.load(out/'initial_predictions.npz',allow_pickle=False) as old:
            for k,v in [('before',outputs['before']),('targets',target),('masks',mask)]:assert np.array_equal(old[k],v)
    if full:
        import torch
        from completed_policy_export import model_sha
        from step_training_state import _digest,UpdateSchedule
        s=torch.load(out/'latest_state.pt',map_location='cpu',weights_only=False,mmap=True)
        p=torch.load(out/'final_policy.pt',map_location='cpu',weights_only=False,mmap=True)
        assert s['format']=='fixed-update-training-state-v1' and s['contract_sha256']==_digest(s['contract'])
        assert p['contract']==s['contract'] and s['contract']['protocol_sha256']==r['protocol_sha256']
        assert s['plan']['completed_updates']==p['completed_updates']==65536
        assert s['plan']['contract']=={'identity':sha(HERE/'batches.npy'),'shape':[65536,8]}
        assert {k:model_sha(v) for k,v in s['model'].items()}=={k:model_sha(v) for k,v in p['state'].items()}==r['final_identity']
        assert s['extra']['initial_identity']==r['initial_identity'] and len(s['extra']['history'])==65536
        assert s['extra']['initial_predictions_sha256']==sha(out/'initial_predictions.npz')
        assert len(s['optimizer']['param_groups'])==3 and s['optimizer']['state']
        for v in s['optimizer']['state'].values():
            assert int(v['step'])==65536
            assert all(torch.isfinite(t).all() for t in v.values() if isinstance(t,torch.Tensor))
        assert [g['lr'] for g in s['optimizer']['param_groups']]==UpdateSchedule(**s['schedule']).rates(65536)
        batches=np.load(HERE/'batches.npy',mmap_mode='r');schedule=UpdateSchedule(**s['schedule'])
        for i,row in enumerate(s['extra']['history']):
            assert row['update']==i+1 and row['indices']==batches[i].tolist() and row['lr']==schedule.rates(i)
            assert np.isfinite([row[k] for k in ['loss','pose_loss','gripper_loss','gradient_norm']]).all()
    return {'verified':True,'arrays_metrics_recomputed':True,'full_policy_adam_plan_checked':full,
        'training_passed':r['metrics']['train']['basic_capability_passed'],
        'development_passed':r['metrics']['validation']['basic_capability_passed'],'automatic_following_action':False}


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('output',type=Path);ap.add_argument('--small-only',action='store_true')
    a=ap.parse_args();print(__import__('json').dumps(verify(a.output,not a.small_only),indent=2))

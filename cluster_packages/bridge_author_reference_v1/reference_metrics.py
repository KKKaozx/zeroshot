"""Same physical pose metrics and fitting thresholds as the frozen project audit."""
import numpy as np

FIELDS=('path_position_cm','path_rotation_deg','endpoint_position_cm','endpoint_rotation_deg')
def summarize(pred,target):
    assert np.isfinite(pred).all() and np.isfinite(target).all()
    a=pred[...,3:7].astype(float);b=target[None,...,3:7].astype(float)
    assert np.linalg.norm(a,axis=-1).min()>1e-8 and np.linalg.norm(b,axis=-1).min()>1e-8
    a/=np.linalg.norm(a,axis=-1,keepdims=True);b/=np.linalg.norm(b,axis=-1,keepdims=True)
    position=np.linalg.norm(pred[...,:3].astype(float)-target[None,...,:3],axis=-1)*10
    rotation=np.degrees(2*np.arccos(np.clip(np.abs((a*b).sum(-1)),0,1)))
    good=((position.mean(-1)<=.5)&(rotation.mean(-1)<=5)).all(0)
    fit=bool(good.mean()>=.9 and (position.mean((1,2))<=.5).all() and (rotation.mean((1,2))<=5).all()
             and (position[...,-1].mean(1)<=1).all() and (rotation[...,-1].mean(1)<=10).all())
    result=dict(zip(FIELDS,map(float,(position.mean(),rotation.mean(),position[...,-1].mean(),rotation[...,-1].mean()))))
    result.update(windows=len(target),passing_windows=int(good.sum()),pose_fit_passed=fit)
    return result

def evaluate_cached(policy,arrays,indices,rows,seeds,batch=8):
    import torch
    policy.eval();targets=np.asarray(arrays['action'][indices]).copy();draws=[]
    for seed in seeds:
        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);blocks=[]
        for start in range(0,len(indices),batch):
            ix=indices[start:start+batch]
            obs={k:torch.from_numpy(np.asarray(arrays[k][ix]).copy()).unsqueeze(1).cuda() for k in ('image','ee_state','language')}
            with torch.no_grad():blocks.append(policy.predict_action(obs)['action_pred'].cpu().numpy())
        draws.append(np.concatenate(blocks))
    pred=np.stack(draws);metric=summarize(pred,targets);static=np.zeros_like(targets);static[...,6]=1
    metric['static']=summarize(static[None],targets);episodes={}
    for j,i in enumerate(indices):
        row=rows[int(i)];key=row['shard']+'::'+str(row['record_index'])
        episodes.setdefault(key,[]).append(j)
    metric['per_episode']=[]
    for key,ix in episodes.items():
        metric['per_episode'].append(dict(episode=key,**summarize(pred[:,ix],targets[ix]),static=summarize(static[None,ix],targets[ix])))
    metric['position_better_than_static_demos']=sum(x['path_position_cm']<x['static']['path_position_cm'] for x in metric['per_episode'])
    return metric,pred,targets

def budget_estimate(step_seconds,sampling_call_upper_seconds,updates=8192):
    #103 train-batch samplings +126 final calls (raw/EMA, train/dev, three seeds, batch8).
    return 1.25*(step_seconds*updates+sampling_call_upper_seconds*(103+126)+600)

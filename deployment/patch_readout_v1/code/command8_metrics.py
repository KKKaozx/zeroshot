"""Saved command errors, never reached-pose or task-success metrics."""
import numpy as np

KEYS=('first_position_normalized','first_rotation_deg','path_position_normalized',
      'path_rotation_deg','endpoint_position_normalized','endpoint_rotation_deg')


def metrics(prediction,target,mask,*,score_gripper=True):
    prediction=np.asarray(prediction)
    if prediction.ndim!=4 or prediction.shape[1:]!=target.shape or target.shape!=mask.shape:
        raise ValueError('Aligned draws,N,H,8 command predictions required')
    if not np.isfinite(prediction).all():raise ValueError('Nonfinite predictions')
    rows=[]
    for i in range(len(target)):
        valid=mask[i,:,:7].all(-1)
        if not valid[0] or not np.array_equal(valid,np.arange(len(valid))<valid.sum()):
            raise ValueError('First command and a nonempty prefix must be valid')
        p=prediction[:,i][:,valid].astype(np.float64);t=target[i,valid].astype(np.float64)
        distance=np.linalg.norm(p[...,:3]-t[None,:,:3],axis=-1)
        q=p[...,3:7];tq=t[:,3:7]
        if (np.linalg.norm(q,axis=-1)<1e-8).any() or (np.linalg.norm(tq,axis=-1)<1e-8).any():raise ValueError('Invalid quaternion')
        q=q/np.linalg.norm(q,axis=-1,keepdims=True);tq=tq/np.linalg.norm(tq,axis=-1,keepdims=True)
        angle=np.degrees(2*np.arccos(np.clip(np.abs((q*tq[None]).sum(-1)),0,1)))
        values=(distance[:,0].mean(),angle[:,0].mean(),distance.mean(),angle.mean(),distance[:,-1].mean(),angle[:,-1].mean())
        rows.append({k:float(v) for k,v in zip(KEYS,values)})
    result={'windows':len(rows),'per_window':rows}
    result.update({k:float(np.mean([r[k] for r in rows])) for k in KEYS})
    accuracy=[]
    if score_gripper:
        for i in range(len(target)):
            valid=mask[i,:,7]>0
            if valid.any():accuracy.append(float(((prediction[:,i,valid,7]>0)==(target[i,valid,7]>0)).mean()))
    result['observed_gripper_command_accuracy']=float(np.mean(accuracy)) if accuracy else None
    return result


def summarize(prediction,target,mask,mean_command):
    static=np.zeros_like(prediction);static[...,6]=1;static[...,7]=1
    mean=np.broadcast_to(mean_command,prediction.shape)
    values={'model':metrics(prediction,target,mask),'zero_command':metrics(static,target,mask,score_gripper=False),
            'train_mean_command':metrics(mean,target,mask)}
    n=len(target);boot=np.random.default_rng(20261031).integers(n,size=(10000,n))
    ci={}
    for name in ('zero_command','train_mean_command'):
        ci[name]={}
        for key in KEYS:
            delta=np.array([a[key]-b[key] for a,b in zip(values['model']['per_window'],values[name]['per_window'])])
            ci[name][key]=np.quantile(delta[boot].mean(1),[.025,.975]).tolist()
    primary=('first_position_normalized','first_rotation_deg')
    means=all(values['model'][key]<values[baseline][key] for key in primary for baseline in ('zero_command','train_mean_command'))
    intervals=all(ci['zero_command'][key][1]<0 for key in primary)
    return {'metrics':values,'model_minus_baseline_group_bootstrap95':ci,
            'first_command_means_better_zero_and_mean':bool(means),
            'first_command_zero_difference_ci_upper_below0':bool(intervals),
            'source_engineering_review_gate_passed':bool(means and intervals),
            'scope':'One source command, normalized recorded units and degrees; fullH16 secondary. No task/transfer-success claim.'}

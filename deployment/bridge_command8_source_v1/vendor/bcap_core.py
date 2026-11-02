"""Fresh Bridge capability: fixed budget, atomic state and native loss loop."""
import json
import time
from pathlib import Path
import numpy as np
import torch
from step_training_state import save_checkpoint, restore_checkpoint, capture_rng, restore_rng
from completed_policy_export import model_sha


class FixedPlan:
    def __init__(self, batches, identity):
        self.batches=np.asarray(batches)
        assert self.batches.ndim==2 and self.batches.dtype==np.int64
        self.total_updates=len(batches);self.completed_updates=0;self.pending=False
        self.contract={'identity':identity,'shape':list(batches.shape)}
    def begin_batch(self):
        assert not self.pending and self.completed_updates<self.total_updates
        self.pending=True
        return self.batches[self.completed_updates].tolist()
    def commit_batch(self):
        assert self.pending
        self.completed_updates+=1;self.pending=False
    def state_dict(self):
        assert not self.pending
        return {'contract':self.contract,'completed_updates':self.completed_updates}
    def validate_state(self,s):
        assert not self.pending and s['contract']==self.contract
        assert type(s['completed_updates']) is int and 0<=s['completed_updates']<=self.total_updates
    def load_state_dict(self,s):
        self.validate_state(s);self.completed_updates=s['completed_updates']


def state(b):
    return {'base':b['api'].trainable_state_dict(b['model']), 'pose_port':b['port'].state_dict()}


def load_state(b,s):
    expected=state(b)
    assert set(s)==set(expected)
    for group in expected:
        assert set(s[group])==set(expected[group])
        assert all(s[group][k].shape==v.shape and s[group][k].dtype==v.dtype for k,v in expected[group].items())
    r=b['model'].load_state_dict(s['base'],strict=False)
    assert not r.unexpected_keys and not set(expected['base']).intersection(r.missing_keys)
    b['port'].load_state_dict(s['pose_port'],strict=True)


def identity(b):
    return {k:model_sha(v) for k,v in state(b).items()}


def backward(b,indices):
    model=b['model'];port=b['port'];model.train();port.train()
    b['optimizer'].zero_grad(set_to_none=True)
    x=b['batch'](indices)
    context=model.get_context_vector(x['images'],x['tokens']['input_ids'],x['tokens']['attention_mask'])+port(x['pose'])
    out=model.diffusion_loss(x['actions'],context,x['current'])
    assert model.diffusion_prediction_type=='sample' and torch.equal(out[1],x['actions'][...,:7])
    loss,pose,grip=b['api'].policy_loss(model,out,x['actions'],torch.nn.MSELoss(),
        current_grippers=x['current'],supervision_masks=x['masks'])
    assert torch.isfinite(loss)
    loss.backward()
    parameters=[p for p in model.parameters() if p.requires_grad]+list(port.parameters())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in parameters)
    assert port.weight.grad is not None and torch.isfinite(port.weight.grad).all()
    assert all(not p.requires_grad and p.grad is None for enc in [model.vision_encoder,model.text_encoder] for p in enc.parameters())
    return {'loss':float(loss.detach()),'pose_loss':float(pose.detach()),'gripper_loss':float(grip.detach())},parameters


def zero_guard(b):
    from pose_runtime import digest
    rng=capture_rng(include_cuda=b['cuda']);before=identity(b);adam=digest(b['optimizer'].state_dict());records=[]
    for _ in range(2):
        restore_rng(rng)
        row,params=backward(b,b['plan'].batches[0].tolist())
        row['gradients']=digest([p.grad for p in params]);row['rng']=digest(capture_rng(include_cuda=b['cuda']))
        records.append(row)
    assert records[0]==records[1] and identity(b)==before and digest(b['optimizer'].state_dict())==adam
    assert not b['optimizer'].state and b['plan'].completed_updates==0
    b['optimizer'].zero_grad(set_to_none=True);restore_rng(rng)
    return {'passed':True,'optimizer_updates':0,'repeat_loss_gradient_rng_exact':True,'first_step':records[0]}


def save(b,path,extra):
    save_checkpoint(path,model_state=state(b),optimizer=b['optimizer'],plan=b['plan'],schedule=b['schedule'],
        contract=b['contract'],include_cuda=b['cuda'],extra=extra)


def resume(b,path):
    extra=restore_checkpoint(path,model_state_loader=lambda s:load_state(b,s),optimizer=b['optimizer'],
        plan=b['plan'],schedule=b['schedule'],contract=b['contract'])
    from pose_runtime import check_moments
    check_moments(b['optimizer'])
    assert len(extra['history'])==b['plan'].completed_updates
    return extra


def train(b,folder,extra,*,stop=None):
    """stop is a test-only/in-process boundary; production always uses total budget."""
    folder=Path(folder);plan=b['plan'];opt=b['optimizer'];schedule=b['schedule']
    end=plan.total_updates if stop is None else stop
    assert plan.completed_updates<=end<=plan.total_updates
    tick=time.monotonic();history=extra['history'];used=extra.get('training_seconds',0.)
    assert len(history)==plan.completed_updates
    while plan.completed_updates<end:
        if used+time.monotonic()-tick>b['training_cap_seconds'] or b.get('request_stop',False):
            extra['training_seconds']=used+time.monotonic()-tick;save(b,folder/'latest_state.pt',extra)
            raise TimeoutError('Bounded training cap reached; checkpoint retained; no automatic continuation')
        i=plan.completed_updates;indices=plan.begin_batch();lr=[g['lr'] for g in opt.param_groups]
        assert lr==schedule.rates(i)
        row,params=backward(b,indices)
        if i==0:
            from pose_runtime import digest
            first=extra['zero_update_guard']['first_step']
            assert all(row[k]==first[k] for k in ['loss','pose_loss','gripper_loss'])
            assert digest([p.grad for p in params])==first['gradients']
            assert digest(capture_rng(include_cuda=b['cuda']))==first['rng']
        norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
        opt.step();plan.commit_batch();schedule.apply(opt,plan.completed_updates)
        row.update(update=i+1,indices=indices,lr=lr,gradient_norm=float(norm));history.append(row)
        if plan.completed_updates%b['checkpoint_every']==0 or plan.completed_updates==end:
            b['check_frozen']();extra['training_seconds']=used+time.monotonic()-tick
            save(b,folder/'latest_state.pt',extra)
        if i<2 or (i+1)%256==0:print('BRIDGE_CAPABILITY_UPDATE',json.dumps(row),flush=True)
    extra['training_seconds']=used+time.monotonic()-tick
    return extra


def summarize(outputs,target,mask,selection):
    from bridge_metrics import metrics, KEYS
    rows=selection['rows'];result={}
    for role in ['train','validation']:
        ids=[i for i,r in enumerate(rows) if r['role']==role]
        assert len({rows[i]['group_id'] for i in ids})==len(ids)
        values={k:metrics(v,target,mask,ids) for k,v in outputs.items()}
        rng=np.random.default_rng(20261028);boot=rng.integers(len(ids),size=(10000,len(ids)))
        ci={}
        for key in KEYS:
            delta=np.array([a[key]-s[key] for a,s in zip(values['after']['per_window'],values['static']['per_window'])])
            ci[key]=np.quantile(delta[boot].mean(1),[.025,.975]).tolist()
        result[role]={'metrics':values,'after_minus_static_bootstrap95':ci,
            'basic_capability_passed':bool(all(values['after'][k]<values['static'][k] for k in KEYS)
                and ci['path_position_normalized'][1]<0),
            'uncertainty':'Reused independent evaluation groups; one training seed; sampling seeds are not training seeds'}
        diagnostics={}
        for name in ['before','after']:
            per=[]
            for i in ids:
                valid=mask[i,:,:3].all(-1);p=outputs[name][:,i,valid,:3].astype(np.float64);t=target[i,valid,:3].astype(np.float64)
                pn=np.linalg.norm(p,axis=-1);tn=np.linalg.norm(t,axis=-1)
                ok=(pn>1e-8)&(tn[None,:]>1e-8)
                cos=(p*t).sum(-1)/np.maximum(pn*tn[None,:],1e-16)
                gm=mask[i,:,7]>.5
                accuracy=float(((outputs[name][:,i,gm,7]>0)==(target[i,gm,7]>0)).mean()) if gm.any() else None
                per.append({'selection_index':int(i),'prediction_amplitude':float(pn.mean()),'target_amplitude':float(tn.mean()),
                    'direction_cosine':float(cos[ok].mean()) if ok.any() else None,'gripper_accuracy':accuracy,
                    'numerically_static_target':bool((tn<=1e-8).all())})
            diagnostics[name]={'per_window':per,'gripper_accuracy_window_equal':float(np.mean([r['gripper_accuracy'] for r in per if r['gripper_accuracy'] is not None])),
                'gripper_scope':'Observed masked commands only; no static/unknown-gripper baseline or grasp-success claim',
                'amplitude_ratio_of_window_equal_means':float(np.mean([r['prediction_amplitude'] for r in per])/max(np.mean([r['target_amplitude'] for r in per]),1e-16)),
                'numeric_static_windows':sum(r['numerically_static_target'] for r in per),
                'target_amplitude_quantiles':np.quantile([r['target_amplitude'] for r in per],[0,.25,.5,.75,1]).tolist()}
        result[role]['motion_and_gripper']=diagnostics
    result['automatic_following_action']=False
    return result

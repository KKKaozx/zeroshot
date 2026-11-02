"""One t0teacher-style input response; original native t0response is reused."""
import numpy as np
import torch


def initial_noise(seed,role,device):
    from rtrace_engine import advance_sample_rng
    torch.manual_seed(seed)
    if torch.device(device).type=='cuda':torch.cuda.manual_seed_all(seed)
    advance_sample_rng(device,0 if role=='train' else 16)
    return torch.randn(8,16,7,device=device)


def require_original_noise(noise,expected):
    assert np.array_equal(noise[...,3:7].detach().cpu().numpy(),expected),'Original initial Gaussian quaternion did not replay exactly'


def teacher_forward(model,context,actions,noise):
    from models import extract
    assert model.diffusion_prediction_type=='sample' and model.num_diffusion_steps==100
    assert actions.shape==noise.shape==(8,16,7)
    t=torch.zeros(8,dtype=torch.long,device=actions.device)
    alpha=extract(model.alpha_bars,t,actions)
    x=alpha.sqrt()*actions+(1-alpha).sqrt()*noise
    # Same pose decoder call as native diffusion_loss, with t fixed instead of randomly selected.
    with torch.no_grad():raw=model.diffusion_decoder(x,t,context)
    assert raw.shape==actions.shape and torch.isfinite(raw).all()
    return x,raw


def summarize(arrays):
    target=arrays['targets'].astype(np.float64);mask=arrays['masks']>.5
    qtarget=target[...,3:7];qtarget=qtarget/np.linalg.norm(qtarget,axis=-1,keepdims=True)
    valid=mask[...,3:7].all(-1);allrows={}
    qoutputs={'teacher_noisy_input':arrays['teacher_input'][...,3:7],
        'teacher_denoised_output':arrays['teacher_raw'][...,3:7],'existing_native_raw_t0':arrays['native_raw_t0']}
    qoutputs['identity']=np.zeros_like(qoutputs['teacher_noisy_input']);qoutputs['identity'][...,3]=1
    for name,value in qoutputs.items():
        value=value.astype(np.float64);norm=np.linalg.norm(value,axis=-1)
        assert (norm>1e-6).all(),'Undefined rotation in raw low-t response; stop instead of silently assigning a favorable score'
        q=value/norm[...,None];dot=np.abs((q*qtarget).sum(-1))
        angle=np.degrees(2*np.arccos(np.clip(dot,0,1)))
        err=np.square(value-target[...,3:7]).sum(-1)/4
        allrows[name]={'angle':(angle*valid).sum(-1)/valid.sum(-1),
            'component_mse':(err*valid).sum(-1)/valid.sum(-1),'norm':norm}
    roles={}
    for role,ids in [('train',list(range(8))),('validation',list(range(8,16)))]:
        values={name:{'angle_deg':float(v['angle'][:,ids].mean()),'component_mse':float(v['component_mse'][:,ids].mean()),
            'per_window_angle_deg':v['angle'][:,ids].mean(0).tolist(),
            'norm_quantiles_valid':np.quantile(v['norm'][:,ids][np.broadcast_to(valid[ids],v['norm'][:,ids].shape)],[0,.01,.5,.99,1]).tolist()} for name,v in allrows.items()}
        comparisons={};boot=np.random.default_rng(20261030).integers(8,size=(10000,8))
        out=allrows['teacher_denoised_output']['angle'][:,ids].mean(0)
        for ref in ['teacher_noisy_input','existing_native_raw_t0','identity']:
            d=out-allrows[ref]['angle'][:,ids].mean(0)
            comparisons['teacher_output_minus_'+ref]={'mean_deg':float(d.mean()),'per_window_deg':d.tolist(),
                'paired8group_bootstrap95':np.quantile(d[boot].mean(1),[.025,.975]).tolist(),'windows_improved':int((d<0).sum())}
        residual=np.square(arrays['teacher_raw'][:,ids].astype(np.float64)-target[ids,...,:7])*mask[ids,...,:7]
        denom=mask[ids,...,:7].sum()
        mse={'pose_mse_three_draw_mean_same_batch_denominator':float(residual.sum()/3/denom),
            'xyz_contribution':float(residual[...,:3].sum()/3/denom),'quaternion_contribution':float(residual[...,3:7].sum()/3/denom)}
        roles[role]={'windows':8,'rotation':values,'comparisons':comparisons,'teacher_eval_pose_loss_components':mse}
    return {'roles':roles,'t':0,'scope':'Same checkpoint/conditions,teacher-style t0input versus archived native t0output. One expert continuation; eval mode,not training dropout replay or a new policy.',
        'optimizer_updates':0,'original_policy_and_gates_unchanged':True,'automatic_following_action':False}

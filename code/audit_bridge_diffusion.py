"""Independent diffusion contract audit, without optimizer steps or learned success claims.

Oracle denoisers deliberately know x0: they test sampler mathematics, not prediction.
NumPy float64 references implement DDPM equations independently of production buffers.
"""
import os
os.environ.setdefault('HF_HUB_OFFLINE','1')
import argparse,copy,hashlib,json,math
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from models import RobotAdapterModel,diffusion_betas
from train import policy_loss,set_seed
from diagnose_bridge_modules import save_json,metrics


def reference_schedule(steps,kind):
    if kind=='linear':beta=np.linspace(1e-4,.02,steps,dtype=np.float64)
    else:
        f=lambda u:math.cos((u+.008)/1.008*math.pi/2)**2
        beta=np.array([min(1-f((i+1)/steps)/f(i/steps),.999) for i in range(steps)])
    alpha=1-beta;abar=np.cumprod(alpha);previous=np.r_[1.,abar[:-1]]
    return {'betas':beta,'alphas':alpha,'alpha_bars':abar,
        'posterior_variance':beta*(1-previous)/(1-abar),
        'posterior_mean_x0':beta*np.sqrt(previous)/(1-abar),
        'posterior_mean_xt':np.sqrt(alpha)*(1-previous)/(1-abar)}


class Recorder(nn.Module):
    def forward(self,x,t,c):
        self.x=x.detach().clone();self.t=t.detach().clone();return torch.zeros_like(x)


class Oracle(nn.Module):
    def __init__(self,target,abar,kind):
        super().__init__();self.target=target;self.abar=abar;self.kind=kind;self.calls=[]
    def forward(self,x,t,c):
        self.calls.append(int(t[0]))
        if self.kind=='sample':return self.target
        a=self.abar[t].reshape(-1,1,1)
        return (x-a.sqrt()*self.target)/(1-a).sqrt()


def reference_sample(target,schedule,device,seed,clip,limit):
    generator=torch.Generator(device=device).manual_seed(seed)
    draw=lambda:torch.randn(target.shape,device=device,generator=generator).cpu().numpy().astype(np.float64)
    x=draw();x0=target.cpu().numpy().astype(np.float64)
    for t in reversed(range(len(schedule['betas']))):
        clean=x0.copy()
        if clip:
            clean[:,:,:3]=np.clip(clean[:,:,:3],-limit,limit);clean[:,:,3:]=np.clip(clean[:,:,3:],-1,1)
        mean=schedule['posterior_mean_x0'][t]*clean+schedule['posterior_mean_xt'][t]*x
        x=mean+math.sqrt(schedule['posterior_variance'][t])*draw() if t else mean
    if limit is not None:x[:,:,:3]=np.clip(x[:,:,:3],-limit,limit)
    q=x[:,:,3:7];norm=np.linalg.norm(q,axis=-1,keepdims=True);identity=np.zeros_like(q);identity[:,:,3]=1
    x[:,:,3:7]=np.where(norm>1e-6,q/np.maximum(norm,1e-12),identity)
    return x


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',default='results/bridge_single_task_full_validation_v1/latest.pt')
    parser.add_argument('--data-cache',default='results/bridge_pooling_readout_v1/pooling_context_cache.pt')
    parser.add_argument('--output-dir',default='results/bridge_diffusion_audit_v1')
    parser.add_argument('--cache-dir',default='D:/ntu_related/dissertation/hf_cache')
    args=parser.parse_args();out=Path(args.output_dir)
    if out.exists() and any(out.iterdir()):raise ValueError('Use a fresh output directory')
    torch.set_num_threads(4);set_seed(42)
    source=Path(args.checkpoint);source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
    checkpoint=torch.load(source,map_location='cpu',weights_only=False)
    cache=torch.load(args.data_cache,map_location='cpu',weights_only=False)
    if cache['dataset_identity']!=checkpoint['dataset_identity'] or cache['source_checkpoint_sha256']!=source_hash:
        raise ValueError('Cache provenance mismatch')
    if cache['indices']!=checkpoint['split_indices']['train']+checkpoint['split_indices']['validation']:raise ValueError('Partition mismatch')
    config=copy.deepcopy(checkpoint['config']);config['model']['decoder_type']='diffusion'
    if not config['model']['separate_gripper_head'] or config['model']['gripper_target_mode']!='state':raise ValueError('Audit scope requires separate/state gripper')
    if config['model']['num_diffusion_steps']!=100 or config['model']['beta_schedule']!='squaredcos_cap_v2':
        raise ValueError('This audit scope requires the 100-step cosine configuration')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=RobotAdapterModel(config,cache_dir=args.cache_dir).to(device).eval()
    original_keys=set(model.state_dict())
    model.load_state_dict({k:v for k,v in checkpoint['trainable_state_dict'].items() if k in original_keys},strict=False)
    parameters_before={k:p.detach().cpu().clone() for k,p in model.named_parameters()}
    decoder=model.diffusion_decoder
    indices=[j for j in range(len(checkpoint['split_indices']['train'])) if bool(cache['targets'][j,:,:3].abs().le(3).all())][:4]
    actions=cache['targets'][indices].to(device);context=cache['contexts']['cls'][indices].to(device);current=cache['current'][indices].to(device)
    checks={};issues=[]
    def add(name,passed,**details):
        checks[name]={'passed':bool(passed),**details};print('[扩散审计]',name,'PASS' if passed else 'FAIL',flush=True)
    reference=reference_schedule(model.num_diffusion_steps,model.beta_schedule)
    errors={k:float(np.max(np.abs(getattr(model,k).cpu().numpy()-v))) for k,v in reference.items()}
    add('cosine_schedule_and_posterior',max(errors.values())<3e-4,max_abs_errors=errors,tolerance=3e-4,
        terminal_alpha_bar=float(reference['alpha_bars'][-1]))
    schedules={kind:{'alpha_bar_terminal':float(reference_schedule(100,kind)['alpha_bars'][-1]),
        'retained_signal_amplitude':float(np.sqrt(reference_schedule(100,kind)['alpha_bars'][-1]))} for kind in ['linear','squaredcos_cap_v2']}
    add('schedule_helper_independent_reference',all(np.max(np.abs(diffusion_betas(100,k).numpy()-reference_schedule(100,k)['betas']))<2e-7 for k in schedules),schedules=schedules)
    spy=Recorder();model.diffusion_decoder=spy
    seen_gripper=[];original_gripper=model.predict_gripper_logits
    def grip_spy(c,pose,g=None):
        seen_gripper.append(pose.detach().clone());return original_gripper(c,pose,g)
    model.predict_gripper_logits=grip_spy
    for kind in ['epsilon','sample']:
        model.diffusion_prediction_type=kind;seed=991;set_seed(seed)
        expected_t=torch.randint(0,model.num_diffusion_steps,(len(actions),),device=device)
        noise=torch.randn_like(actions[:,:,:7]);set_seed(seed)
        output=model.diffusion_loss(actions,context,current)
        t=spy.t.cpu().numpy();a=reference['alpha_bars'][t,None,None]
        expected=np.sqrt(a)*actions[:,:,:7].cpu().numpy()+np.sqrt(1-a)*noise.cpu().numpy()
        gap=float(np.max(np.abs(expected-spy.x.cpu().numpy())))
        target=noise if kind=='epsilon' else actions[:,:,:7]
        add('forward_'+kind,bool(torch.equal(spy.t,expected_t) and torch.equal(output[1],target)) and gap<3e-6,
            max_noisy_input_error=gap,tolerance=3e-6,timesteps=t.tolist(),teacher_pose_exact=bool(torch.equal(seen_gripper[-1],actions[:,:,:7])))
        add('teacher_gripper_'+kind,torch.equal(seen_gripper[-1],actions[:,:,:7]))
    # Independent posterior-mean identity, including all 100 timesteps, in double precision.
    rng=np.random.default_rng(27);x0=rng.normal(size=(100,7));eps=rng.normal(size=(100,7));abar=reference['alpha_bars'][:,None]
    xt=np.sqrt(abar)*x0+np.sqrt(1-abar)*eps
    mean_eps=(xt-reference['betas'][:,None]*eps/np.sqrt(1-abar))/np.sqrt(reference['alphas'][:,None])
    mean_post=reference['posterior_mean_x0'][:,None]*x0+reference['posterior_mean_xt'][:,None]*xt
    recovered=(xt-np.sqrt(1-abar)*eps)/np.sqrt(abar)
    add('independent_x0_and_mean_identity',np.max(np.abs(mean_eps-mean_post))<1e-9 and np.max(np.abs(recovered-x0))<1e-9,
        x0_max_error=float(np.max(np.abs(recovered-x0))),mean_max_error=float(np.max(np.abs(mean_eps-mean_post))))
    oracle_results={}
    for kind in ['epsilon','sample']:
        for clip in [False,True]:
            model.diffusion_prediction_type=kind;model.clip_denoised=clip
            oracle=Oracle(actions[:,:,:7],model.alpha_bars,kind);model.diffusion_decoder=oracle
            seed=209;set_seed(seed);result=model.sample(context,current)
            expected=reference_sample(actions[:,:,:7],reference,device,seed,clip,model.max_normalized_position)
            gap=float(np.max(np.abs(result[:,:,:7].cpu().numpy()-expected)))
            target_error=float((result[:,:,:7]-actions[:,:,:7]).abs().max())
            name=f'oracle_{kind}_clip_{clip}';oracle_results[name]={'reference_max_error':gap,'target_max_error':target_error}
            add(name,gap<3e-3 and target_error<3e-3 and oracle.calls==list(reversed(range(100))),
                **oracle_results[name],tolerance=3e-3,steps=len(oracle.calls))
            add('sample_gripper_pose_'+name,torch.equal(seen_gripper[-1],result[:,:,:7]))
    # Deliberate out-of-range and degenerate quaternion fixtures, not dataset edits.
    boundary=actions[:,:,:7].clone();boundary[:,:,:3]=4.;boundary[:,:,3:7]=0.
    model.diffusion_prediction_type='sample';model.clip_denoised=True;model.diffusion_decoder=Oracle(boundary,model.alpha_bars,'sample')
    set_seed(209);boundary_result=model.sample(context,current)
    expected_q=torch.zeros_like(boundary_result[:,:,3:7]);expected_q[:,:,3]=1.
    add('position_clip_and_zero_quaternion_fallback',bool(boundary_result[:,:,:3].abs().max()<=3) and torch.equal(boundary_result[:,:,3:7],expected_q),
        max_position=float(boundary_result[:,:,:3].abs().max()),position_min=float(boundary_result[:,:,:3].min()),
        quaternion_identity_exact=bool(torch.equal(boundary_result[:,:,3:7],expected_q)))
    logits=original_gripper(context,boundary_result[:,:,:7],current)
    add('state_gripper_threshold_and_range',torch.equal(boundary_result[:,:,7],torch.where(logits>=0,1.,-1.)),
        states=sorted(set(boundary_result[:,:,7].cpu().flatten().tolist())))
    model.predict_gripper_logits=original_gripper;model.diffusion_decoder=decoder
    model.diffusion_prediction_type=config['model']['diffusion_prediction_type'];model.clip_denoised=config['model']['clip_denoised']
    # Loss masking independently checked, including exact zero gradients on masked entries.
    predicted=torch.randn_like(actions[:,:,:7],requires_grad=True);target=torch.randn_like(predicted)
    mask=torch.ones_like(actions);mask[:,::2,3:7]=0.;mask[:,:,7]=0.
    grip_logits=torch.full_like(actions[:,:,7],100.,requires_grad=True)
    total,pose,grip=policy_loss(model,(predicted,target,grip_logits),actions,nn.MSELoss(),current,mask)
    expected=((predicted.detach().double()-target.double()).square()*mask[:,:,:7].double()).sum()/mask[:,:,:7].sum()
    grads=torch.autograd.grad(total,(predicted,grip_logits),allow_unused=True)
    grip_has_no_gradient=grads[1] is None or bool(grads[1].eq(0).all())
    add('separate_pose_and_gripper_masks',abs(float(pose.detach())-float(expected))<2e-6 and float(grip.detach())==0 and bool(grads[0][mask[:,:,:7]==0].eq(0).all()) and grip_has_no_gradient,
        masked_pose_loss=float(pose.detach()),reference_loss=float(expected),masked_gripper_loss=float(grip.detach()))
    # Actual initialized temporal U-Net and actual sampling, without an oracle.
    set_seed(314);noisy=torch.randn(2,16,7,device=device);c=context[:2].detach().clone().requires_grad_(True)
    t=torch.tensor([0,99],device=device);prediction=decoder(noisy,t,c)
    gradient=torch.autograd.grad(prediction.square().mean(),c)[0]
    changed=decoder(noisy,t,c.detach()+.1)
    changed_time=decoder(noisy,torch.tensor([99,0],device=device),c.detach())
    add('unet_shape_condition_and_time_connectivity',prediction.shape==noisy.shape and bool(torch.isfinite(prediction).all()) and bool(torch.isfinite(gradient).all()) and bool(gradient.abs().max()>0) and bool((changed-prediction).abs().max()>1e-6) and bool((changed_time-prediction).abs().max()>1e-6),
        output_shape=list(prediction.shape),max_context_gradient=float(gradient.abs().max()),max_context_effect=float((changed-prediction).detach().abs().max()),max_time_effect=float((changed_time-prediction).detach().abs().max()))
    set_seed(805);sample_a=model.sample(context[:2],current[:2]);set_seed(805);sample_b=model.sample(context[:2],current[:2]);set_seed(806);sample_c=model.sample(context[:2],current[:2])
    add('real_unet_sampling_seed_and_valid_output',torch.equal(sample_a,sample_b) and not torch.equal(sample_a,sample_c) and bool(torch.isfinite(sample_a).all()) and bool(sample_a[:,:,:3].abs().max()<=3) and bool(torch.allclose(sample_a[:,:,3:7].norm(dim=-1),torch.ones_like(sample_a[:,:,7]),atol=2e-6)),
        same_seed_max_error=float((sample_a-sample_b).abs().max()),different_seed_max_change=float((sample_a-sample_c).abs().max()),output_shape=list(sample_a.shape))
    # Compatibility probe beyond the current separate-head configuration.
    generic=SimpleNamespace(decoder_type='diffusion',gripper_loss_weight=.25)
    eight_pred=torch.zeros_like(actions);eight_target=torch.ones_like(actions)
    eight_nomask=policy_loss(generic,(eight_pred,eight_target),actions,nn.MSELoss())[1]
    try:
        policy_loss(generic,(eight_pred,eight_target),actions,nn.MSELoss(),supervision_masks=torch.ones_like(actions))
        compatibility={'eight_dim_masked_loss_works':True}
    except RuntimeError as exc:
        compatibility={'eight_dim_masked_loss_works':False,'exception':str(exc),'eight_dim_without_mask_loss':float(eight_nomask)}
        issues.append({'kind':'confirmed_compatibility_bug','current_separate_gripper_path_affected':False,
            'location':'code/train.py policy_loss diffusion mask uses :7 for an 8-dimensional output',
            'detail':compatibility['exception']})
    floors={};n=len(checkpoint['split_indices']['train'])
    for part,s in [('train',slice(0,n)),('validation',slice(n,None))]:
        target=cache['targets'][s].numpy().astype(np.float64);bounded=target.copy();bounded[:,:,:3]=np.clip(bounded[:,:,:3],-3,3)
        floors[part]={'position_components_outside_range':int(np.sum(np.abs(target[:,:,:3])>3)),
            'position_targets_outside_range':int(np.any(np.abs(target[:,:,:3])>3,axis=-1).sum()),
            'windows_outside_range':int(np.any(np.abs(target[:,:,:3])>3,axis=(1,2)).sum()),
            'minimum_possible_mean_position_error_cm':metrics(bounded,target)['position_error_cm']}
    model.max_normalized_position=None;model.diffusion_prediction_type='sample';model.clip_denoised=True
    model.diffusion_decoder=Oracle(boundary,model.alpha_bars,'sample');set_seed(209);none_range=model.sample(context,current)
    compatibility['none_position_limit_with_clip_denoised_observed_max']=float(none_range[:,:,:3].abs().max())
    model.max_normalized_position=config['action']['max_normalized_position'];model.diffusion_prediction_type=config['model']['diffusion_prediction_type'];model.clip_denoised=config['model']['clip_denoised'];model.diffusion_decoder=decoder
    if any(not torch.equal(p.detach().cpu(),parameters_before[k]) for k,p in model.named_parameters()):raise ValueError('Diagnostic changed parameters')
    if hashlib.sha256(source.read_bytes()).hexdigest()!=source_hash:raise ValueError('Source checkpoint changed')
    files={str(Path(f).resolve()):hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in ['code/models.py','code/diffusion_decoder.py','code/train.py']}
    report={'purpose':'diffusion_math_and_current_separate_state_contract_audit_not_learned_success','arguments':vars(args),
        'config':config,'source_checkpoint_sha256':source_hash,'source_code_sha256':files,'dataset_identity':checkpoint['dataset_identity'],
        'trained':False,'optimizer_steps':0,'test_targets_used':False,'current_path_passed':all(c['passed'] for c in checks.values()),
        'checks':checks,'confirmed_issues':issues,'compatibility':compatibility,'real_target_clipping_floor':floors,
        'scope':'100-step cosine schedule, 7D diffusion + separate state gripper, original Bridge contract, float32',
        'oracle_windows':[cache['indices'][j] for j in indices],'reference_sources':['https://arxiv.org/html/2006.11239v2','https://arxiv.org/html/2102.09672v1'],
        'limits':['Oracle receives the true target; it only tests mathematics and cannot demonstrate learned performance.',
            'Random U-Net connectivity and seeded output checks do not demonstrate learning or semantic alignment.',
            'Masks ignore invalid loss entries, not information in invalid input dimensions; heterogeneous sources need separate contracts.',
            'No diffusion-trained checkpoint was evaluated; original source uses a regression head.'],
        'verification':{'original_checkpoint_unchanged':True,'model_parameters_unchanged':True,'diagnostic_only_decoder_substitutions_restored':True}}
    out.mkdir(parents=True,exist_ok=True);save_json(out/'audit.json',report)
    print('[完成]',out.resolve(),'current_path_passed=',report['current_path_passed'],'compatibility_issues=',len(issues),flush=True)


if __name__=='__main__':main()

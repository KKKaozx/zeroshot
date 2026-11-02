"""One fixed U-Net comparison on saved CLIP/Adapter conditions, train only."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/panda_frozen_condition_heads_20261015'
PROBE=ROOT/'reports/panda_initial_approach_probe_20261015'
FIT=ROOT/'reports/panda_initial_only_fit_20261015'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')


def inputs():
    with np.load(PROBE/'outputs.npz') as z:
        visual=z['visual_cls_patch_mean'][:16].astype(float)
    with np.load(FIT/'final_train_predictions.npz') as z:
        target=z['targets'].copy();adapter=z['contexts'].astype(float)
    raw={'clip_pooled':.5*(visual[:,:1024]+visual[:,1024:]),'adapter_initial_only':adapter}
    values={};stats={}
    for name,a in raw.items():
        mean=a.mean(0);scale=float(np.sqrt(np.square(a-mean).mean()))
        assert scale>1e-8
        values[name]=((a-mean)/scale).astype(np.float32)
        stats[name]=dict(mean=mean.tolist(),global_rms_scale=scale)
    return values,target,stats


def make_model(config):
    # Reuse the existing cache-only wrapper and the actual production methods.
    # The original wrapper has a compact head; replace it with the production
    # multiscale class before training. No CLIP/Adapter forward or weight load.
    import importlib.util
    import torch
    import run_panda_spatial_fixed_trial_20261015 as trial
    from replay_and_fit_bridge_positive import head
    from diffusion_decoder import MultiscaleConditionalUnet1D
    path=ROOT/'diagnostics/full_x0/cached_diffusion.py'
    spec=importlib.util.spec_from_file_location('existing_cached_head',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    trial.seed(43);old=module.RobotAdapterModel(config)
    model=head(config,old.state_dict());del old
    trial.seed(43);cfg=config['model']
    model.diffusion_decoder=MultiscaleConditionalUnet1D(action_dim=7,chunk_size=16,context_dim=1024,
        down_dims=tuple(cfg['diffusion_down_dims']),diffusion_step_embed_dim=cfg['diffusion_step_embed_dim'],
        kernel_size=cfg['diffusion_kernel_size'],num_steps=100)
    model.diffusion_architecture='multiscale'
    assert model.sample.__func__ is trial.RobotAdapterModel.sample
    assert model.diffusion_loss.__func__ is trial.RobotAdapterModel.diffusion_loss
    assert sum(p.numel() for p in model.diffusion_decoder.parameters())==77670023
    for name,p in model.named_parameters():p.requires_grad_(name.startswith('diffusion_decoder.'))
    return model


def fit():
    import torch
    import run_panda_spatial_fixed_trial_20261015 as trial
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True,warn_only=True)
    assert torch.cuda.is_available()
    OUT.mkdir(exist_ok=False)
    original=read(trial.OUT/'plan.json');trial.guard(original)
    assert read(ROOT/'reports/panda_feature_readout_20261015/independent_audit.json')['verified']
    values,target,statistics=inputs();selection=read(FIT/'plan.json')['selection']
    assert len(selection)==16 and all(r['start']==0 and r['split']=='train' for r in selection)
    source=[PROBE/'outputs.npz',FIT/'final_train_predictions.npz',FIT/'plan.json',ROOT/'diagnostics/full_x0/cached_diffusion.py']
    config=original['config'].copy();config['model']=config['model'].copy();config['frozen_context_dim']=1024
    # Supply explicit values already assumed by the production constructor.
    config['model'].setdefault('diffusion_step_embed_dim',128);config['model'].setdefault('diffusion_kernel_size',5)
    schedule=torch.cat([torch.randperm(128,generator=torch.Generator().manual_seed(440001+c)).reshape(16,8) for c in range(512)]).flatten()
    schedule=(schedule[schedule%8==0]//8).reshape(1024,8)
    assert torch.equal(torch.bincount(schedule.flatten()),torch.full((16,),512))
    manifest=dict(stage='cached_condition_pose_unet_pair',entry_sha256=sha(Path(__file__)),
        production_source_sha256=original['source_sha256'],source_sha256={str(p.relative_to(ROOT)):sha(p) for p in source},
        config=config,selection=selection,arms=list(values),initialization=43,updates_per_arm=1024,batch_size=8,
        draws_per_window=512,batches_sha256=hashlib.sha256(schedule.numpy().tobytes()).hexdigest(),
        learning_rates=[dict(first=1,last=512,lr=1e-4),dict(first=513,last=1024,lr=1e-5)],
        sampling_seeds=[1101,1102,1103],native_sampling_steps=100,
        condition_processing='Each representation centered per feature over16 training rows, divided by a single training RMS across rows/features. No target statistics or whitening.',
        condition_statistics=statistics,optimized='Only77670023 U-Net pose parameters; all gripper parameters frozen/uninterpreted',
        loss='Unchanged production diffusion_loss first2 outputs, MSE on7D x0. No gripper loss or Adapter forward.',
        common_randomness='Same batch order and per-step seed420000+step. Hash actual noised actions and timesteps before each forward.',
        stopping='One pair of1024-update arms. No added updates,seeds,condition variants,or physical rollouts.',
        interpretation='Frozen representation comparison with common condition scaling; not an exact replay of the previous joint recipe or proof of a unique faulty module.',
        no_new_encodings=True,development_used=False,test_used=False,physical_rollout=False,
        checkpoint_limit_bytes=1024**3,checkpoint_scope='One final cache-only state per arm; no CLIP/Adapter or optimizer weights; not deployable')
    write(OUT/'plan.json',manifest);(OUT/'entry_snapshot.py').write_bytes(Path(__file__).read_bytes())
    model=make_model(config)
    initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    initial_sha=trial.inputs.module_sha(model,('diffusion_decoder.',))
    frozen_sha=trial.inputs.module_sha(model,('gripper_','current_gripper_projection.'))
    current=[]
    for r in selection:
        with np.load(trial.DATA/'episodes'/(r['episode']+'.npz')) as z:current.append([float(z['currents'][0])])
    current=torch.tensor(current,dtype=torch.float32,device='cuda');target_gpu=torch.from_numpy(target).cuda()
    results={};noise_reference=None
    for name,condition in values.items():
        model.cpu();model.load_state_dict(initial,strict=True);model.cuda()
        assert trial.inputs.module_sha(model,('diffusion_decoder.',))==initial_sha
        context=torch.from_numpy(condition).cuda();params=list(model.diffusion_decoder.parameters())
        optimizer=torch.optim.Adam(params,lr=1e-4);torch.cuda.reset_peak_memory_stats();started=time.monotonic();noise_hashes=[];calls=[]
        def capture(module,args):
            assert not args[2].requires_grad
            h=hashlib.sha256(args[0].detach().cpu().numpy().tobytes()+args[1].detach().cpu().numpy().tobytes()).hexdigest()
            calls.append(h)
        handle=model.diffusion_decoder.register_forward_pre_hook(capture)
        try:
            with (OUT/(name+'_updates.jsonl')).open('w') as log:
                for step,chosen in enumerate(schedule,1):
                    lr=1e-4 if step<=512 else 1e-5
                    for group in optimizer.param_groups:group['lr']=lr
                    model.train();chosen=chosen.cuda();trial.seed(420000+step);calls.clear()
                    output=model.diffusion_loss(target_gpu[chosen],context[chosen],current[chosen])
                    loss=torch.nn.functional.mse_loss(output[0],output[1]);assert torch.isfinite(loss)
                    optimizer.zero_grad(set_to_none=True);loss.backward()
                    assert len(calls)==1;noise_hashes.append(calls[0])
                    norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);optimizer.step()
                    log.write(json.dumps(dict(update=step,loss=float(loss.detach()),lr=lr,gradient_norm=float(norm),noise_and_timestep_sha256=calls[0]))+'\n')
                    if step==1 or step%128==0:
                        log.flush();print('FROZEN_HEAD_UPDATE',name,step,'/1024',float(loss.detach()),'seconds',round(time.monotonic()-started,1),flush=True)
        finally:handle.remove()
        if noise_reference is None:noise_reference=noise_hashes
        else:assert noise_reference==noise_hashes
        model.eval();prediction,metrics=trial.evaluate(model,context,current,target,manifest['sampling_seeds'],batch=2)
        np.savez_compressed(OUT/(name+'_predictions.npz'),predictions=prediction,targets=target,contexts=condition)
        after_sha=trial.inputs.module_sha(model,('diffusion_decoder.',));assert after_sha!=initial_sha
        assert trial.inputs.module_sha(model,('gripper_','current_gripper_projection.'))==frozen_sha
        checkpoint=OUT/(name+'_final.pt')
        state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        torch.save(dict(config=config,state_dict=state,condition=condition,condition_statistics=statistics[name],
            cached_only=True,plan_sha256=sha(OUT/'plan.json'),updates=1024,initial_decoder_sha256=initial_sha),checkpoint)
        assert sum(p.stat().st_size for p in OUT.glob('*_final.pt'))<manifest['checkpoint_limit_bytes']
        # Read back every saved tensor and replay the first original sampling batch.
        payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert all(torch.equal(v,payload['state_dict'][k]) for k,v in state.items())
        model.load_state_dict(initial,strict=True);model.load_state_dict(payload['state_dict'],strict=True)
        trial.seed(1101)
        with torch.no_grad():replay=model.sample(context[:2],current[:2]).cpu().numpy()
        replay_delta=float(np.abs(replay-prediction[0,:2]).max());assert replay_delta==0
        results[name]=dict(updates=1024,metrics=metrics,initial_decoder_sha256=initial_sha,final_decoder_sha256=after_sha,
            trainable_parameters=sum(p.numel() for p in params),frozen_gripper_unchanged=True,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,elapsed_seconds=time.monotonic()-started,
            checkpoint_bytes=checkpoint.stat().st_size,checkpoint_sha256=sha(checkpoint),saved_tensors_exact=True,
            reload_first_batch_prediction_difference=replay_delta)
        write(OUT/'partial_report.json',dict(arms=results,complete=False))
        print('FROZEN_HEAD_FINAL',name,json.dumps(results[name]),flush=True)
        del optimizer,params,output,loss,state,payload;torch.cuda.empty_cache()
    trial.guard(original)
    assert all(sha(ROOT/n)==h for n,h in manifest['source_sha256'].items())
    write(OUT/'report.json',dict(complete=True,arms=results,noise_streams_exact=True,
        no_clip_or_adapter_forward=True,no_clip_or_adapter_optimization=True,development_used=False,test_used=False,physical_rollout=False))


def analyze():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import sys
    sys.path.insert(0,str(ROOT/'scripts'))
    from analyze_bridge_growth64_200600 import summarize
    p,r=read(OUT/'plan.json'),read(OUT/'report.json');assert r['complete'] and r['noise_streams_exact']
    assert sha(Path(__file__))==p['entry_sha256']==sha(OUT/'entry_snapshot.py')
    assert all(sha(ROOT/n)==h for n,h in {**p['production_source_sha256'],**p['source_sha256']}.items())
    values,target,statistics=inputs();initial=[];noise=[];result={}
    for arm,a in r['arms'].items():
        assert a['updates']==1024 and a['frozen_gripper_unchanged'] and a['reload_first_batch_prediction_difference']==0
        assert sha(OUT/(arm+'_final.pt'))==a['checkpoint_sha256']
        initial.append(a['initial_decoder_sha256'])
        log=[json.loads(row) for row in (OUT/(arm+'_updates.jsonl')).read_text().splitlines()]
        assert [row['update'] for row in log]==list(range(1,1025))
        assert all(row['lr']==(1e-4 if row['update']<=512 else 1e-5) and np.isfinite(row['loss']) for row in log)
        noise.append([row['noise_and_timestep_sha256'] for row in log])
        with np.load(OUT/(arm+'_predictions.npz')) as z:
            pred,truth,context=z['predictions'],z['targets'],z['contexts']
        assert np.array_equal(truth,target) and np.array_equal(context,values[arm]) and pred.shape==(3,16,16,8)
        m,good,gate,_,_=summarize(pred,truth)
        assert gate==a['metrics']['pose_fit_passed'] and int(good.sum())==a['metrics']['windows_all_three_seeds_pass']
        assert abs(m['path_position_cm']-a['metrics']['position_cm'])<1e-4
        result[arm]=dict(**m,pose_fit_passed=gate,passing_windows=int(good.sum()))
    assert len(set(initial))==1 and noise[0]==noise[1]
    verdict=dict(verified=True,identical_initial_decoder=True,all1024_noise_inputs_and_timesteps_exact=True,
        conditions_and_targets_exact=True,arms=result,development_used=False,test_used=False,physical_rollout=False)
    write(OUT/'independent_audit.json',verdict)
    fig,ax=plt.subplots(figsize=(8,4),constrained_layout=True)
    ax.bar(['Frozen CLIP pooled','Frozen initial-only Adapter'],[v['path_position_cm'] for v in result.values()])
    ax.axhline(.5,color='#999999',linestyle='--',label='5mm train fitting threshold')
    ax.set(ylabel='Train initial-approach position error (cm)',title='Same fresh U-Net,1024 updates per arm; nativeDDPM100');ax.legend();ax.grid(axis='y',alpha=.2)
    fig.savefig(OUT/'head_pair.png',dpi=170);plt.close(fig)
    print(json.dumps(verdict,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--analyze',action='store_true')
    if parser.parse_args().analyze:analyze()
    else:fit()

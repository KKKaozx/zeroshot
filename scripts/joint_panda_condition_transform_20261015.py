"""One new joint condition-processing arm; reuse the original raw joint trial."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/panda_joint_condition_transform_20261015'
REF=ROOT/'reports/panda_initial_only_fit_20261015'


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def fit():
    import torch
    import run_panda_spatial_fixed_trial_20261015 as trial
    from population_condition_transform import PopulationConditionDecoder
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True,warn_only=True)
    assert torch.cuda.is_available();OUT.mkdir(exist_ok=False)
    baseline=read(REF/'report.json');baseplan=read(REF/'plan.json');original=read(trial.OUT/'plan.json')
    trial.guard(original);assert read(REF/'independent_audit.json')['verified'] and read(REF/'reload_report.json')['predictions_exact']
    images,targets,current,rows=trial.inputs.examples('train');selection=list(range(0,128,8))
    assert [rows[i] for i in selection]==baseplan['selection']
    images=images[selection];targets=targets[selection].cuda();current=current[selection].cuda()
    schedule=torch.cat([torch.randperm(128,generator=torch.Generator().manual_seed(440001+c)).reshape(16,8) for c in range(512)]).flatten()
    schedule=(schedule[schedule%8==0]//8).reshape(1024,8)
    noise_path=ROOT/'reports/panda_frozen_condition_heads_20261015/adapter_initial_only_updates.jsonl'
    noise_ref=[json.loads(s) for s in noise_path.read_text().splitlines()]
    oldlog=[json.loads(s) for s in (REF/'updates.jsonl').read_text().splitlines()]
    assert len(oldlog)==len(noise_ref)==len(schedule)==1024
    assert hashlib.sha256(schedule.numpy().tobytes()).hexdigest()==read(noise_path.parent/'plan.json')['batches_sha256']
    sources=[Path(__file__),ROOT/'code/population_condition_transform.py',noise_path,REF/'plan.json',REF/'report.json',REF/'updates.jsonl',REF/'final_train_predictions.npz']
    manifest=dict(stage='joint_adapter_unet_population_condition',source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources},
        production_source_sha256=original['source_sha256'],config=original['config'],selection=baseplan['selection'],
        reused_raw_joint_run=str(REF.relative_to(ROOT)),reused_control_retrained=False,
        initialization=43,updates=1024,batch_size=8,draws_per_window=512,sampling_seeds=original['sample_seeds'],
        learning_rate='First512 at1e-4,last512 at1e-5; unchanged Adam,clip1,dropout0.1 and joint gripper loss',
        only_changed_interface='U-Net condition=(raw Adapter context - current train-population eval mean)/one global RMS. Raw context still enters the gripper head.',
        calibration='Before each update, eval-mode Adapter on all16 training observations, no gradients, no RNG consumption; detached stats. Final recalibration at final weights, saved and frozen at inference.',
        encoder_cache='Original token sequences were not retained; encode16 once, check existing pooled cache, and retain tokens to avoid future re-encoding.',
        final_selection='Final fixed update only; no intermediate action evaluation or development selection',
        stopping='One1024-update new arm; no budget/seed/normalizer variants or physics after results',
        candidate='Original pose fit gate; otherwise partial/failure, no automatic extension',
        development_used=False,test_used=False,physical_rollout=False,checkpoint_limit_bytes=600*1024**2)
    write(OUT/'plan.json',manifest);(OUT/'entry_snapshot.py').write_bytes(Path(__file__).read_bytes())
    trial.seed(43);model=trial.RobotAdapterModel(original['config']).cuda().eval()
    before={k:trial.inputs.module_sha(model,p) for k,p in trial.GROUPS.items()};assert before==baseline['initial_group_sha256']
    tokenizer=trial.CLIPTokenizer.from_pretrained(original['config']['model']['name'],local_files_only=True)
    text=tokenizer([read(trial.DATA/'protocol.json')['instruction']]*16,padding=True,truncation=True,max_length=77,return_tensors='pt')
    text={k:v.cuda() for k,v in text.items()}
    with torch.no_grad():
        visual=torch.cat([model.vision_encoder(pixel_values=images[i:i+2].cuda()).last_hidden_state for i in range(0,16,2)]).detach()
        language=model.text_encoder(**text).last_hidden_state.detach()
    with np.load(ROOT/'reports/panda_initial_approach_probe_20261015/outputs.npz') as z:
        pooled=torch.cat([visual[:,0],visual[:,1:].mean(1)],dim=1).cpu().numpy()
        token_delta=float(np.abs(pooled-z['visual_cls_patch_mean'][:16]).max());assert token_delta<1e-4
    np.savez_compressed(OUT/'frozen_tokens.npz',visual=visual.cpu().numpy(),language=language.cpu().numpy(),attention_mask=text['attention_mask'].cpu().numpy())
    model.vision_encoder.cpu();model.text_encoder.cpu();torch.cuda.empty_cache()
    # Identity wrapper and a single loss replay certify reuse of the old control.
    model.diffusion_decoder=PopulationConditionDecoder(model.diffusion_decoder).cuda()
    model.diffusion_decoder.enabled=False;model.train();first=schedule[0].cuda();trial.seed(430001)
    ctx=model.adapter(visual[first],language[first],text['attention_mask'][first])
    trial.seed(420001);output=model.diffusion_loss(targets[first],ctx,current[first])
    raw_loss,_,_=trial.policy_loss(model,output,targets[first],torch.nn.MSELoss(),current_grippers=current[first])
    first_loss_delta=abs(float(raw_loss.detach())-oldlog[0]['loss']);assert first_loss_delta<1e-5
    del ctx,output,raw_loss
    model.diffusion_decoder.enabled=True
    trainable=[p for n,p in model.named_parameters() if not n.startswith(trial.GROUPS['clip'])]
    optimizer=torch.optim.Adam(trainable,lr=1e-4);started=time.monotonic();torch.cuda.reset_peak_memory_stats()
    calls=[]
    def capture(module,args):calls.append(hashlib.sha256(args[0].detach().cpu().numpy().tobytes()+args[1].detach().cpu().numpy().tobytes()).hexdigest())
    handle=model.diffusion_decoder.core.register_forward_pre_hook(capture)
    def calibrate():
        model.adapter.eval();rng=torch.cuda.get_rng_state().clone()
        with torch.no_grad():
            values=torch.cat([model.adapter(visual[i:i+2],language[i:i+2],text['attention_mask'][i:i+2]) for i in range(0,16,2)])
            stat=model.diffusion_decoder.calibrate(values)
        assert torch.equal(rng,torch.cuda.get_rng_state())
        return values,stat
    first_grad=None;first_stats=None
    try:
        with (OUT/'updates.jsonl').open('w') as log:
            for step,ix in enumerate(schedule,1):
                _,stat=calibrate()
                if first_stats is None:first_stats=stat
                lr=oldlog[step-1]['lr']
                for group in optimizer.param_groups:group['lr']=lr
                model.train();ix=ix.cuda();trial.seed(430000+step)
                context=model.adapter(visual[ix],language[ix],text['attention_mask'][ix])
                trial.seed(420000+step);calls.clear();output=model.diffusion_loss(targets[ix],context,current[ix])
                loss,pose,grip=trial.policy_loss(model,output,targets[ix],torch.nn.MSELoss(),current_grippers=current[ix])
                assert torch.isfinite(loss) and len(calls)==1 and calls[0]==noise_ref[step-1]['noise_and_timestep_sha256']
                optimizer.zero_grad(set_to_none=True);loss.backward()
                if step==1:
                    first_grad={k:any(p.grad is not None and bool(torch.count_nonzero(p.grad)) for n,p in model.named_parameters() if n.startswith(prefix)) for k,prefix in trial.GROUPS.items()}
                    assert first_grad==dict(clip=False,adapter=True,decoder=True,gripper=True)
                norm=torch.nn.utils.clip_grad_norm_(trainable,1.,error_if_nonfinite=True);optimizer.step()
                row=dict(update=step,loss=float(loss.detach()),pose=float(pose.detach()),gripper=float(grip.detach()),lr=lr,
                    gradient_norm=float(norm),noise_and_timestep_sha256=calls[0],population_scale=stat['scale'],population_mean_norm=stat['mean_norm'],scale_clamped=stat['scale_clamped'])
                log.write(json.dumps(row)+'\n')
                if step==1 or step%128==0:
                    log.flush();print('JOINT_CONDITION_UPDATE',step,'/1024','loss',row['loss'],'scale',stat['scale'],'seconds',round(time.monotonic()-started,1),flush=True)
    finally:handle.remove()
    model.eval();context,final_stats=calibrate()
    with torch.no_grad():
        processed=model.diffusion_decoder.transform(context)
        direct=((context.double()-context.double().mean(0))/((context.double()-context.double().mean(0)).square().mean().sqrt())).float()
        transform_delta=float((processed-direct).abs().max());assert transform_delta<1e-4
        batch_delta=float((torch.cat([model.diffusion_decoder.transform(row[None]) for row in context])-processed).abs().max());assert batch_delta==0
    prediction,metrics=trial.evaluate(model,context,current,targets.cpu().numpy(),original['sample_seeds'],batch=2)
    np.savez_compressed(OUT/'predictions.npz',predictions=prediction,targets=targets.cpu().numpy(),contexts=context.cpu().numpy(),
        normalized_contexts=processed.cpu().numpy(),context_mean=model.diffusion_decoder.context_mean.cpu().numpy(),context_scale=model.diffusion_decoder.context_scale.cpu().numpy())
    state={n:v.detach().cpu().clone() for n,v in model.state_dict().items() if not n.startswith(trial.GROUPS['clip'])}
    checkpoint=OUT/'final.pt';torch.save(dict(config=original['config'],state_dict_nonclip=state,
        condition_transform='train_population_eval_v1_unet_only',plan_sha256=sha(OUT/'plan.json'),updates=1024),checkpoint)
    assert checkpoint.stat().st_size<manifest['checkpoint_limit_bytes']
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False);assert all(torch.equal(v,payload['state_dict_nonclip'][k]) for k,v in state.items())
    missing=model.load_state_dict(payload['state_dict_nonclip'],strict=False)
    assert not missing.unexpected_keys and all(n.startswith(trial.GROUPS['clip']) for n in missing.missing_keys)
    trial.seed(1101)
    with torch.no_grad():replay=model.sample(context[:2],current[:2]).cpu().numpy()
    assert np.array_equal(replay,prediction[0,:2])
    after={k:trial.inputs.module_sha(model,p) for k,p in trial.GROUPS.items()}
    assert before['clip']==after['clip'] and all(before[k]!=after[k] for k in ['adapter','decoder','gripper'])
    assert all(sha(ROOT/n)==h for n,h in manifest['source_sha256'].items());trial.guard(original)
    report=dict(complete=True,updates=1024,metrics=metrics,baseline_metrics=baseline['metrics'],
        first_raw_control_loss_difference=first_loss_delta,initial_model_matches_original=True,
        all1024_noise_inputs_and_timesteps_exact=True,first_step_gradient=first_grad,initial_population=first_stats,final_population=final_stats,
        context_transform_formula_max_difference=transform_delta,context_transform_batch_size_difference=batch_delta,
        frozen_encoder_pooled_token_difference=token_delta,clip_unchanged=True,initial_group_sha256=before,final_group_sha256=after,
        checkpoint_sha256=sha(checkpoint),checkpoint_bytes=checkpoint.stat().st_size,reload_first_batch_prediction_difference=0.,
        peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,elapsed_seconds=time.monotonic()-started,
        baseline_retrained=False,baseline_single_forward_replayed=True,development_used=False,test_used=False,physical_rollout=False)
    write(OUT/'report.json',report);print('JOINT CONDITION FINAL',json.dumps(report,indent=2),flush=True)


def analyze():
    import sys
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sys.path.insert(0,str(ROOT/'scripts'))
    from analyze_bridge_growth64_200600 import summarize
    p,r=read(OUT/'plan.json'),read(OUT/'report.json');assert r['complete'] and not r['baseline_retrained']
    assert sha(OUT/'entry_snapshot.py')==sha(Path(__file__))
    assert all(sha(ROOT/n)==h for n,h in {**p['source_sha256'],**p['production_source_sha256']}.items())
    assert sha(OUT/'final.pt')==r['checkpoint_sha256'] and r['clip_unchanged']
    logs=[json.loads(s) for s in (OUT/'updates.jsonl').read_text().splitlines()]
    old=[json.loads(s) for s in (REF/'updates.jsonl').read_text().splitlines()]
    noise=[json.loads(s) for s in (ROOT/'reports/panda_frozen_condition_heads_20261015/adapter_initial_only_updates.jsonl').read_text().splitlines()]
    assert len(logs)==1024 and all(a['update']==b['update']==i+1 and a['lr']==b['lr'] and np.isfinite(a['loss'])
        and a['noise_and_timestep_sha256']==noise[i]['noise_and_timestep_sha256'] for i,(a,b) in enumerate(zip(logs,old)))
    with np.load(OUT/'predictions.npz') as z:
        pred,target,raw,norm=z['predictions'],z['targets'],z['contexts'],z['normalized_contexts']
        mean,scale=z['context_mean'],z['context_scale']
    with np.load(REF/'final_train_predictions.npz') as z:
        assert np.array_equal(target,z['targets']);previous=z['predictions']
    assert np.abs((raw-mean)/scale-norm).max()<1e-6
    assert np.abs(raw.astype(float).mean(0)-mean).max()<1e-5
    assert abs(np.sqrt(np.square(raw.astype(float)-raw.astype(float).mean(0)).mean())-float(scale))<1e-5
    result={}
    for name,values,record in [('new_processed_joint',pred,r['metrics']),('original_raw_joint_reused',previous,r['baseline_metrics'])]:
        m,good,gate,_,_=summarize(values,target)
        assert gate==record['pose_fit_passed'] and int(good.sum())==record['windows_all_three_seeds_pass']
        assert abs(m['path_position_cm']-record['position_cm'])<1e-4
        result[name]=dict(**m,passing_windows=int(good.sum()),pose_fit_passed=gate)
    verdict=dict(verified=True,groups=result,baseline_retrained=False,
        statistics_use_final_training_contexts_only=True,training_and_inference_transform_verified=True,
        mean_position_relative_change=result['new_processed_joint']['path_position_cm']/result['original_raw_joint_reused']['path_position_cm']-1,
        development_used=False,test_used=False,physical_rollout=False)
    write(OUT/'independent_audit.json',verdict)
    fig,ax=plt.subplots(figsize=(8,4),constrained_layout=True)
    values=[result[n]['path_position_cm']*10 for n in ['original_raw_joint_reused','new_processed_joint']]
    ax.bar(['Raw joint (reused)','Population-scaled U-Net condition'],values);ax.set(ylabel='Train initial approach position error (mm)',title='Joint Adapter/U-Net, same16 windows and1024 updates');ax.grid(axis='y',alpha=.2)
    fig.savefig(OUT/'joint_condition_comparison.png',dpi=170);plt.close(fig)
    print(json.dumps(verdict,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--analyze',action='store_true')
    if parser.parse_args().analyze:analyze()
    else:fit()

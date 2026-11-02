"""Add only the missing raw-Adapter arm; reuse the completed processed arm."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import compare_panda_frozen_conditions_20261015 as pair

ROOT=pair.ROOT
REF=pair.OUT
OUT=ROOT/'reports/panda_raw_condition_control_20261015'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')


def fit():
    import torch
    import run_panda_spatial_fixed_trial_20261015 as trial
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True,warn_only=True)
    assert torch.cuda.is_available();OUT.mkdir(exist_ok=False)
    p,r=read(REF/'plan.json'),read(REF/'report.json');arm=r['arms']['adapter_initial_only']
    assert read(REF/'independent_audit.json')['verified'] and r['complete']
    assert sha(ROOT/'scripts/compare_panda_frozen_conditions_20261015.py')==p['entry_sha256']
    assert p['updates_per_arm']==1024 and p['sampling_seeds']==[1101,1102,1103] and p['initialization']==43
    assert all(sha(ROOT/n)==h for n,h in {**p['production_source_sha256'],**p['source_sha256']}.items())
    with np.load(pair.FIT/'final_train_predictions.npz') as z:condition=z['contexts'].copy();target=z['targets'].copy()
    with np.load(REF/'adapter_initial_only_predictions.npz') as z:
        assert np.array_equal(target,z['targets'])
        stats=p['condition_statistics']['adapter_initial_only']
        reconstructed=((condition.astype(float)-np.array(stats['mean']))/stats['global_rms_scale']).astype(np.float32)
        assert np.array_equal(reconstructed,z['contexts'])
    oldlog=[json.loads(s) for s in (REF/'adapter_initial_only_updates.jsonl').read_text().splitlines()]
    schedule=torch.cat([torch.randperm(128,generator=torch.Generator().manual_seed(440001+c)).reshape(16,8) for c in range(512)]).flatten()
    schedule=(schedule[schedule%8==0]//8).reshape(1024,8)
    assert hashlib.sha256(schedule.numpy().tobytes()).hexdigest()==p['batches_sha256']
    refs=['plan.json','report.json','independent_audit.json','adapter_initial_only_updates.jsonl','adapter_initial_only_predictions.npz']
    manifest=dict(entry_sha256=sha(Path(__file__)),reused_reference_sha256={n:sha(REF/n) for n in refs},
        original_pair_plan_sha256=sha(REF/'plan.json'),production_source_sha256=p['production_source_sha256'],
        source_sha256=p['source_sha256'],selection=p['selection'],config=p['config'],
        new_arm='Raw saved Adapter condition, no centering or scale transform',
        reused_arm='Existing adapter_initial_only processed-condition arm; no training or inference rerun',
        changed_factor='Only remove condition centering/global RMS processing; same frozen Adapter snapshot, targets, U-Net initialization, batches, noise, Adam, rate, clip and native sampling.',
        initialization=43,updates=1024,batch_size=8,draws_per_window=512,learning_rates=p['learning_rates'],
        sampling_seeds=p['sampling_seeds'],train_only=True,development_used=False,test_used=False,physical_rollout=False,
        stop='One raw arm,1024 updates; no extra seeds,updates,scale variants,or closed-loop runs',
        checkpoint_limit_bytes=400*1024**2,gripper='Frozen and not interpreted; pose-only cached diagnostic')
    write(OUT/'plan.json',manifest);(OUT/'entry_snapshot.py').write_bytes(Path(__file__).read_bytes())
    model=pair.make_model(p['config']);initial_sha=trial.inputs.module_sha(model,('diffusion_decoder.',))
    assert initial_sha==arm['initial_decoder_sha256']
    frozen=trial.inputs.module_sha(model,('gripper_','current_gripper_projection.'))
    initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};model.cuda()
    context=torch.from_numpy(condition).cuda();target_gpu=torch.from_numpy(target).cuda();current=[]
    for row in p['selection']:
        with np.load(trial.DATA/'episodes'/(row['episode']+'.npz')) as z:current.append([float(z['currents'][0])])
    current=torch.tensor(current,dtype=torch.float32,device='cuda')
    params=list(model.diffusion_decoder.parameters());optimizer=torch.optim.Adam(params,lr=1e-4)
    calls=[]
    def capture(module,args):
        calls.append(hashlib.sha256(args[0].detach().cpu().numpy().tobytes()+args[1].detach().cpu().numpy().tobytes()).hexdigest())
    handle=model.diffusion_decoder.register_forward_pre_hook(capture)
    torch.cuda.reset_peak_memory_stats();started=time.monotonic()
    try:
        with (OUT/'updates.jsonl').open('w') as log:
            for step,chosen in enumerate(schedule,1):
                lr=oldlog[step-1]['lr']
                for group in optimizer.param_groups:group['lr']=lr
                model.train();chosen=chosen.cuda();trial.seed(420000+step);calls.clear()
                output=model.diffusion_loss(target_gpu[chosen],context[chosen],current[chosen])
                loss=torch.nn.functional.mse_loss(output[0],output[1]);assert torch.isfinite(loss)
                assert len(calls)==1 and calls[0]==oldlog[step-1]['noise_and_timestep_sha256']
                optimizer.zero_grad(set_to_none=True);loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);optimizer.step()
                log.write(json.dumps(dict(update=step,loss=float(loss.detach()),lr=lr,gradient_norm=float(norm),noise_and_timestep_sha256=calls[0]))+'\n')
                if step==1 or step%128==0:
                    log.flush();print('RAW_CONDITION_UPDATE',step,'/1024',float(loss.detach()),'seconds',round(time.monotonic()-started,1),flush=True)
    finally:handle.remove()
    model.eval();prediction,metrics=trial.evaluate(model,context,current,target,p['sampling_seeds'],batch=2)
    np.savez_compressed(OUT/'predictions.npz',predictions=prediction,targets=target,contexts=condition)
    final_sha=trial.inputs.module_sha(model,('diffusion_decoder.',));assert final_sha!=initial_sha
    assert frozen==trial.inputs.module_sha(model,('gripper_','current_gripper_projection.'))
    state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};checkpoint=OUT/'final.pt'
    torch.save(dict(config=p['config'],state_dict=state,condition=condition,cached_only=True,condition_processing='identity',
        plan_sha256=sha(OUT/'plan.json'),updates=1024,initial_decoder_sha256=initial_sha),checkpoint)
    assert checkpoint.stat().st_size<manifest['checkpoint_limit_bytes']
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert all(torch.equal(v,payload['state_dict'][k]) for k,v in state.items())
    model.load_state_dict(initial,strict=True);model.load_state_dict(payload['state_dict'],strict=True);trial.seed(1101)
    with torch.no_grad():replay=model.sample(context[:2],current[:2]).cpu().numpy()
    assert np.array_equal(replay,prediction[0,:2])
    assert all(sha(REF/n)==h for n,h in manifest['reused_reference_sha256'].items())
    trial.guard(read(trial.OUT/'plan.json'))
    report=dict(complete=True,updates=1024,raw_metrics=metrics,processed_reference_metrics=arm['metrics'],
        identical_initial_decoder=True,initial_decoder_sha256=initial_sha,final_decoder_sha256=final_sha,
        all1024_noise_inputs_and_timesteps_exact=True,processed_arm_reexecuted=False,
        frozen_gripper_unchanged=True,checkpoint_bytes=checkpoint.stat().st_size,checkpoint_sha256=sha(checkpoint),
        saved_tensors_exact=True,reload_first_batch_prediction_difference=0.,peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,
        elapsed_seconds=time.monotonic()-started,development_used=False,test_used=False,physical_rollout=False)
    write(OUT/'report.json',report);print('RAW CONDITION FINAL',json.dumps(report,indent=2),flush=True)


def analyze():
    import sys
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sys.path.insert(0,str(ROOT/'scripts'))
    from analyze_bridge_growth64_200600 import summarize
    from audit_bridge_end_to_end_20261015 import errors
    p,r=read(OUT/'plan.json'),read(OUT/'report.json');assert r['complete'] and not r['processed_arm_reexecuted']
    assert sha(Path(__file__))==p['entry_sha256']==sha(OUT/'entry_snapshot.py')
    assert all(sha(REF/n)==h for n,h in p['reused_reference_sha256'].items())
    assert all(sha(ROOT/n)==h for n,h in {**p['production_source_sha256'],**p['source_sha256']}.items())
    newlog=[json.loads(s) for s in (OUT/'updates.jsonl').read_text().splitlines()]
    oldlog=[json.loads(s) for s in (REF/'adapter_initial_only_updates.jsonl').read_text().splitlines()]
    assert len(newlog)==len(oldlog)==1024
    assert all(a['update']==b['update']==i+1 and a['lr']==b['lr'] and a['noise_and_timestep_sha256']==b['noise_and_timestep_sha256']
        and np.isfinite(a['loss']) for i,(a,b) in enumerate(zip(newlog,oldlog)))
    assert sha(OUT/'final.pt')==r['checkpoint_sha256']
    with np.load(pair.FIT/'final_train_predictions.npz') as z:raw=z['contexts'];target=z['targets']
    cases={'raw':OUT/'predictions.npz','processed_reused':REF/'adapter_initial_only_predictions.npz'}
    result={};window_errors={}
    for name,path in cases.items():
        with np.load(path) as z:pred,truth,context=z['predictions'],z['targets'],z['contexts']
        assert np.array_equal(truth,target) and pred.shape==(3,16,16,8)
        if name=='raw':assert np.array_equal(context,raw)
        m,good,gate,_,_=summarize(pred,truth)
        recorded=r['raw_metrics' if name=='raw' else 'processed_reference_metrics']
        assert gate==recorded['pose_fit_passed'] and int(good.sum())==recorded['windows_all_three_seeds_pass']
        assert abs(m['path_position_cm']-recorded['position_cm'])<1e-4
        result[name]=dict(**m,pose_fit_passed=gate,passing_windows=int(good.sum()))
        position,_,_=errors(pred,truth);window_errors[name]=position.mean((0,2))
    a,b=result['raw']['path_position_cm'],result['processed_reused']['path_position_cm']
    verdict=dict(verified=True,one_new_arm_only=True,original_processed_arm_reexecuted=False,
        initial_decoder_and_all1024_noised_inputs_match=True,groups=result,
        processed_minus_raw_position_cm=b-a,processed_position_relative_change=float(b/a-1),
        windows_better_with_processing=int((window_errors['processed_reused']<window_errors['raw']).sum()),
        development_used=False,test_used=False,physical_rollout=False)
    write(OUT/'independent_audit.json',verdict)
    fig,ax=plt.subplots(figsize=(8,4),constrained_layout=True)
    values=[result[n]['path_position_cm']*10 for n in cases]
    bars=ax.bar(['Raw frozen Adapter','Processed frozen Adapter (reused)'],values)
    for bar,name in zip(bars,cases):ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+max(values)*.025,f"{result[name]['passing_windows']}/16 windows",ha='center')
    ax.set(ylabel='Train initial-approach path error (mm)',title='Only condition centering / scalar RMS processing differs');ax.set_ylim(0,max(values)*1.2);ax.grid(axis='y',alpha=.2)
    fig.savefig(OUT/'condition_processing_pair.png',dpi=170);plt.close(fig)
    print(json.dumps(verdict,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--analyze',action='store_true')
    if parser.parse_args().analyze:analyze()
    else:fit()

"""Replay original train draws, recording raw x0 without modifying sampling."""
import json
from pathlib import Path
import time
import numpy as np
import run_panda_spatial_fixed_trial_20261015 as trial
from probe_panda_initial_approach_20261015 import metrics
import torch

OUT=trial.ROOT/'reports/panda_initial_approach_probe_20261015'


@torch.inference_mode()
def main():
    assert not (OUT/'trace_report.json').exists()
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True,warn_only=True)
    model,_,plan,checkpoint=trial.load_final();started=time.monotonic()
    before={k:trial.inputs.module_sha(model,p) for k,p in trial.GROUPS.items()}
    with np.load(trial.OUT/'offline_train.npz') as z:
        contexts=torch.from_numpy(z['contexts']).cuda();targets=z['targets'];old=z['predictions']
    _,_,current,rows=trial.inputs.examples('train');current=current.cuda()
    selected=np.array([j for i in range(0,128,8) for j in [i,i+1]])
    selection=[rows[i] for i in selected]
    manifest=dict(checkpoint_sha256=checkpoint,source_sha256=plan['source_sha256'],
        trace_sha256=trial.sha(Path(__file__)),indices=selected.tolist(),selection=selection,
        sampling_seeds=plan['sample_seeds'],recorded_raw_x0_steps=[99,49,0],
        method='Unchanged native sample with read-only forward hook. Skip unselected batches by consuming exactly100 original-shaped Gaussian draws.',
        training_updates=0,development_used=False,test_used=False)
    trial.write(OUT/'trace_plan.json',manifest)
    arrays={};summary=[];max_replay=0.;max_replay_xyz=0.
    for si,draw in enumerate(plan['sample_seeds']):
        trial.seed(draw);collected={t:[] for t in [99,49,0]};final=[]
        def hook(module,args,out):
            t=int(args[1][0])
            if t in collected:collected[t].append(out.detach().cpu().numpy().copy())
        handle=model.diffusion_decoder.register_forward_hook(hook)
        try:
            for start in range(0,128,2):
                if start%8:
                    for _ in range(100):torch.randn((2,16,7),device='cuda')
                else:
                    value=model.sample(contexts[start:start+2],current[start:start+2]).cpu().numpy()
                    final.append(value)
        finally:handle.remove()
        final=np.concatenate(final);assert final.shape==(32,16,8)
        assert all(len(v)==16 for v in collected.values())
        delta=np.abs(final-old[si,selected]);max_replay=max(max_replay,float(delta.max()))
        max_replay_xyz=max(max_replay_xyz,float(delta[...,:3].max())*10)
        assert delta.max()<1e-4,'Do not interpret unmatched random streams'
        arrays[f'{draw}_native']=final
        for t,values in collected.items():arrays[f'{draw}_t{t}']=np.concatenate(values)
        for start in [0,12]:
            choose=np.array([r['start']==start for r in selection]);truth=targets[selected][choose]
            for name in ['t99','t49','t0','native']:
                pred=arrays[f'{draw}_{name}'][choose]
                summary.append(dict(seed=draw,observation_start=start,stage=name,**metrics(pred,truth)))
        print('MATCHED_INITIAL_TRACE',draw,'max_replay',float(delta.max()),flush=True)
    after={k:trial.inputs.module_sha(model,p) for k,p in trial.GROUPS.items()};assert before==after
    arrays['targets']=targets[selected];arrays['indices']=selected
    np.savez_compressed(OUT/'native_trace.npz',**arrays)
    means=[]
    for start in [0,12]:
        for stage in ['t99','t49','t0','native']:
            group=[r for r in summary if r['observation_start']==start and r['stage']==stage]
            means.append(dict(observation_start=start,stage=stage,**{k:float(np.mean([r[k] for r in group])) for k in ['position_cm','rotation_deg','endpoint_cm']}))
    trial.guard(plan)
    report=dict(complete=True,parameters_unchanged=before==after,training_updates=0,development_used=False,test_used=False,
        checkpoint_sha256=checkpoint,max_replay_component_difference=max_replay,max_replay_xyz_axis_difference_cm=max_replay_xyz,
        means=means,per_seed=summary,output_sha256=trial.sha(OUT/'native_trace.npz'),elapsed_seconds=time.monotonic()-started)
    trial.write(OUT/'trace_report.json',report);print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()

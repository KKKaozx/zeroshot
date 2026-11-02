"""Frozen, train-only check of the failed initial approach; no optimization."""
import argparse
import json
from pathlib import Path
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'reports/panda_spatial_fixed_trial_20261015'
DATA=ROOT/'reports/panda_spatial_control_v2_preparation_20261015'
OUT=ROOT/'reports/panda_initial_approach_probe_20261015'


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')


def metrics(pred,target):
    xyz=np.linalg.norm((pred[...,:3].astype(float)-target[...,:3])*10,axis=-1)
    a=pred[...,3:7].astype(float);b=target[...,3:7].astype(float)
    assert np.linalg.norm(a,axis=-1).min()>1e-8
    a/=np.linalg.norm(a,axis=-1,keepdims=True);b/=np.linalg.norm(b,axis=-1,keepdims=True)
    angle=np.degrees(2*np.arccos(np.clip(np.abs((a*b).sum(-1)),0,1)))
    return dict(position_cm=float(xyz.mean()),rotation_deg=float(angle.mean()),endpoint_cm=float(xyz[...,-1].mean()))


def probe():
    import run_panda_spatial_fixed_trial_20261015 as trial
    import torch
    from audit_bridge_end_to_end_20261015 import qmatrix
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True,warn_only=True)
    OUT.mkdir(exist_ok=False);started=time.monotonic()
    model,tokenizer,plan,checkpoint=trial.load_final()
    before={k:trial.inputs.module_sha(model,p) for k,p in trial.GROUPS.items()}
    images,target,current,rows=trial.inputs.examples('train')
    indices=np.array([i for start in [0,12] for i,r in enumerate(rows) if r['start']==start])
    assert len(indices)==32 and len({rows[i]['episode'] for i in indices})==16
    selected=[rows[i] for i in indices];truth=target[indices,:,:7].numpy()
    swap=np.r_[np.roll(np.arange(16),8),16+np.roll(np.arange(16),8)]
    receipt=dict(stage='frozen_train_initial_approach',checkpoint_sha256=checkpoint,
        source_sha256=plan['source_sha256'],probe_sha256=trial.sha(Path(__file__)),
        selection=selected,swap_indices=swap.tolist(),training_updates=0,development_used=False,test_used=False,
        levels=['t0','t49','t99','pure_t99'],noise_seeds=[1101,1102,1103],
        swap='Fixed cyclic8 shift within same phase; same noisy action and timestep; diagnostic, not a deployable intervention')
    write(OUT/'plan.json',receipt)
    # Audit target XYZ directly against future measured world states, without the production decoder.
    max_restore=0.;tracking=[];initial=[]
    for row,y in zip(selected,truth):
        with np.load(DATA/'episodes'/(row['episode']+'.npz')) as z:
            t=row['start'];world=z['positions'][t]+(qmatrix(z['rotations'][t]) @ (y[:,:3].astype(float)*.1).T).T
            max_restore=max(max_restore,float(np.abs(world-z['positions'][t+1:t+17]).max()))
            tracking.append(float(np.linalg.norm(z['positions'][t+1:t+17]-z['commanded_poses'][t:t+16,:3],axis=-1).mean()*100))
            if t==0:initial.append(z['images'][0])
    assert max_restore<2e-7 and np.abs(truth[...,:3]).max()<3
    text=tokenizer([read(DATA/'protocol.json')['instruction']]*128,padding=True,truncation=True,max_length=77,return_tensors='pt')
    # Recreate the cached-token training route and online route for exactly the same observations.
    with torch.inference_mode():
        visual=torch.cat([model.vision_encoder(pixel_values=images[i:i+2].cuda()).last_hidden_state for i in range(0,128,2)])
        language=torch.cat([model.text_encoder(**{k:v[i:i+16].cuda() for k,v in text.items()}).last_hidden_state for i in range(0,128,16)])
        cachectx=torch.cat([model.adapter(visual[i:i+2],language[i:i+2],text['attention_mask'][i:i+2].cuda()) for i in range(0,128,2)])
        online=torch.cat([model.get_context_vector(images[i:i+2].cuda(),text['input_ids'][i:i+2].cuda(),text['attention_mask'][i:i+2].cuda()) for i in range(0,128,2)])
        with np.load(RUN/'offline_train.npz') as z:
            savedctx=z['contexts'];savedpred=z['predictions'];savedtarget=z['targets']
        assert np.array_equal(target.numpy(),savedtarget)
        cached_online_delta=float((cachectx-online).abs().max())
        saved_delta=float(np.abs(online.cpu().numpy()-savedctx).max())
        assert cached_online_delta<1e-4 and saved_delta<1e-4
        # First original batch, same seed and order: verify unchanged native sampling.
        trial.seed(1101);replay=model.sample(online[:2],current[:2].cuda()).cpu().numpy()
        replay_delta=float(np.abs(replay-savedpred[0,:2]).max());assert replay_delta<1e-4
        ctx=online[indices];y=torch.from_numpy(truth).cuda();outputs={};summary=[]
        for draw in receipt['noise_seeds']:
            trial.seed(draw);noise=torch.randn_like(y)
            for level,t in [('t0',0),('t49',49),('t99',99),('pure_t99',99)]:
                ab=model.alpha_bars[t]
                noisy=noise if level=='pure_t99' else ab.sqrt()*y+(1-ab).sqrt()*noise
                timesteps=torch.full((32,),t,device='cuda',dtype=torch.long)
                for name,condition in [('correct',ctx),('image_swap',ctx[swap])]:
                    out=torch.cat([model.diffusion_decoder(noisy[i:i+8],timesteps[i:i+8],condition[i:i+8]) for i in range(0,32,8)]).cpu().numpy()
                    outputs[f'{draw}_{level}_{name}']=out
                for start,lo,hi in [(0,0,16),(12,16,32)]:
                    correct=outputs[f'{draw}_{level}_correct'][lo:hi];changed=outputs[f'{draw}_{level}_image_swap'][lo:hi]
                    summary.append(dict(seed=draw,level=level,observation_start=start,
                        correct=metrics(correct,truth[lo:hi]),image_swap=metrics(changed,truth[lo:hi]),
                        paired_output_change=metrics(changed,correct)))
                print('INITIAL_PROBE',draw,level,'initial_cm',round(summary[-2]['correct']['position_cm'],4),
                    'aligned_cm',round(summary[-1]['correct']['position_cm'],4),flush=True)
        visual_summary=torch.cat([visual[indices,0],visual[indices,1:].mean(1)],dim=1).cpu().numpy()
    after={k:trial.inputs.module_sha(model,p) for k,p in trial.GROUPS.items()};assert before==after
    trial.guard(plan)
    outputs.update(targets=truth,indices=indices,contexts=ctx.cpu().numpy(),visual_cls_patch_mean=visual_summary,
        saved_native_predictions=savedpred[:,indices],initial_images=np.stack(initial))
    np.savez_compressed(OUT/'outputs.npz',**outputs)
    phase=[]
    for start in [0,12]:
        for level in receipt['levels']:
            rows_level=[r for r in summary if r['observation_start']==start and r['level']==level]
            phase.append(dict(observation_start=start,level=level,**{name:{k:float(np.mean([r[name][k] for r in rows_level])) for k in rows_level[0][name]}
                for name in ['correct','image_swap','paired_output_change']}))
    result=dict(complete=True,verified=True,checkpoint_sha256=checkpoint,parameters_unchanged=before==after,
        source_unchanged=True,training_updates=0,development_used=False,test_used=False,
        max_label_restore_difference_m=max_restore,max_xyz_normalized=float(np.abs(truth[...,:3]).max()),
        mean_expert_tracking_error_cm=float(np.mean(tracking)),cached_online_context_max_difference=cached_online_delta,
        previous_saved_context_max_difference=saved_delta,native_first_batch_replay_max_difference=replay_delta,
        alpha_bar={str(t):float(model.alpha_bars[t]) for t in [0,49,99]},summary=phase,per_seed=summary,
        elapsed_seconds=time.monotonic()-started,output_sha256=trial.sha(OUT/'outputs.npz'))
    write(OUT/'report.json',result)
    print('FROZEN INITIAL APPROACH COMPLETE',json.dumps(result,indent=2),flush=True)


def render():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image,ImageDraw
    from audit_bridge_end_to_end_20261015 import qmatrix
    r=read(OUT/'report.json');p=read(OUT/'plan.json');assert r['complete'] and r['verified']
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for ax,start in zip(axes,[0,12]):
        group=[row for row in r['summary'] if row['observation_start']==start];x=np.arange(4)
        for offset,name,label in [(-.18,'correct','Correct image'),(.18,'image_swap','Swapped same-phase image')]:
            ax.bar(x+offset,[row[name]['position_cm'] for row in group],width=.36,label=label)
        ax.set_xticks(x,[row['level'] for row in group]);ax.set(title=f'Training observation start {start}',ylabel='Direct clean-action position error (cm)')
        ax.legend();ax.grid(axis='y',alpha=.2)
    fig.savefig(OUT/'noise_and_image.png',dpi=170);plt.close(fig)
    with np.load(OUT/'outputs.npz') as z:
        truth=z['targets'];pred=z['saved_native_predictions'];images=z['initial_images']
    fig,ax=plt.subplots(figsize=(6,6),constrained_layout=True)
    for i,row in enumerate(p['selection'][:16]):
        with np.load(DATA/'episodes'/(row['episode']+'.npz')) as z:
            ref=z['positions'][0];R=qmatrix(z['rotations'][0])
        goal=ref+R @ (truth[i,11,:3]*.1)
        estimate=ref+R @ (pred[:,i,11,:3].mean(0)*.1)
        ax.plot([goal[0],estimate[0]],[goal[1],estimate[1]],color='#bbbbbb',linewidth=1)
        ax.scatter(*goal[:2],marker='o',color='#2475ae');ax.scatter(*estimate[:2],marker='x',color='#c55236')
    ax.scatter([],[],marker='o',label='True future step12',color='#2475ae')
    ax.scatter([],[],marker='x',label='Native DDPM mean of3 draws',color='#c55236')
    ax.set(xlabel='World X (m)',ylabel='World Y (m)',title='Initial observations: training XY approach predictions');ax.set_aspect('equal');ax.legend();ax.grid(alpha=.2)
    fig.savefig(OUT/'initial_training_xy.png',dpi=170);plt.close(fig)
    sheet=Image.new('RGB',(896,996),'white');draw=ImageDraw.Draw(sheet)
    for i,im in enumerate(images):
        x,y=i%4*224,i//4*249;sheet.paste(Image.fromarray(im),(x,y));draw.text((x+5,y+229),p['selection'][i]['episode'],fill='black')
    sheet.save(OUT/'all_initial_training_images.png')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--render',action='store_true')
    if parser.parse_args().render:render()
    else:probe()

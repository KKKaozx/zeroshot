"""One frozen-budget Panda trial: fresh joint fit, reload, offline and rollout.

The recipe and split were fixed before this entry was written. No model selection,
expert handoff, object-coordinate features, test rollout or extra training arms.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.update(USE_TF='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'reports/panda_spatial_control_v2_preparation_20261015'
OUT=ROOT/'reports/panda_spatial_fixed_trial_20261015'
sys.path.insert(0,str(ROOT/'code'))
import numpy as np
import torch
from transformers import CLIPTokenizer
import preflight_panda_minimal_generalization_20261015 as inputs
from models import RobotAdapterModel
from dataset import prepare_image,decode_relative_pose
from train import policy_loss
from replay_and_fit_bridge_positive import seed,evaluate,score
from fit_bridge_adapter_pair import gripper_metrics

inputs.DATA=DATA
GROUPS=dict(clip=('vision_encoder.','text_encoder.'),adapter=('adapter.',),decoder=('diffusion_decoder.',),
            gripper=('gripper_','current_gripper_projection.'))


def sha(p):return inputs.sha(p)
def read(p):return inputs.read(p)
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')


def sources():
    files=[ROOT/'code'/n for n in ['models.py','adapter.py','diffusion_decoder.py','dataset.py','train.py',
        'replay_and_fit_bridge_positive.py','fit_bridge_adapter_pair.py','evaluate_bridge_endpoint_metrics.py','pybullet_fixed_grasp.py']]
    files += [Path(__file__),ROOT/'scripts/preflight_panda_minimal_generalization_20261015.py']
    return {p.relative_to(ROOT).as_posix():sha(p) for p in files}


def guard(plan):
    assert sha(DATA/'protocol.json')==plan['protocol_sha256']
    assert all(sha(ROOT/n)==h for n,h in plan['source_sha256'].items())
    assert not any((DATA/'episodes'/(r['episode']+'.npz')).exists() for r in read(DATA/'protocol.json')['episodes'] if r['split']=='test')


def config():
    cfg=copy.deepcopy(read(ROOT/'reports/panda_spatial_control_v2_gpu_preflight_fixed_20261015/plan.json')['config'])
    assert cfg['model']['num_adapter_layers']==8 and cfg['model']['num_diffusion_steps']==100
    assert cfg['model']['decoder_type']=='diffusion' and cfg['model']['diffusion_prediction_type']=='sample'
    assert cfg['model']['separate_gripper_head'] and Path(cfg['model']['name']).is_dir()
    return cfg


def fit():
    OUT.mkdir(exist_ok=False)
    protocol=read(DATA/'protocol.json');preflight=read(ROOT/'reports/panda_spatial_control_v2_gpu_preflight_fixed_20261015/report.json')
    assert preflight['passed'] and read(DATA/'verification.json')['verified']
    assert preflight['collection_report_sha256']==sha(DATA/'collection.json')
    assert protocol['training_recipe']['updates']==8192 and protocol['training_recipe']['batch_size']==8
    for n,h in preflight['source_sha256'].items():assert sha(ROOT/n if n.startswith('scripts/') else ROOT/'code'/n)==h
    plan=dict(protocol_sha256=sha(DATA/'protocol.json'),source_sha256=sources(),config=config(),
        initialization=43,updates=8192,batch_size=8,learning_rates=[dict(first=1,last=4096,lr=1e-4),dict(first=4097,last=8192,lr=1e-5)],
        sample_seeds=[1101,1102,1103],offline_batch_size=2,checkpoint_selection='Final fixed update only',
        inputs='RGB,instruction,measured finger width; no EE-state port or object coordinates',
        loss='Unchanged production policy_loss; pose and gripper jointly trained',
        rollout=dict(split='development',replans=28,executed_prefix=4,total_commands=112,
            position_min=[.25,-.3,.64],position_max=[.75,.3,1.05],
            success='Last10 object heights >initial+0.10m and final bilateral contacts; no precise-height condition',
            candidate='At least3/4 positions succeed in at least2/3 sampling draws'),
        storage=dict(stage='Formal training, separate from preparation-only no-weight rule',
            checkpoint_bytes_limit=1024**3,one_final_nonclip_checkpoint=True,save_optimizer=False,intermediate_weights=False),
        preflight_weight_loaded=False,test_used=False)
    write(OUT/'plan.json',plan);(OUT/'runner_snapshot.py').write_bytes(Path(__file__).read_bytes());guard(plan)
    images,targets,current,rows=inputs.examples('train')
    assert len(rows)==128 and len({r['episode'] for r in rows})==16
    batches=torch.cat([torch.randperm(128,generator=torch.Generator().manual_seed(440001+c)).reshape(16,8) for c in range(512)])
    assert batches.shape==(8192,8) and torch.equal(torch.bincount(batches.flatten()),torch.full((128,),512))
    seed(43);model=RobotAdapterModel(plan['config']).cuda().eval()
    tokenizer=CLIPTokenizer.from_pretrained(plan['config']['model']['name'],local_files_only=True)
    texts=tokenizer([protocol['instruction']]*128,padding=True,truncation=True,max_length=77,return_tensors='pt')
    texts={k:v.cuda() for k,v in texts.items()}
    with torch.no_grad():
        visual=torch.cat([model.vision_encoder(pixel_values=images[i:i+2].cuda()).last_hidden_state for i in range(0,128,2)]).detach()
        language=torch.cat([model.text_encoder(**{k:v[i:i+16] for k,v in texts.items()}).last_hidden_state for i in range(0,128,16)]).detach()
    model.vision_encoder.cpu();model.text_encoder.cpu();torch.cuda.empty_cache()
    before={k:inputs.module_sha(model,p) for k,p in GROUPS.items()}
    trainable=[p for n,p in model.named_parameters() if not n.startswith(GROUPS['clip'])]
    assert all(not p.requires_grad for p in model.vision_encoder.parameters())
    optimizer=torch.optim.Adam(trainable,lr=1e-4)
    targets=targets.cuda();current=current.cuda();started=time.monotonic();recent=[]
    torch.cuda.reset_peak_memory_stats()
    report=dict(stage='formal_fixed_budget_fit',complete=False,updates_completed=0,trained=True,
        source_sha256=plan['source_sha256'],protocol_sha256=plan['protocol_sha256'],
        train_windows=128,train_episodes=16,development_used_for_optimization=False,test_used=False,
        fresh_initialization=43,preflight_weight_loaded=False,gpu=torch.cuda.get_device_name(),torch_version=torch.__version__,
        batches_sha256=hashlib.sha256(batches.numpy().tobytes()).hexdigest(),initial_group_sha256=before,
        parameters=sum(p.numel() for p in model.parameters()),trainable_parameters=sum(p.numel() for p in trainable))
    write(OUT/'training_report.json',report)
    with (OUT/'updates.jsonl').open('w',encoding='utf-8') as log:
        for step,ix in enumerate(batches,1):
            lr=1e-4 if step<=4096 else 1e-5
            for group in optimizer.param_groups:group['lr']=lr
            ix=ix.cuda();model.train();seed(430000+step)
            context=model.adapter(visual[ix],language[ix],texts['attention_mask'][ix])
            seed(420000+step);output=model.diffusion_loss(targets[ix],context,current[ix])
            loss,pose,grip=policy_loss(model,output,targets[ix],torch.nn.MSELoss(),current_grippers=current[ix])
            assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True);loss.backward()
            if step==1:
                report['first_step_gradient']={k:any(p.grad is not None and bool(torch.count_nonzero(p.grad)) for n,p in model.named_parameters() if n.startswith(prefix)) for k,prefix in GROUPS.items()}
                assert report['first_step_gradient']==dict(clip=False,adapter=True,decoder=True,gripper=True)
            norm=torch.nn.utils.clip_grad_norm_(trainable,1.,error_if_nonfinite=True);optimizer.step()
            row=dict(update=step,loss=float(loss.detach()),pose=float(pose.detach()),gripper=float(grip.detach()),lr=lr,gradient_norm=float(norm))
            log.write(json.dumps(row)+'\n');recent.append(row['loss'])
            if step==1 or step%256==0:
                log.flush();elapsed=time.monotonic()-started
                report.update(updates_completed=step,elapsed_seconds=elapsed,last_loss=row['loss'],mean_recent_loss=float(np.mean(recent)))
                write(OUT/'training_report.json',report)
                print('PANDA_FIXED_UPDATE',step,'/8192','loss',round(row['loss'],7),'mean',round(report['mean_recent_loss'],7),'seconds',round(elapsed,1),flush=True);recent=[]
    after={k:inputs.module_sha(model,p) for k,p in GROUPS.items()}
    changed={k:before[k]!=after[k] for k in GROUPS};assert changed==dict(clip=False,adapter=True,decoder=True,gripper=True)
    guard(plan)
    state={n:v.detach().cpu().clone() for n,v in model.state_dict().items() if not n.startswith(GROUPS['clip'])}
    checkpoint=OUT/'final.pt'
    torch.save(dict(config=plan['config'],state_dict_nonclip=state,protocol_sha256=plan['protocol_sha256'],
        execution_plan_sha256=sha(OUT/'plan.json'),update=8192,source_sha256=plan['source_sha256'],
        final_group_sha256=after,initialization=43),checkpoint)
    assert checkpoint.stat().st_size<plan['storage']['checkpoint_bytes_limit']
    report.update(complete=True,updates_completed=8192,changed=changed,final_group_sha256=after,
        checkpoint_sha256=sha(checkpoint),checkpoint_bytes=checkpoint.stat().st_size,
        actual_draws_per_window=torch.bincount(batches.flatten(),minlength=128).tolist(),
        peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,elapsed_seconds=time.monotonic()-started)
    write(OUT/'training_report.json',report);print('PANDA FIXED FIT COMPLETE',checkpoint,flush=True)


def load_final():
    plan=read(OUT/'plan.json');guard(plan);r=read(OUT/'training_report.json')
    assert r['complete'] and r['updates_completed']==8192 and sha(OUT/'final.pt')==r['checkpoint_sha256']
    payload=torch.load(OUT/'final.pt',map_location='cpu',weights_only=False)
    assert payload['execution_plan_sha256']==sha(OUT/'plan.json') and payload['update']==8192
    seed(43);model=RobotAdapterModel(payload['config']).cuda().eval()
    missing=model.load_state_dict(payload['state_dict_nonclip'],strict=False)
    assert not missing.unexpected_keys and all(n.startswith(GROUPS['clip']) for n in missing.missing_keys)
    assert {k:inputs.module_sha(model,p) for k,p in GROUPS.items()}==payload['final_group_sha256']
    tokenizer=CLIPTokenizer.from_pretrained(payload['config']['model']['name'],local_files_only=True)
    return model,tokenizer,plan,r['checkpoint_sha256']


@torch.inference_mode()
def offline():
    assert not (OUT/'offline_report.json').exists()
    model,tokenizer,plan,checkpoint=load_final();groups={}
    for split in ['train','development']:
        images,target,current,rows=inputs.examples(split)
        texts=tokenizer([read(DATA/'protocol.json')['instruction']]*len(rows),padding=True,truncation=True,max_length=77,return_tensors='pt')
        ctx=torch.cat([model.get_context_vector(images[i:i+2].cuda(),texts['input_ids'][i:i+2].cuda(),texts['attention_mask'][i:i+2].cuda()) for i in range(0,len(rows),2)])
        predictions,metrics=evaluate(model,ctx,current.cuda(),target.numpy(),plan['sample_seeds'],batch=2)
        metrics['gripper']=gripper_metrics(predictions,target.numpy());metrics['episodes']=len({r['episode'] for r in rows})
        groups[split]=metrics
        np.savez_compressed(OUT/f'offline_{split}.npz',predictions=predictions,targets=target.numpy(),contexts=ctx.cpu().numpy(),row_indices=np.arange(len(rows)))
        print('PANDA OFFLINE',split,json.dumps(metrics),flush=True)
    guard(plan);write(OUT/'offline_report.json',dict(stage='offline_final_checkpoint',complete=True,
        checkpoint_sha256=checkpoint,sampling_seeds=plan['sample_seeds'],groups=groups,test_used=False))


@torch.inference_mode()
def rollout():
    assert not (OUT/'rollout_report.json').exists()
    model,tokenizer,plan,checkpoint=load_final()
    import pybullet_fixed_grasp as sim
    from PIL import Image
    protocol=read(DATA/'protocol.json');text=tokenizer(protocol['instruction'],return_tensors='pt')
    text={k:v.cuda() for k,v in text.items()};folder=OUT/'rollouts';folder.mkdir(exist_ok=False)
    report=dict(stage='learned_policy_held_out_xy_rollout',complete=False,checkpoint_sha256=checkpoint,
        source_sha256=plan['source_sha256'],test_used=False,expert_handoff=False,object_pose_given_to_model=False,
        sampling_seeds=plan['sample_seeds'],results=[],started_at=time.time())
    sim.p.connect(sim.p.DIRECT)
    try:
        for row in protocol['episodes']:
            if row['split']!='development':continue
            for draw in plan['sample_seeds']:
                seed(draw);robot,cube=sim.scene()
                # Initial-condition setup only; no object reset after first policy image.
                sim.p.resetBasePositionAndOrientation(cube,[*row['cube_xy_m'],.66],[0,0,0,1])
                sim.p.resetBaseVelocity(cube,[0,0,0],[0,0,0]);sim.p.setTimeStep(1/240)
                sim.command(robot,[.5,0,.85],1,ticks=120)
                initial=sim.p.getBasePositionAndOrientation(cube)[0][2]
                frames=[];actions=[];ee=[];widths=[];heights=[];contacts=[];requested=[];unclipped=[];reference=[]
                started=time.monotonic()
                for replan in range(28):
                    image=sim.rgb();frames.append(image);refpos,refquat,current=sim.observation(robot)
                    reference.append(np.r_[refpos,refquat,current])
                    context=model.get_context_vector(prepare_image(image).unsqueeze(0).cuda(),text['input_ids'],text['attention_mask'])
                    chunk=model.sample(context,torch.tensor([[current]],dtype=torch.float32,device='cuda'))[0].cpu().numpy()
                    assert chunk.shape==(16,8) and np.isfinite(chunk).all()
                    for action in chunk[:4]:
                        p,q=decode_relative_pose(refpos,refquat,action)
                        unclipped.append(p.copy());p=np.clip(p,plan['rollout']['position_min'],plan['rollout']['position_max'])
                        sim.command(robot,p.tolist(),float(action[7]),q.tolist())
                        pos,quat,width=sim.observation(robot);ee.append(pos);widths.append(width);actions.append(action)
                        requested.append(np.r_[p,q,action[7]])
                        heights.append(float(sim.p.getBasePositionAndOrientation(cube)[0][2]))
                        contacts.append(sorted({c[3] for c in sim.p.getContactPoints(robot,cube) if c[3] in (9,10) and c[9]>.01}))
                assert sim.p.getNumConstraints()==0 and len(actions)==112
                result=sim.outcome(robot,cube,float(initial),heights,contacts)
                result.update(episode=row['episode'],split='development',sampling_seed=draw,commands=112,
                    learned_policy_used=True,expert_handoff=False,object_pose_given_to_model=False,
                    workspace_clipped_commands=int((np.abs(np.asarray(unclipped)-np.asarray(requested)[:,:3])>1e-12).any(1).sum()),
                    elapsed_seconds=time.monotonic()-started)
                prefix=f"{row['episode']}_seed{draw}";frames.append(sim.rgb())
                np.savez_compressed(folder/(prefix+'.npz'),images=np.stack(frames),actions=actions,ee_positions=ee,
                    measured_widths=widths,object_heights=heights,requested_poses=requested,unclipped_xyz=unclipped,
                    replan_reference=reference,finger_contacts=[[j in c for j in (9,10)] for c in contacts])
                views=[Image.fromarray(x) for x in frames]
                views[0].save(folder/(prefix+'.gif'),save_all=True,append_images=views[1:],duration=400,loop=0)
                views[-1].save(folder/(prefix+'_final.png'))
                result['trajectory_sha256']=sha(folder/(prefix+'.npz'));report['results'].append(result)
                write(OUT/'rollout_partial.json',report)
                print('PANDA LEARNED ROLLOUT',json.dumps(result),flush=True)
                assert sum(p.stat().st_size for p in folder.iterdir())<200*1024**2
    finally:sim.p.disconnect()
    assert len(report['results'])==12
    counts={r['episode']:sum(x['success'] for x in report['results'] if x['episode']==r['episode']) for r in protocol['episodes'] if r['split']=='development'}
    report.update(complete=True,successes=sum(x['success'] for x in report['results']),episodes=12,
        per_position_successes=counts,positions_with_at_least_two_successes=sum(n>=2 for n in counts.values()),
        development_screen_passed=sum(n>=2 for n in counts.values())>=3,elapsed_seconds=time.time()-report['started_at'])
    guard(plan);write(OUT/'rollout_report.json',report)
    print('PANDA FIXED TRIAL COMPLETE',json.dumps(counts),'screen',report['development_screen_passed'],flush=True)


if __name__=='__main__':
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('stage',choices=['fit','offline','rollout'])
    args=ap.parse_args()
    assert torch.cuda.is_available(),'Run this GPU entry directly, without importing CPU collection wrappers'
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True,warn_only=True)
    {'fit':fit,'offline':offline,'rollout':rollout}[args.stage]()

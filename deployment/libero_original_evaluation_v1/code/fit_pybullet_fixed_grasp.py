"""Train and reload the project's policy on one native simulator demonstration.

This is a seen-scene overfit positive control, not Bridge transfer/generalization.
Uses unchanged production CLIP, Adapter, diffusion_loss, policy_loss and sample.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.update(USE_TF='0', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
import numpy as np
import torch
from transformers import CLIPTokenizer
from dataset import relative_pose_action, decode_relative_pose, prepare_image
from models import RobotAdapterModel
from train import policy_loss
from fit_bridge_adapter_pair import file_sha, gripper_metrics
from replay_and_fit_bridge_positive import evaluate, digest, seed
import pybullet_fixed_grasp as sim

ROOT=Path(__file__).resolve().parents[1]
CONTRACT=dict(robot='pybullet_data/franka_panda/panda.urdf', end_effector_link=11,
              finger_joints=[9,10], ticks_per_action=sim.TICKS, simulation_hz=240,
              current_gripper='2*(sum(measured finger joint positions)/0.08)-1',
              command_gripper='negative=closed motor target 0; positive=open target 0.04',
              pose='All 16 future measured EE arrival poses relative to observation EE pose',
              scene='Fixed cube/table/Panda scene in pybullet_fixed_grasp.scene',
              success='Cube >=10cm above initial height for last 10 actions and final two-finger contact',
              artificial_attachment=False, generalization_test=False)


def trainable_hash(model):
    h=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        if not name.startswith(('vision_encoder.','text_encoder.')):
            h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def train(args):
    args.output.mkdir(parents=True,exist_ok=False)
    data=np.load(args.demonstration)
    physics=json.loads((args.demonstration.parent/'physics_report.json').read_text())
    if not physics['success'] or physics['artificial_attachment']:
        raise ValueError('Training requires verified physical collection')
    starts=args.starts or [0,12,24,30,38,46,58,74]
    if not starts or any(i<0 or i+16>=len(data['images']) for i in starts):
        raise ValueError('Every selected observation needs 16 actually recorded future states')
    images=torch.stack([prepare_image(data['images'][i]) for i in starts])
    targets=torch.tensor(np.stack([[relative_pose_action(data['positions'][i],data['rotations'][i],
        data['positions'][j],data['rotations'][j],data['commands'][j])
        for j in range(i+1,i+17)] for i in starts]))
    currents=torch.tensor(data['currents'][starts],dtype=torch.float32).reshape(-1,1)
    config=json.loads((ROOT/'reports/bridge_joint_both_events_1000_local_20261011/plan.json').read_text())['config']
    config['model']['current_gripper_encoding']='panda_measured_finger_width_affine_v1'
    np.savez_compressed(args.output/'training_examples.npz',images=data['images'][starts],
                        targets=targets.numpy(),currents=currents.numpy(),starts=starts)
    seed(42)
    model=RobotAdapterModel(config).cuda().eval()
    initial_hash=trainable_hash(model)
    source_hashes={n:file_sha(ROOT/'code'/n) for n in
        ['fit_pybullet_fixed_grasp.py','pybullet_fixed_grasp.py','models.py','adapter.py','diffusion_decoder.py','dataset.py','train.py']}
    (args.output/'runner_snapshot.py').write_bytes(Path(__file__).read_bytes())
    tokenizer=CLIPTokenizer.from_pretrained(config['model']['name'],local_files_only=True)
    text={k:v.cuda() for k,v in tokenizer([sim.INSTRUCTION]*len(starts),padding=True,return_tensors='pt').items()}
    with torch.no_grad():
        visual=model.vision_encoder(pixel_values=images.cuda()).last_hidden_state
        language=model.text_encoder(**text).last_hidden_state
    model.vision_encoder.cpu();model.text_encoder.cpu()
    clip_before=[digest(model.vision_encoder),digest(model.text_encoder)]
    for name,param in model.named_parameters():
        param.requires_grad_(not name.startswith(('vision_encoder.','text_encoder.')))
    optimizer=torch.optim.Adam([x for x in model.parameters() if x.requires_grad],lr=1e-4)
    truth=targets.numpy();targets=targets.cuda();currents=currents.cuda()
    masks=torch.ones_like(targets);history=[];started=time.monotonic()
    for update in range(1,args.updates+1):
        model.train();seed(430000+update)
        ctx=model.adapter(visual,language,text['attention_mask'])
        seed(420000+update)
        output=model.diffusion_loss(targets,ctx,currents)
        loss,pose,grip=policy_loss(model,output,targets,torch.nn.MSELoss(),
                                  current_grippers=currents,supervision_masks=masks)
        optimizer.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_([x for x in model.parameters() if x.requires_grad],1.,error_if_nonfinite=True)
        optimizer.step()
        if update==1 or update%100==0:
            print('SIM_UPDATE',update,'loss',float(loss.detach()),'seconds',round(time.monotonic()-started,1),flush=True)
        if update in {500,args.updates}:
            model.eval()
            with torch.no_grad():
                ctx=model.adapter(visual,language,text['attention_mask'])
                pred,metrics=evaluate(model,ctx,currents,truth,[1101,1102,1103],batch=len(starts))
            metrics['gripper']=gripper_metrics(pred,truth)
            history.append(dict(update=update,**metrics))
            print('SIM_FIT',json.dumps(history[-1]),flush=True)
    assert clip_before==[digest(model.vision_encoder),digest(model.text_encoder)]
    state={k:v.detach().cpu() for k,v in model.state_dict().items() if not k.startswith(('vision_encoder.','text_encoder.'))}
    torch.save(dict(config=config,trainable_state_dict=state,contract=CONTRACT,
                    experiment_kind='pybullet_fixed_scene_training_positive_control'),args.output/'final.pt')
    report=dict(stage='fixed_scene_training_only',trained=True,simulation_policy_tested=False,
                updates=args.updates,windows=len(starts),batch_size=len(starts),starts=starts,
                initial_trainable_sha256=initial_hash,final_trainable_sha256=trainable_hash(model),
                source_sha256=source_hashes,demonstration_sha256=file_sha(args.demonstration),
                checkpoint_sha256=file_sha(args.output/'final.pt'),contract=CONTRACT,
                clip_unchanged=True,history=history,elapsed_seconds=time.monotonic()-started)
    assert all(file_sha(ROOT/'code'/n)==h for n,h in source_hashes.items())
    (args.output/'training_report.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(args.output/'offline_predictions.npz',predictions=pred,targets=truth)


@torch.inference_mode()
def rollout(args):
    payload=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    if payload.get('contract')!=CONTRACT: raise ValueError('Simulator/checkpoint contract differs')
    model=RobotAdapterModel(payload['config']).cuda().eval()
    incompatible=model.load_state_dict(payload['trainable_state_dict'],strict=False)
    assert not incompatible.unexpected_keys
    assert all(n.startswith(('vision_encoder.','text_encoder.')) for n in incompatible.missing_keys)
    tokenizer=CLIPTokenizer.from_pretrained(payload['config']['model']['name'],local_files_only=True)
    text={k:v.cuda() for k,v in tokenizer(sim.INSTRUCTION,return_tensors='pt').items()}
    args.output.mkdir(parents=True,exist_ok=False)
    results=[]
    source_hashes={n:file_sha(ROOT/'code'/n) for n in
        ['fit_pybullet_fixed_grasp.py','pybullet_fixed_grasp.py','models.py','adapter.py','diffusion_decoder.py','dataset.py']}
    (args.output/'runner_snapshot.py').write_bytes(Path(__file__).read_bytes())
    from PIL import Image
    sim.p.connect(sim.p.DIRECT)
    try:
        for sampling_seed in [1101,1102,1103]:
            seed(sampling_seed);robot,cube=sim.scene()
            initial=sim.p.getBasePositionAndOrientation(cube)[0][2]
            frames=[];heights=[];contacts=[];actions=[];ee=[];widths=[];target_positions=[]
            for replan in range(28):
                image=sim.rgb();frames.append(image)
                refpos,refquat,current=sim.observation(robot)
                context=model.get_context_vector(prepare_image(image).unsqueeze(0).cuda(),
                                                 text['input_ids'],text['attention_mask'])
                chunk=model.sample(context,torch.tensor([[current]],dtype=torch.float32,device='cuda'))[0].cpu().numpy()
                if not np.isfinite(chunk).all():raise ValueError('Nonfinite generated action')
                for action in chunk[:4]:
                    pos,quat=decode_relative_pose(refpos,refquat,action)
                    # Same physical workspace guard for every generated command.
                    pos=np.clip(pos,[.25,-.3,.64],[.75,.3,1.05])
                    sim.command(robot,pos.tolist(),float(action[7]),quat.tolist())
                    heights.append(float(sim.p.getBasePositionAndOrientation(cube)[0][2]))
                    contacts.append(sorted({c[3] for c in sim.p.getContactPoints(robot,cube)
                                            if c[3] in (9,10) and c[9]>.01}))
                    ep,eq,ew=sim.observation(robot);ee.append(ep);widths.append(ew);actions.append(action)
                    target_positions.append(pos)
            result=sim.outcome(robot,cube,initial,heights,contacts)
            result.update(sampling_seed=sampling_seed,controller='own_learned_policy',replans=28,
                          executed_actions=len(actions),expert_handoff=False,object_pose_given_to_model=False)
            if args.hold_height is not None:
                last=np.asarray(heights[-20:]);error=np.abs(last-args.hold_height)
                contact_count=sum(c==[9,10] for c in contacts[-20:])
                result['hold']=dict(target_object_height_m=args.hold_height,
                    last_two_seconds_max_height_error_cm=float(error.max()*100),
                    last_two_seconds_height_range_cm=float(np.ptp(last)*100),
                    last_two_seconds_two_finger_contacts=contact_count,
                    passed=bool(error.max()<=.02 and np.ptp(last)<=.01 and contact_count==20))
            results.append(result)
            frames.append(sim.rgb())
            Image.fromarray(frames[-1]).save(args.output/f'policy_seed_{sampling_seed}_final.png')
            views=[Image.fromarray(x) for x in frames]
            views[0].save(args.output/f'policy_seed_{sampling_seed}.gif',save_all=True,
                          append_images=views[1:],duration=160,loop=0)
            np.savez_compressed(args.output/f'policy_seed_{sampling_seed}.npz',images=frames,
                                actions=actions,ee_positions=ee,measured_widths=widths,object_heights=heights,
                                target_positions=target_positions,
                                finger_contacts=[[j in row for j in (9,10)] for row in contacts])
            print('MODEL_PHYSICS',json.dumps(result),flush=True)
    finally: sim.p.disconnect()
    report=dict(stage='own_model_closed_loop_fixed_training_scene',contract=CONTRACT,
                checkpoint_sha256=file_sha(args.checkpoint),successes=sum(x['success'] for x in results),
                episodes=len(results),results=results,generalization_test=False,source_sha256=source_hashes,
                success_rate_is_training_scene_only=True)
    assert all(file_sha(ROOT/'code'/n)==h for n,h in source_hashes.items())
    (args.output/'rollout_report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage',choices=['train','rollout'])
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--demonstration',type=Path)
    ap.add_argument('--checkpoint',type=Path)
    ap.add_argument('--updates',type=int,default=1000)
    ap.add_argument('--starts',type=int,nargs='+',help='Explicit window observations; duplicates represent explicit sampling weight')
    ap.add_argument('--hold-height',type=float,help='Evaluation-only object height; never provided to model/controller')
    args=ap.parse_args()
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True,warn_only=True)
    if args.stage=='train': train(args)
    else: rollout(args)

"""Matched current-pose input diagnostic: zero vector versus real input state.

Frozen original CLIP/Adapter contexts; same pose/gripper heads and an identically
zero-initialized state projection. State goes only into the pose head. Gripper
keeps original context, current opening and teacher-pose training convention.
No future state, episode ID, or time index is passed as an input.
"""
import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('HF_HUB_OFFLINE','1')
import numpy as np
import pybullet as bullet
import torch
import torch.nn.functional as F

from compare_bridge_adapter_training import digest_state, initial_metrics_match
from diagnose_bridge_modules import reconstruct, save_json, metrics, matrices, group_report
from models import RobotAdapterModel
from train import collate_batch, policy_loss, set_seed


def raw_pose_inputs(ds,indices):
    raw={}; values=[]
    for i in indices:
        s=ds.samples[i]; key=(s['file_path'],s['record_index'])
        if key not in raw:
            f=ds._load_tfrecord_example(*key).features.feature
            raw[key]=np.asarray(f['steps/observation/state'].float_list.value,dtype=np.float64).reshape(-1,7)
        current=raw[key][s['start_index']]
        rotation=matrices([bullet.getQuaternionFromEuler(current[3:6].tolist())])[0]
        # Continuous representation of observed orientation, not future targets.
        values.append(np.concatenate([current[:3],rotation[:,0],rotation[:,1]]))
    result=np.array(values)
    if not np.isfinite(result).all(): raise ValueError('Invalid current pose inputs')
    return result


def outputs(model,projection,c,state,current,targets=None):
    pose_context=c+projection(state)
    raw=model.regression_head(pose_context).reshape(-1,model.chunk_size,7)
    if targets is not None:
        logits=model.predict_gripper_logits(c,targets[...,:7],current)
        return raw,logits
    xyz=raw[...,:3]
    if model.max_normalized_position is not None:
        xyz=xyz.clamp(-model.max_normalized_position,model.max_normalized_position)
    q=raw[...,3:7]
    identity=torch.zeros_like(q); identity[...,3]=1
    q=torch.where(q.norm(dim=-1,keepdim=True)>1e-6,F.normalize(q,dim=-1),identity)
    pose=torch.cat([xyz,q],dim=-1)
    logits=model.predict_gripper_logits(c,pose,current)
    grip=torch.where(logits>=0,torch.ones_like(logits),-torch.ones_like(logits))
    return torch.cat([pose,grip.unsqueeze(-1)],dim=-1),logits


@torch.no_grad()
def evaluate(model,projection,data):
    predicted=[]; teachers=[]; loss_sum=0.
    for start in range(0,len(data['targets']),64):
        s=slice(start,start+64); c=data['context'][s]; state=data['state'][s]
        target=data['targets'][s]; current=data['current'][s]
        raw,teacher=outputs(model,projection,c,state,current,target)
        loss,_,_=policy_loss(model,(raw,target[...,:7],teacher),target,torch.nn.MSELoss(),
            current_grippers=current,supervision_masks=data['masks'][s])
        prediction,_=outputs(model,projection,c,state,current)
        predicted.append(prediction.cpu().numpy().astype(np.float64)); teachers.append(teacher.cpu().numpy())
        loss_sum+=float(loss)*len(target)
    prediction=np.concatenate(predicted); teacher=np.concatenate(teachers)
    target=data['targets'].cpu().numpy().astype(np.float64)
    m=metrics(prediction,target,teacher); m['loss']=loss_sum/len(target)
    return m,prediction,teacher


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',default='results/bridge_single_task_full_validation_v1/latest.pt')
    parser.add_argument('--features',default='results/bridge_input_neighbors_v1/feature_cache.pt')
    parser.add_argument('--audit',default='results/bridge_module_diagnostic_v3/data_contract_audit.json')
    parser.add_argument('--output-dir',default='results/bridge_current_pose_input_v1')
    parser.add_argument('--cache-dir',default='D:/ntu_related/dissertation/hf_cache')
    parser.add_argument('--steps',type=int,default=1500)
    parser.add_argument('--batch-size',type=int,default=64)
    parser.add_argument('--eval-interval',type=int,default=250)
    parser.add_argument('--learning-rate',type=float,default=3e-4)
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    if min(args.steps,args.batch_size,args.eval_interval)<1 or args.learning_rate<=0: parser.error('Invalid fixed budget')
    output=Path(args.output_dir)
    if output.exists() and any(output.iterdir()): raise ValueError('Use a fresh output directory')
    torch.set_num_threads(4); set_seed(args.seed)
    source=Path(args.checkpoint); source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
    checkpoint=torch.load(source,map_location='cpu',weights_only=False)
    cfg=checkpoint['config']['model']
    if cfg['decoder_type']!='regression' or cfg['gripper_target_mode']!='state' or cfg.get('dropout',.1)!=0:
        raise ValueError('Requires deterministic regression/state model')
    features=torch.load(args.features,map_location='cpu',weights_only=False)
    audit=json.loads(Path(args.audit).read_text(encoding='utf-8'))
    if (features['dataset_identity']!=checkpoint['dataset_identity'] or features['checkpoint_sha256']!=source_hash
            or audit['dataset_identity']!=checkpoint['dataset_identity'] or audit['passed_windows']!=audit['windows']):
        raise ValueError('Feature or audit provenance mismatch')
    ds=reconstruct(checkpoint)
    partitions={p:list(checkpoint['split_indices'][p]) for p in ('train','validation')}
    indices=partitions['train']+partitions['validation']
    if set(indices)!=set(features['indices']): raise ValueError('Cached window scope mismatch')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lookup={i:k for k,i in enumerate(features['indices'])}
    context=features['adapter_context'][[lookup[i] for i in indices]].to(device)
    decoded=[ds[i] for i in indices]
    _,_,current,targets,masks=collate_batch(decoded)
    pose=raw_pose_inputs(ds,indices); n=len(partitions['train'])
    mean=pose[:n].mean(0); std=np.maximum(pose[:n].std(0),1e-6)
    normalized=(pose-mean)/std
    output.mkdir(parents=True,exist_ok=True)
    normalization={'fields':['x_m','y_m','z_m','R00','R10','R20','R01','R11','R21'],
        'mean':mean.tolist(),'std':std.tolist(),'fitted_on_train_indices':partitions['train'],
        'future_pose_used_as_input':False,'state_rows':'observation/state[start_index] only',
        'validation_max_abs_standardized_state':float(np.abs(normalized[n:]).max())}
    save_json(output/'state_normalization.json',normalization)
    torch.save({'indices':indices,'raw_current_pose':torch.tensor(pose),'normalized_current_pose':torch.tensor(normalized),
        'dataset_identity':checkpoint['dataset_identity']},output/'current_pose_cache.pt')
    base={'context':context,'state':torch.tensor(normalized,dtype=torch.float32,device=device),
        'current':current.to(device),'targets':targets.to(device),'masks':masks.to(device)}
    real={p:{k:v[s] for k,v in base.items()} for p,s in (('train',slice(0,n)),('validation',slice(n,None)))}
    model=RobotAdapterModel(checkpoint['config'],cache_dir=args.cache_dir).to(device).eval()
    initial=copy.deepcopy(checkpoint['trainable_state_dict'])
    missing=model.load_state_dict(initial,strict=False)
    if missing.unexpected_keys or {k for k,p in model.named_parameters() if p.requires_grad}.intersection(missing.missing_keys):
        raise ValueError('Missing model parameters')
    for name,p in model.named_parameters(): p.requires_grad=not name.startswith(('vision_encoder.','text_encoder.','adapter.'))
    projection=torch.nn.Linear(9,context.shape[-1],bias=False).to(device)
    torch.nn.init.zeros_(projection.weight)
    with torch.no_grad():
        baseline=model.sample(context[:8],current[:8].to(device))
        trial,_=outputs(model,projection,context[:8],base['state'][:8],current[:8].to(device))
        equivalence=float((baseline-trial).abs().max())
        if equivalence>1e-6: raise ValueError('Initial sampling differs from original model')
    generator=torch.Generator().manual_seed(args.seed)
    schedule=torch.stack([torch.randperm(n,generator=generator)[:min(n,args.batch_size)] for _ in range(args.steps)])
    torch.save({'schedule':schedule,'partition_indices':partitions},output/'batch_schedule.pt')
    frozen_before=digest_state({k:p for k,p in model.named_parameters() if not p.requires_grad})
    report={'purpose':'matched_current_pose_information_diagnostic_not_robot_success','arguments':vars(args),
        'source_checkpoint_sha256':source_hash,'dataset_identity':checkpoint['dataset_identity'],'config':checkpoint['config'],
        'partition_indices':partitions,'normalization':normalization,'initial_sampling_max_error':equivalence,
        'optimizer':'AdamW','weight_decay':0,'changed_factor':'real normalized current EEF pose versus zero input',
        'state_projection':'9 -> context_dim, no bias, zero initialization; pose head only',
        'clip_adapter_frozen_in_both':True,'schedule_sha256':digest_state({'schedule':schedule}),
        'test_targets_used':False,'validation_used_for_optimization':False,'arms':{},
        'limits':['Single seed and warm start, fixed 1500-step diagnostic.',
            'Absolute position is current Bridge robot-frame state, not cross-embodiment invariant.',
            'This tests current pose information only; no phase/history or future targets as inputs.',
            'Gripper training keeps original context and teacher pose, so only predicted-pose propagation can differ.']}
    for arm in ('zero_pose','real_pose'):
        set_seed(args.seed); model.load_state_dict(initial,strict=False); torch.nn.init.zeros_(projection.weight)
        data={p:{**v,'state':torch.zeros_like(v['state']) if arm=='zero_pose' else v['state']} for p,v in real.items()}
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad]+list(projection.parameters()),
            lr=args.learning_rate,weight_decay=0)
        history=[]; report['arms'][arm]={'history':history,
            'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad)+sum(p.numel() for p in projection.parameters())}
        for step in range(args.steps+1):
            if step:
                chosen=schedule[step-1].to(device); train=data['train']
                raw,logits=outputs(model,projection,train['context'][chosen],train['state'][chosen],
                    train['current'][chosen],train['targets'][chosen])
                loss,_,_=policy_loss(model,(raw,train['targets'][chosen,:,:7],logits),train['targets'][chosen],torch.nn.MSELoss(),
                    current_grippers=train['current'][chosen],supervision_masks=train['masks'][chosen])
                if not torch.isfinite(loss): raise ValueError('Nonfinite diagnostic loss')
                optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            if step%args.eval_interval==0 or step==args.steps:
                evaluated={p:evaluate(model,projection,d) for p,d in data.items()}
                row={'step':step,**{p:v[0] for p,v in evaluated.items()}}; history.append(row)
                t,v=row['train'],row['validation']
                print(f"[位姿输入对照] {arm} {step}: train={t['position_error_cm']:.2f}cm/{t['rotation_error_deg']:.2f}deg; validation={v['position_error_cm']:.2f}cm/{v['rotation_error_deg']:.2f}deg switch={v['closed_to_open_correct']}/{v['closed_to_open_pairs']}",flush=True)
                torch.save({'experiment_kind':'current_pose_input_diagnostic','arm':arm,'step':step,
                    'head_state_dict':{k:p.detach().cpu().clone() for k,p in model.named_parameters() if p.requires_grad},
                    'state_projection':projection.state_dict(),'normalization':normalization,'config':checkpoint['config'],
                    'source_checkpoint_sha256':source_hash,'metrics':row,'not_deployment_checkpoint':True},output/f'{arm}_step_{step:04d}.pt')
                if step==args.steps:
                    values={i:{'prediction':evaluated['validation'][1][j],'target':data['validation']['targets'][j].cpu().numpy().astype(np.float64),
                        'teacher_logits':evaluated['validation'][2][j]} for j,i in enumerate(partitions['validation'])}
                    save_json(output/f'{arm}_validation_episodes.json',group_report(ds,partitions['validation'],values))
                save_json(output/'comparison.json',report)
        if digest_state({k:p for k,p in model.named_parameters() if not p.requires_grad})!=frozen_before:
            raise ValueError('Frozen CLIP/Adapter changed')
        if arm=='zero_pose' and bool(projection.weight.detach().ne(0).any()): raise ValueError('Zero input projection should remain zero')
        report['arms'][arm]['final']=history[-1]
        save_json(output/'comparison.json',report)
    if not initial_metrics_match(report['arms']['zero_pose']['history'][0],report['arms']['real_pose']['history'][0]):
        raise ValueError('Initial arm metrics differ')
    if hashlib.sha256(source.read_bytes()).hexdigest()!=source_hash: raise ValueError('Original source checkpoint changed')
    report['completed']=True
    report['verification']={'same_initial_predictions':True,'same_batch_sequence':True,'same_architecture_parameter_count':True,
        'frozen_clip_adapter_unchanged':True,'normalization_train_only':True,'source_checkpoint_unchanged':True}
    save_json(output/'comparison.json',report)
    columns=['arm','step','partition','position_error_cm','rotation_error_deg','gripper_accuracy','balanced_accuracy',
        'open_to_closed_correct','open_to_closed_pairs','closed_to_open_correct','closed_to_open_pairs','loss']
    with (output/'history.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore'); writer.writeheader()
        for arm,a in report['arms'].items():
            for row in a['history']:
                for p in ('train','validation'): writer.writerow({'arm':arm,'step':row['step'],'partition':p,**row[p]})
    print('[完成] 当前位姿输入对照',output.resolve(),flush=True)


if __name__=='__main__': main()

"""Matched warm-start diagnosis: only Adapter trainability differs between arms.

Caches deterministic frozen CLIP features. For CLS pooling, queries are tokenwise:
only CLS contributes to the readout. Validate full-vs-CLS outputs and gradients
before using this exact computational shortcut. No production modules changed.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('HF_HUB_OFFLINE','1')
import numpy as np
import torch
from transformers import CLIPTokenizer

from diagnose_bridge_modules import reconstruct, save_json, metrics
from models import RobotAdapterModel
from train import collate_batch, policy_loss, set_seed, tokenise


def digest_state(state):
    digest=hashlib.sha256()
    for name,value in sorted(state.items()):
        digest.update(name.encode()); digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def initial_metrics_match(left,right):
    """CUDA reductions may differ at ~1e-6 despite bit-identical weights."""
    return all(np.allclose(left[p][k],right[p][k],atol=1e-5,rtol=1e-7)
        if left[p][k] is not None and right[p][k] is not None else left[p][k]==right[p][k]
        for p in ('train','validation') for k in left[p])


def check_cls_equivalence(model,visual,text,mask):
    full=model.adapter(visual,text,mask)
    compact=model.adapter(visual[:,:1],text,mask)
    difference=float((full-compact).abs().max().detach())
    weight=torch.randn_like(full)
    params=list(model.adapter.parameters())
    grad_full=torch.autograd.grad((full*weight).sum(),params)
    grad_compact=torch.autograd.grad((compact*weight).sum(),params)
    gradient_error=max(float((a-b).abs().max()) for a,b in zip(grad_full,grad_compact))
    if difference>2e-5 or gradient_error>2e-4:
        raise ValueError(f'CLS shortcut equivalence failed: output={difference}, gradient={gradient_error}')
    return {'max_output_difference':difference,'max_gradient_difference':gradient_error,
        'full_visual_tokens':visual.shape[1],'cached_visual_tokens':1,'pass':True}


@torch.no_grad()
def cache_partition(model,tokenizer,ds,indices,device,equivalence=None):
    batches=[]
    for start in range(0,len(indices),8):
        ids=indices[start:start+8]
        texts,images,current,targets,masks=collate_batch([ds[i] for i in ids])
        encoded=tokenise(tokenizer,texts,device)
        visual=model.vision_encoder(pixel_values=images.to(device)).last_hidden_state
        text=model.text_encoder(input_ids=encoded['input_ids'],attention_mask=encoded.get('attention_mask')).last_hidden_state
        if start==0 and equivalence is not None:
            with torch.enable_grad():
                equivalence.update(check_cls_equivalence(model,visual.detach(),text.detach(),encoded.get('attention_mask')))
        batches.append({'visual':visual[:,:1].detach(),'text':text.detach(),'attention_mask':encoded['attention_mask'].detach(),
            'current':current.to(device),'targets':targets.to(device),'masks':masks.to(device)})
    return {key:torch.cat([b[key] for b in batches]) for key in batches[0]}


def context(model,data,chosen):
    return model.adapter(data['visual'][chosen],data['text'][chosen],data['attention_mask'][chosen])


@torch.no_grad()
def evaluate(model,data):
    predictions=[]; teacher=[]; weighted_loss=0.; weighted_pose=0.; weighted_grip=0.
    for start in range(0,len(data['targets']),64):
        selection=slice(start,start+64)
        c=context(model,data,selection)
        targets=data['targets'][selection]; current=data['current'][selection]
        raw=model.regression_head(c).reshape(-1,model.chunk_size,7)
        logits=model.predict_gripper_logits(c,targets[...,:7],current)
        losses=policy_loss(model,(raw,targets[...,:7],logits),targets,torch.nn.MSELoss(),
            current_grippers=current,supervision_masks=data['masks'][selection])
        count=len(targets)
        weighted_loss+=float(losses[0])*count; weighted_pose+=float(losses[1])*count; weighted_grip+=float(losses[2])*count
        predictions.append(model.sample(c,current).cpu().numpy().astype(np.float64))
        teacher.append(logits.cpu().numpy())
    result=metrics(np.concatenate(predictions),data['targets'].cpu().numpy().astype(np.float64),np.concatenate(teacher))
    result.update(loss=weighted_loss/len(data['targets']),pose_loss=weighted_pose/len(data['targets']),gripper_bce=weighted_grip/len(data['targets']))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',default='results/bridge_single_task_full_validation_v1/latest.pt')
    parser.add_argument('--audit',default='results/bridge_module_diagnostic_v3/data_contract_audit.json')
    parser.add_argument('--output-dir',default='results/bridge_adapter_trainability_v1')
    parser.add_argument('--cache-dir',default='D:/ntu_related/dissertation/hf_cache')
    parser.add_argument('--steps',type=int,default=1500)
    parser.add_argument('--batch-size',type=int,default=64)
    parser.add_argument('--eval-interval',type=int,default=250)
    parser.add_argument('--learning-rate',type=float,default=3e-4)
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    if min(args.steps,args.batch_size,args.eval_interval)<1 or args.learning_rate<=0: parser.error('Invalid fixed budget')
    output=Path(args.output_dir)
    if output.exists() and any(output.iterdir()): raise ValueError('Use a new output directory')
    torch.set_num_threads(4); set_seed(args.seed)
    source=Path(args.checkpoint)
    source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
    checkpoint=torch.load(source,map_location='cpu',weights_only=False)
    config=checkpoint['config']; cfg=config['model']
    if cfg['decoder_type']!='regression' or cfg.get('adapter_pooling','cls')!='cls' or cfg.get('dropout',.1)!=0:
        raise ValueError('This exact cached comparison requires deterministic CLS regression checkpoint')
    ds=reconstruct(checkpoint)
    audit=json.loads(Path(args.audit).read_text(encoding='utf-8'))
    if audit.get('dataset_identity')!=checkpoint['dataset_identity'] or audit['windows']!=audit['passed_windows']:
        raise ValueError('Independent data audit not passed for this identity')
    partitions={p:list(checkpoint['split_indices'][p]) for p in ('train','validation')}
    if not set(partitions['train']+partitions['validation']).issubset({r['index'] for r in audit['rows']}):
        raise ValueError('Unaudited windows')
    output.mkdir(parents=True,exist_ok=True)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=RobotAdapterModel(config,cache_dir=args.cache_dir).to(device).eval()
    initial=copy.deepcopy(checkpoint['trainable_state_dict'])
    missing=model.load_state_dict(initial,strict=False)
    trainable={n for n,p in model.named_parameters() if p.requires_grad}
    if missing.unexpected_keys or trainable.intersection(missing.missing_keys): raise ValueError('Missing trainable weights')
    tokenizer=CLIPTokenizer.from_pretrained(cfg['name'],cache_dir=args.cache_dir,local_files_only=True)
    equivalence={}
    data={p:cache_partition(model,tokenizer,ds,ids,device,equivalence if p=='train' else None) for p,ids in partitions.items()}
    # Materialize one shared schedule, independently of optimizer/evaluation RNG.
    generator=torch.Generator().manual_seed(args.seed)
    schedule=torch.stack([torch.randperm(len(partitions['train']),generator=generator)[:min(args.batch_size,len(partitions['train']))] for _ in range(args.steps)])
    torch.save({'partition_indices':partitions,'schedule':schedule,'dataset_identity':checkpoint['dataset_identity']},output/'batch_schedule.pt')
    report={'purpose':'matched_adapter_trainability_warm_start_diagnostic_not_robot_success',
        'arguments':vars(args),'config':config,'source_checkpoint_sha256':source_hash,'dataset_identity':checkpoint['dataset_identity'],
        'partition_indices':partitions,'clip_feature_equivalence':equivalence,'schedule_sha256':digest_state({'schedule':schedule}),
        'test_targets_used':False,'validation_used_for_optimization':False,'changed_factor':'Adapter requires_grad only',
        'optimizer':'AdamW','weight_decay':0,'common_learning_rate':args.learning_rate,'initial_state_sha256':digest_state(initial),
        'arms':{},'limits':['Single-seed, warm-start diagnosis; not a from-scratch main experiment.',
            'No branch-specific tuning or early termination. Validation is measured at fixed intervals.',
            'Only CLS queries are cached after checking equivalence to full-token outputs and Adapter gradients.']}
    frozen_original=digest_state({n:p for n,p in model.named_parameters() if n.startswith(('vision_encoder.','text_encoder.'))})
    adapter_original=digest_state({n:p for n,p in model.named_parameters() if n.startswith('adapter.')})
    for arm in ('adapter_frozen','adapter_joint'):
        set_seed(args.seed); model.load_state_dict(initial,strict=False); model.eval()
        for name,p in model.named_parameters():
            p.requires_grad=not name.startswith(('vision_encoder.','text_encoder.')) and (arm=='adapter_joint' or not name.startswith('adapter.'))
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=args.learning_rate,weight_decay=0)
        history=[]
        report['arms'][arm]={'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'history':history}
        for step in range(args.steps+1):
            if step:
                chosen=schedule[step-1].to(device); training=data['train']
                c=context(model,training,chosen)
                targets=training['targets'][chosen]; current=training['current'][chosen]
                raw=model.regression_head(c).reshape(-1,model.chunk_size,7)
                logits=model.predict_gripper_logits(c,targets[...,:7],current)
                loss,_,_=policy_loss(model,(raw,targets[...,:7],logits),targets,torch.nn.MSELoss(),
                    current_grippers=current,supervision_masks=training['masks'][chosen])
                if not torch.isfinite(loss): raise RuntimeError(f'Nonfinite loss in {arm} at {step}')
                optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            if step%args.eval_interval==0 or step==args.steps:
                row={'step':step,**{p:evaluate(model,d) for p,d in data.items()}}; history.append(row)
                t,v=row['train'],row['validation']
                print(f"[严格对照] {arm} step={step}: train={t['position_error_cm']:.2f}cm/{t['rotation_error_deg']:.2f}deg switch={t['closed_to_open_correct']}/{t['closed_to_open_pairs']}; validation={v['position_error_cm']:.2f}cm/{v['rotation_error_deg']:.2f}deg switch={v['closed_to_open_correct']}/{v['closed_to_open_pairs']}",flush=True)
                state={n:p.detach().cpu().clone() for n,p in model.named_parameters() if n in initial}
                torch.save({'experiment_kind':'matched_adapter_trainability_diagnostic','arm':arm,'step':step,
                    'trainable_state_dict':state,'config':config,'dataset_identity':checkpoint['dataset_identity'],
                    'source_checkpoint':str(source.resolve()),'source_checkpoint_sha256':source_hash,'metrics':row,
                    'not_standard_deployment_checkpoint':True},output/f'{arm}_step_{step:04d}.pt')
                save_json(output/'comparison.json',report)
        adapter_final=digest_state({n:p for n,p in model.named_parameters() if n.startswith('adapter.')})
        report['arms'][arm]['adapter_changed']=adapter_final!=adapter_original
        if (arm=='adapter_frozen') != (adapter_final==adapter_original): raise RuntimeError('Adapter trainability contract violated')
        if digest_state({n:p for n,p in model.named_parameters() if n.startswith(('vision_encoder.','text_encoder.'))})!=frozen_original:
            raise RuntimeError('Frozen CLIP changed')
        report['arms'][arm]['final']=history[-1]
        report['arms'][arm]['clip_unchanged']=True
        save_json(output/'comparison.json',report)
    if not initial_metrics_match(report['arms']['adapter_frozen']['history'][0],report['arms']['adapter_joint']['history'][0]):
        raise RuntimeError('Arm initial predictions differ')
    if hashlib.sha256(source.read_bytes()).hexdigest()!=source_hash: raise RuntimeError('Source checkpoint changed')
    report['verification']={'same_initial_predictions':True,'same_batch_schedule':True,'same_head_learning_rate':True,
        'initial_metric_atol':1e-5,'clip_unchanged':True,'adapter_frozen_unchanged':True,'adapter_joint_updated':True,'source_checkpoint_unchanged':True}
    report['completed']=True
    save_json(output/'comparison.json',report)
    print('[完成] Adapter训练状态严格对照：',output.resolve(),flush=True)


if __name__=='__main__': main()

"""Matched frozen-Adapter readout diagnosis: CLS vs CLS+patch_mean.

Same parameters, head initialization, train-only updates and batch schedule.
CLIP and Adapter frozen in both arms. Readout information alone differs.
This is not an end-to-end pooling or robot task success experiment.
"""
import argparse
import copy
import csv
import hashlib
import os
from pathlib import Path
os.environ.setdefault('HF_HUB_OFFLINE','1')
import numpy as np
import torch
from transformers import CLIPTokenizer
from models import RobotAdapterModel
from train import collate_batch,policy_loss,set_seed,tokenise
from diagnose_bridge_modules import reconstruct,save_json,metrics,group_report
from compare_bridge_adapter_training import digest_state


@torch.no_grad()
def evaluate(model,data):
    predicted=[];teachers=[];total=0.
    for start in range(0,len(data['targets']),64):
        s=slice(start,start+64);c=data['context'][s];target=data['targets'][s];current=data['current'][s]
        raw=model.regression_head(c).reshape(-1,model.chunk_size,7)
        teacher=model.predict_gripper_logits(c,target[...,:7],current)
        loss,_,_=policy_loss(model,(raw,target[...,:7],teacher),target,torch.nn.MSELoss(),
            current_grippers=current,supervision_masks=data['masks'][s])
        total+=float(loss)*len(target);predicted.append(model.sample(c,current).cpu().numpy().astype(np.float64));teachers.append(teacher.cpu().numpy())
    prediction=np.concatenate(predicted);teacher=np.concatenate(teachers)
    m=metrics(prediction,data['targets'].cpu().numpy().astype(np.float64),teacher)
    m['loss']=total/len(data['targets'])
    return m,prediction,teacher


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',default='results/bridge_single_task_full_validation_v1/latest.pt')
    parser.add_argument('--features',default='results/bridge_input_neighbors_v1/feature_cache.pt')
    parser.add_argument('--audit',default='results/bridge_module_diagnostic_v3/data_contract_audit.json')
    parser.add_argument('--output-dir',default='results/bridge_pooling_readout_v1')
    parser.add_argument('--cache-dir',default='D:/ntu_related/dissertation/hf_cache')
    parser.add_argument('--steps',type=int,default=1500);parser.add_argument('--batch-size',type=int,default=64)
    parser.add_argument('--eval-interval',type=int,default=250);parser.add_argument('--learning-rate',type=float,default=3e-4)
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    if min(args.steps,args.batch_size,args.eval_interval)<1 or args.learning_rate<=0:parser.error('Invalid budget')
    out=Path(args.output_dir)
    if out.exists() and any(out.iterdir()):raise ValueError('Use a fresh output directory')
    torch.set_num_threads(4);set_seed(args.seed)
    source=Path(args.checkpoint);source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
    checkpoint=torch.load(source,map_location='cpu',weights_only=False);cfg=checkpoint['config']['model']
    if cfg['decoder_type']!='regression' or cfg.get('adapter_pooling','cls')!='cls' or cfg.get('dropout',.1)!=0:
        raise ValueError('Requires deterministic CLS regression source')
    cache=torch.load(args.features,map_location='cpu',weights_only=False)
    import json
    audit=json.loads(Path(args.audit).read_text(encoding='utf-8'))
    if cache['dataset_identity']!=checkpoint['dataset_identity'] or cache['checkpoint_sha256']!=source_hash or audit['dataset_identity']!=checkpoint['dataset_identity'] or audit['windows']!=audit['passed_windows']:
        raise ValueError('Provenance/audit mismatch')
    ds=reconstruct(checkpoint);partitions={p:list(checkpoint['split_indices'][p]) for p in ['train','validation']}
    indices=partitions['train']+partitions['validation'];n=len(partitions['train'])
    if set(indices)!=set(cache['indices']) or not set(indices).issubset({row['index'] for row in audit['rows']}):raise ValueError('Unaudited/mismatched scope')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=RobotAdapterModel(checkpoint['config'],cache_dir=args.cache_dir).to(device).eval()
    initial=copy.deepcopy(checkpoint['trainable_state_dict']);missing=model.load_state_dict(initial,strict=False)
    if missing.unexpected_keys or {k for k,p in model.named_parameters() if p.requires_grad}.intersection(missing.missing_keys):raise ValueError('Missing weights')
    for name,p in model.named_parameters():p.requires_grad=not name.startswith(('vision_encoder.','text_encoder.','adapter.'))
    frozen_hash=digest_state({k:p for k,p in model.named_parameters() if not p.requires_grad})
    adapter_hash=digest_state(model.adapter.state_dict())
    tokenizer=CLIPTokenizer.from_pretrained(cfg['name'],cache_dir=args.cache_dir,local_files_only=True)
    data_lists=[];patch_context=[];cls_context=[]
    with torch.no_grad():
        for start in range(0,len(indices),8):
            texts,images,current,targets,masks=collate_batch([ds[i] for i in indices[start:start+8]])
            encoded=tokenise(tokenizer,texts,device)
            visual=model.vision_encoder(pixel_values=images.to(device)).last_hidden_state
            text=model.text_encoder(input_ids=encoded['input_ids'],attention_mask=encoded['attention_mask']).last_hidden_state
            model.adapter.pooling='cls';cls_context.append(model.adapter(visual,text,encoded['attention_mask']).cpu())
            model.adapter.pooling='cls_patch_mean';patch_context.append(model.adapter(visual,text,encoded['attention_mask']).cpu())
            data_lists.append({'current':current,'targets':targets,'masks':masks})
            if start%64==0:print(f'[汇聚对照] 缓存 {min(start+8,len(indices))}/{len(indices)}',flush=True)
    model.adapter.pooling='cls'
    lookup={i:k for k,i in enumerate(cache['indices'])}
    cls=cache['adapter_context'][[lookup[i] for i in indices]]
    cache_error=float((torch.cat(cls_context)-cls).abs().max())
    if cache_error>2e-5:raise ValueError('Recomputed CLS cache mismatch')
    contexts={'cls':cls,'cls_patch_mean':torch.cat(patch_context)}
    common={k:torch.cat([b[k] for b in data_lists]) for k in data_lists[0]}
    out.mkdir(parents=True,exist_ok=True)
    torch.save({'indices':indices,'contexts':contexts,**common,'dataset_identity':checkpoint['dataset_identity'],
        'source_checkpoint_sha256':source_hash,'adapter_state_sha256':adapter_hash},out/'pooling_context_cache.pt')
    generator=torch.Generator().manual_seed(args.seed)
    schedule=torch.stack([torch.randperm(n,generator=generator)[:min(n,args.batch_size)] for _ in range(args.steps)])
    torch.save({'schedule':schedule,'partition_indices':partitions,'dataset_identity':checkpoint['dataset_identity']},out/'batch_schedule.pt')
    head_initial={k:p.detach().cpu().clone() for k,p in model.named_parameters() if p.requires_grad}
    report={'purpose':'matched_frozen_adapter_pooling_readout_diagnostic_not_success','arguments':vars(args),
        'source_checkpoint_sha256':source_hash,'dataset_identity':checkpoint['dataset_identity'],'partition_indices':partitions,
        'changed_factor':'pooling only: CLS vs 0.5*(CLS+mean patches) after frozen Adapter',
        'adapter_frozen_in_both':True,'clip_frozen_in_both':True,'same_parameter_names':list(head_initial),
        'initial_head_sha256':digest_state(head_initial),'adapter_state_sha256':adapter_hash,'frozen_parameters_sha256':frozen_hash,
        'schedule_sha256':digest_state({'schedule':schedule}),'recomputed_cls_context_max_error':cache_error,
        'optimizer':'AdamW','weight_decay':0,'validation_used_for_optimization':False,'test_targets_used':False,
        'limits':['Single-seed warm start from a CLS-trained model; not a from-scratch or end-to-end pooling comparison.',
            'Initial predictions differ intentionally because readout inputs change, despite identical weights.',
            'Fixed update budget and common learning rate, no branch-specific tuning or early stopping.',
            'Training gripper uses teacher pose; inference uses predicted pose in both arms.'],'arms':{}}
    base={k:v.to(device) for k,v in common.items()}
    for arm in ['cls','cls_patch_mean']:
        set_seed(args.seed);model.load_state_dict(initial,strict=False);model.eval();model.adapter.pooling=arm
        data={p:{**{k:v[s] for k,v in base.items()},'context':contexts[arm][s].to(device)}
            for p,s in [('train',slice(0,n)),('validation',slice(n,None))]}
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=args.learning_rate,weight_decay=0)
        history=[];report['arms'][arm]={'history':history,'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad)}
        for step in range(args.steps+1):
            if step:
                chosen=schedule[step-1].to(device);d=data['train'];c=d['context'][chosen];target=d['targets'][chosen];current=d['current'][chosen]
                raw=model.regression_head(c).reshape(-1,model.chunk_size,7);teacher=model.predict_gripper_logits(c,target[...,:7],current)
                loss,_,_=policy_loss(model,(raw,target[...,:7],teacher),target,torch.nn.MSELoss(),current_grippers=current,supervision_masks=d['masks'][chosen])
                if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
                optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
            if step%args.eval_interval==0 or step==args.steps:
                evaluated={p:evaluate(model,d) for p,d in data.items()};row={'step':step,**{p:v[0] for p,v in evaluated.items()}};history.append(row)
                t,v=row['train'],row['validation']
                print(f"[汇聚对照] {arm} {step}: train={t['position_error_cm']:.3f}cm/{t['rotation_error_deg']:.2f}deg; val={v['position_error_cm']:.2f}cm/{v['rotation_error_deg']:.2f}deg; switches={v['open_to_closed_correct']}/{v['open_to_closed_pairs']},{v['closed_to_open_correct']}/{v['closed_to_open_pairs']}",flush=True)
                config=copy.deepcopy(checkpoint['config']);config['model']['adapter_pooling']=arm
                torch.save({'experiment_kind':'frozen_adapter_pooling_readout_diagnostic','arm':arm,'step':step,
                    'head_state_dict':{k:p.detach().cpu().clone() for k,p in model.named_parameters() if p.requires_grad},
                    'config':config,'source_checkpoint_sha256':source_hash,'adapter_state_sha256':adapter_hash,
                    'dataset_identity':checkpoint['dataset_identity'],'metrics':row,'not_standard_deployment_checkpoint':True},out/f'{arm}_step_{step:04d}.pt')
                values={i:{'prediction':evaluated['validation'][1][j],'target':data['validation']['targets'][j].cpu().numpy().astype(np.float64),
                    'teacher_logits':evaluated['validation'][2][j]} for j,i in enumerate(partitions['validation'])}
                save_json(out/f'{arm}_validation_step_{step:04d}.json',group_report(ds,partitions['validation'],values))
                save_json(out/'comparison.json',report)
        if digest_state({k:p for k,p in model.named_parameters() if not p.requires_grad})!=frozen_hash:raise ValueError('Frozen parameters changed')
        report['arms'][arm]['final']=history[-1];save_json(out/'comparison.json',report)
    if hashlib.sha256(source.read_bytes()).hexdigest()!=source_hash:raise ValueError('Original checkpoint changed')
    a,b=[torch.load(out/f'{arm}_step_0000.pt',map_location='cpu',weights_only=False)['head_state_dict'] for arm in ['cls','cls_patch_mean']]
    if digest_state(a)!=digest_state(b) or digest_state(a)!=report['initial_head_sha256']:raise ValueError('Initial weights mismatch')
    if report['arms']['cls']['trainable_parameters']!=report['arms']['cls_patch_mean']['trainable_parameters']:raise ValueError('Parameter count mismatch')
    report['completed']=True;report['verification']={'identical_initial_head_weights':True,'same_architecture_parameter_count':True,
        'frozen_clip_adapter_unchanged':True,'source_checkpoint_unchanged':True,'same_batch_schedule':True}
    save_json(out/'comparison.json',report)
    columns=['arm','step','partition','position_error_cm','rotation_error_deg','gripper_accuracy','balanced_accuracy',
        'open_to_closed_correct','open_to_closed_pairs','closed_to_open_correct','closed_to_open_pairs','loss']
    with (out/'history.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');writer.writeheader()
        for arm,a in report['arms'].items():
            for row in a['history']:
                for p in ['train','validation']:writer.writerow({'arm':arm,'step':row['step'],'partition':p,**row[p]})
    print('[完成]',out.resolve(),flush=True)


if __name__=='__main__':main()

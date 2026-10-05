"""Bounded train-only head probes; never modifies the original model checkpoint.

Compare frozen current contexts with individually learnable window contexts.
Both arms start with identical heads/contexts, use identical batches and losses.
The window-ID arm is a memorization control, not a deployable robot policy.
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from transformers import CLIPTokenizer

from diagnose_bridge_modules import reconstruct, save_json, metrics
from models import RobotAdapterModel
from train import collate_batch, policy_loss, set_seed, tokenise


@torch.no_grad()
def measure(model, contexts, targets, current):
    predictions=model.sample(contexts,current)
    teacher=model.predict_gripper_logits(contexts,targets[...,:7],current)
    return metrics(predictions.cpu().numpy().astype(np.float64),targets.cpu().numpy().astype(np.float64),teacher.cpu().numpy())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',default='results/bridge_single_task_full_validation_v1/latest.pt')
    parser.add_argument('--output-dir',default='results/bridge_head_isolation_v1')
    parser.add_argument('--audit',default='results/bridge_module_diagnostic_v3/data_contract_audit.json')
    parser.add_argument('--cache-dir',default='D:/ntu_related/dissertation/hf_cache')
    parser.add_argument('--steps',type=int,default=1500)
    parser.add_argument('--batch-size',type=int,default=64)
    parser.add_argument('--learning-rate',type=float,default=3e-4)
    args=parser.parse_args()
    if args.steps<1 or args.batch_size<1 or args.learning_rate<=0: parser.error('Invalid probe budget')
    output=Path(args.output_dir)
    if output.exists() and any(output.iterdir()): raise ValueError('Use a fresh probe output directory')
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    if checkpoint['config']['model']['decoder_type']!='regression': raise ValueError('Regression only')
    audit=json.loads(Path(args.audit).read_text(encoding='utf-8'))
    if audit['passed_windows']!=audit['windows']: raise ValueError('Independent contract audit did not pass')
    torch.set_num_threads(4); set_seed(42)
    ds=reconstruct(checkpoint)
    if audit.get('dataset_identity')!=checkpoint['dataset_identity']:
        raise ValueError('Audit belongs to a different dataset identity')
    indices=list(checkpoint['split_indices']['train'])
    if not set(indices).issubset({r['index'] for r in audit['rows']}): raise ValueError('Unaudited probe samples')
    output.mkdir(parents=True,exist_ok=True)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=RobotAdapterModel(checkpoint['config'],cache_dir=args.cache_dir).to(device).eval()
    initial=copy.deepcopy(checkpoint['trainable_state_dict'])
    missing=model.load_state_dict(initial,strict=False)
    trainable_names={n for n,p in model.named_parameters() if p.requires_grad}
    if missing.unexpected_keys or trainable_names.intersection(missing.missing_keys): raise ValueError('Missing trainable weights')
    tokenizer=CLIPTokenizer.from_pretrained(checkpoint['config']['model']['name'],cache_dir=args.cache_dir,local_files_only=True)
    contexts,targets,current,masks=[],[],[],[]
    with torch.no_grad():
        for start in range(0,len(indices),8):
            text,image,grip,target,mask=collate_batch([ds[i] for i in indices[start:start+8]])
            encoded=tokenise(tokenizer,text,device)
            contexts.append(model.get_context_vector(image.to(device),encoded['input_ids'],encoded.get('attention_mask')).detach())
            targets.append(target.to(device)); current.append(grip.to(device)); masks.append(mask.to(device))
    contexts,targets,current,masks=map(torch.cat,(contexts,targets,current,masks))
    torch.save({'indices':indices,'contexts':contexts.cpu(),'targets':targets.cpu(),'current':current.cpu(),
        'masks':masks.cpu(),'checkpoint':args.checkpoint,'dataset_identity':checkpoint['dataset_identity']},output/'train_context_cache.pt')
    report={'purpose':'train_only_head_isolation_not_generalization_or_control_success','arguments':vars(args),
        'dataset_identity':checkpoint['dataset_identity'],'selected_indices':indices,'seed':42,
        'test_validation_targets_used':False,'original_checkpoint_modified':False,'config':checkpoint['config'],
        'interpretation_limits':['Both arms adapt diagnostic copies of existing heads; no new architecture or label change.',
            'Learnable per-window contexts are an oracle memorization control and cannot be used at deployment.',
            'Same fixed budget failure does not prove impossibility; learning rate differs from the original pose group.',
            'Frozen-context success establishes that the saved representation permits training-set fitting, not generalization.'],
        'arms':{}}
    frozen_digest=[p.detach().cpu().clone() for p in model.adapter.parameters()]
    for arm in ('frozen_context','window_id_context'):
        set_seed(42); model.load_state_dict(initial,strict=False)
        for name,p in model.named_parameters():
            p.requires_grad=not name.startswith(('vision_encoder.','text_encoder.','adapter.'))
        model.eval()  # dropout=0; frozen representations are deterministic
        context_values=torch.nn.Parameter(contexts.clone(),requires_grad=arm=='window_id_context')
        parameters=[p for p in model.parameters() if p.requires_grad]
        groups=[{'params':parameters,'lr':args.learning_rate}]
        if context_values.requires_grad: groups.append({'params':[context_values],'lr':args.learning_rate})
        optimizer=torch.optim.AdamW(groups,weight_decay=0)
        generator=torch.Generator().manual_seed(42)
        history=[{'step':0,**measure(model,context_values,targets,current)}]
        report['arms'][arm]={'history':history}
        for step in range(1,args.steps+1):
            chosen=torch.randperm(len(indices),generator=generator)[:min(args.batch_size,len(indices))].to(device)
            context=context_values[chosen]
            poses=model.regression_head(context).reshape(-1,model.chunk_size,7)
            logits=model.predict_gripper_logits(context,targets[chosen,:,:7],current[chosen])
            loss,pose_loss,grip_loss=policy_loss(model,(poses,targets[chosen,:,:7],logits),targets[chosen],
                torch.nn.MSELoss(),current_grippers=current[chosen],supervision_masks=masks[chosen])
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            if step%250==0 or step==args.steps:
                row={'step':step,'batch_loss':float(loss.detach()),**measure(model,context_values,targets,current)}
                history.append(row)
                print(f"[隔离诊断] {arm} {step}/{args.steps}: xyz={row['position_error_cm']:.2f}cm rot={row['rotation_error_deg']:.2f}deg pairs={row['open_to_closed_correct']}/{row['open_to_closed_pairs']}, {row['closed_to_open_correct']}/{row['closed_to_open_pairs']}",flush=True)
                save_json(output/'probe_results.json',report)
        report['arms'][arm]['final']=history[-1]
        if any(not torch.equal(old,p.detach().cpu()) for old,p in zip(frozen_digest,model.adapter.parameters())):
            raise RuntimeError('Frozen Adapter changed unexpectedly')
        torch.save({'experiment_kind':'training_fit_diagnostic','arm':arm,'config':checkpoint['config'],
            'head_state_dict':{n:p.detach().cpu() for n,p in model.named_parameters() if p.requires_grad},
            'train_window_contexts':context_values.detach().cpu(),'indices':indices,
            'dataset_identity':checkpoint['dataset_identity'],'seed':42,'steps':args.steps,
            'not_a_deployable_checkpoint':True},output/f'{arm}_diagnostic.pt')
    save_json(output/'probe_results.json',report)
    print('[完成] 两组固定预算诊断副本已保存；原始权重未修改',output.resolve(),flush=True)


if __name__=='__main__': main()

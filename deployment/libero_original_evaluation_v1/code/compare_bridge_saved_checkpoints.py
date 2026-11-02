"""Read-only full train/validation comparison of original best and latest weights."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
os.environ.setdefault('HF_HUB_OFFLINE','1')
import numpy as np
import torch
from transformers import CLIPTokenizer
from models import RobotAdapterModel
from train import set_seed
from compare_bridge_adapter_training import digest_state
from diagnose_bridge_modules import reconstruct,save_json,predict,group_report


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--best',default='results/bridge_single_task_full_validation_v1/best.pt')
    parser.add_argument('--latest',default='results/bridge_single_task_full_validation_v1/latest.pt')
    parser.add_argument('--audit',default='results/bridge_module_diagnostic_v3/data_contract_audit.json')
    parser.add_argument('--output-dir',default='results/bridge_saved_checkpoints_v1')
    parser.add_argument('--cache-dir',default='D:/ntu_related/dissertation/hf_cache')
    parser.add_argument('--batch-size',type=int,default=8)
    args=parser.parse_args()
    if args.batch_size<1:parser.error('batch-size must be positive')
    out=Path(args.output_dir)
    if out.exists() and any(out.iterdir()):raise ValueError('Use a fresh output directory')
    torch.set_num_threads(4);set_seed(42)
    paths={'best':Path(args.best),'latest':Path(args.latest)}
    hashes={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in paths.items()}
    checkpoints={name:torch.load(path,map_location='cpu',weights_only=False) for name,path in paths.items()}
    reference=checkpoints['latest']
    for c in checkpoints.values():
        for key in ['config','data_config','dataset_identity','dataset_size','split_indices']:
            if c[key]!=reference[key]:raise ValueError(f'Checkpoint mismatch: {key}')
        if c['config']['model']['decoder_type']!='regression':raise ValueError('Requires deterministic regression source')
        if c.get('validation_protocol')!='all_held_out_bridge_windows':raise ValueError('Unexpected original validation protocol')
    ds=reconstruct(reference);partitions={p:list(reference['split_indices'][p]) for p in ['train','validation']}
    indices=partitions['train']+partitions['validation']
    audit=json.loads(Path(args.audit).read_text(encoding='utf-8'))
    if audit['dataset_identity']!=reference['dataset_identity'] or audit['windows']!=audit['passed_windows'] or set(indices)!={r['index'] for r in audit['rows']}:
        raise ValueError('Audit mismatch or incomplete window coverage')
    decoded={i:ds[i] for i in indices}
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=RobotAdapterModel(reference['config'],cache_dir=args.cache_dir).to(device).eval()
    tokenizer=CLIPTokenizer.from_pretrained(reference['config']['model']['name'],cache_dir=args.cache_dir,local_files_only=True)
    out.mkdir(parents=True,exist_ok=True)
    report={'purpose':'read_only_best_latest_full_action_metrics_not_robot_success','arguments':vars(args),
        'trained':False,'test_targets_used':False,'validation_used_for_optimization':False,
        'dataset_identity':reference['dataset_identity'],'config':reference['config'],'partition_indices':partitions,
        'checkpoints':{},'limits':['Original best was selected by aggregate validation loss, not action success.',
            'Validation has only three independent demonstrations; window and transition counts overlap.',
            'Same validation set already used for checkpoint selection and repeated diagnostics; it is not a new test set.',
            'No diffusion, language discrimination, simulation success or embodiment transfer is assessed.']}
    frozen_before=digest_state({k:p for k,p in model.named_parameters() if k.startswith(('vision_encoder.','text_encoder.'))})
    for name,c in checkpoints.items():
        missing=model.load_state_dict(c['trainable_state_dict'],strict=False);model.eval()
        if missing.unexpected_keys or {k for k,p in model.named_parameters() if p.requires_grad}.intersection(missing.missing_keys):raise ValueError('Missing checkpoint parameters')
        before=digest_state({k:p for k,p in model.named_parameters() if k in c['trainable_state_dict']})
        if before!=digest_state(c['trainable_state_dict']):raise ValueError('Loaded state differs from checkpoint')
        print(f'[原权重复核] {name}: epoch={c["epoch"]}',flush=True)
        values=predict(model,tokenizer,ds,decoded,indices,device,args.batch_size)
        results={p:group_report(ds,ids,values) for p,ids in partitions.items()}
        entry={'path':str(paths[name].resolve()),'sha256':hashes[name],'epoch':c['epoch'],
            'original_validation_loss':c['validation_loss'],'original_best_validation_loss':c['best_validation_loss'],
            'original_updates':sum(x['optimized_batches'] for x in c['history']), 'reports':results}
        history_row=next(x for x in c['history'] if x['epoch']==c['epoch'])
        measured=results['validation']['overall']
        mapping={'position_error_cm':'position_error_cm','rotation_error_deg':'rotation_error_deg',
            'gripper_accuracy':'gripper_accuracy','balanced_accuracy':'gripper_balanced_accuracy'}
        entry['historical_validation_metric_differences']={k:float(measured[k]-history_row[v]) for k,v in mapping.items()}
        if any(abs(x)>1e-4 for x in entry['historical_validation_metric_differences'].values()):raise ValueError('Historical validation metrics did not reproduce')
        if digest_state({k:p for k,p in model.named_parameters() if k in c['trainable_state_dict']})!=before:raise ValueError('Inference changed weights')
        if digest_state({k:p for k,p in model.named_parameters() if k.startswith(('vision_encoder.','text_encoder.'))})!=frozen_before:raise ValueError('Frozen encoders changed')
        entry['weights_unchanged_after_inference']=True;report['checkpoints'][name]=entry
        save_json(out/f'{name}_per_episode_metrics.json',results)
        torch.save({'indices':indices,'prediction':torch.tensor(np.stack([values[i]['prediction'] for i in indices])),
            'target':torch.tensor(np.stack([values[i]['target'] for i in indices])),
            'teacher_logits':torch.tensor(np.stack([values[i]['teacher_logits'] for i in indices])),
            'logits':torch.tensor(np.stack([values[i]['logits'] for i in indices])),
            'dataset_identity':reference['dataset_identity'],'checkpoint_sha256':hashes[name]},out/f'{name}_window_predictions.pt')
        save_json(out/'comparison.json',report)
        for p,r in results.items():
            m=r['overall'];print(f"{name} {p}: {m['position_error_cm']:.3f}cm/{m['rotation_error_deg']:.3f}deg; grip={m['gripper_accuracy']:.1%} balanced={m['balanced_accuracy']:.1%}; switches={m['open_to_closed_correct']}/{m['open_to_closed_pairs']},{m['closed_to_open_correct']}/{m['closed_to_open_pairs']}",flush=True)
    for name,path in paths.items():
        if hashlib.sha256(path.read_bytes()).hexdigest()!=hashes[name]:raise ValueError('Source file changed')
    report['completed']=True;report['verification']={'same_data_config_and_partitions':True,'all_371_audited_windows_evaluated':True,
        'historical_validation_metrics_reproduced_atol':1e-4,'source_files_unchanged':True,'all_loaded_parameters_unchanged':True}
    save_json(out/'comparison.json',report)
    columns=['checkpoint','epoch','partition','shard','record_index','windows','action_targets','position_error_cm',
        'zero_motion_position_error_cm','rotation_error_deg','identity_rotation_error_deg','gripper_accuracy','balanced_accuracy',
        'open_to_closed_correct','open_to_closed_pairs','closed_to_open_correct','closed_to_open_pairs']
    with (out/'per_episode_metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');writer.writeheader()
        for name,c in report['checkpoints'].items():
            for p,r in c['reports'].items():
                for e in r['episodes']:writer.writerow({'checkpoint':name,'epoch':c['epoch'],'partition':p,**e})
    print('[完成]',out.resolve(),flush=True)


if __name__=='__main__':main()

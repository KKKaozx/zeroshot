"""Validate the frozen-context head probe without any optimization or ID vectors."""
import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import CLIPTokenizer

from diagnose_bridge_modules import reconstruct, save_json, metrics, group_report
from models import RobotAdapterModel
from train import collate_batch, tokenise, set_seed


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',default='results/bridge_single_task_full_validation_v1/latest.pt')
    parser.add_argument('--probe',default='results/bridge_head_isolation_v1/frozen_context_diagnostic.pt')
    parser.add_argument('--output-json',default='results/bridge_head_isolation_v1/frozen_context_validation.json')
    parser.add_argument('--cache-dir',default='D:/ntu_related/dissertation/hf_cache')
    args=parser.parse_args()
    if Path(args.output_json).exists(): raise ValueError('Do not overwrite an existing validation report')
    torch.set_num_threads(4); set_seed(42)
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    probe=torch.load(args.probe,map_location='cpu',weights_only=False)
    if (probe['arm']!='frozen_context' or probe['dataset_identity']!=checkpoint['dataset_identity']
            or probe['config']!=checkpoint['config'] or set(probe['indices'])!=set(checkpoint['split_indices']['train'])):
        raise ValueError('Only matching train-only frozen-context probe can be validated')
    ds=reconstruct(checkpoint)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=RobotAdapterModel(checkpoint['config'],cache_dir=args.cache_dir).to(device)
    combined={**checkpoint['trainable_state_dict'],**probe['head_state_dict']}
    missing=model.load_state_dict(combined,strict=False)
    if missing.unexpected_keys or {n for n,p in model.named_parameters() if p.requires_grad}.intersection(missing.missing_keys):
        raise ValueError('Missing parameters')
    model.eval()
    tokenizer=CLIPTokenizer.from_pretrained(checkpoint['config']['model']['name'],cache_dir=args.cache_dir,local_files_only=True)
    indices=list(checkpoint['split_indices']['validation']); values={}
    for start in range(0,len(indices),8):
        ids=indices[start:start+8]
        texts,images,current,targets,masks=collate_batch([ds[i] for i in ids])
        text=tokenise(tokenizer,texts,device)
        context=model.get_context_vector(images.to(device),text['input_ids'],text.get('attention_mask'))
        prediction=model.sample(context,current.to(device)).cpu().numpy().astype(np.float64)
        teacher=model.predict_gripper_logits(context,targets[...,:7].to(device),current.to(device)).cpu().numpy()
        logits=model.predict_gripper_logits(context,torch.tensor(prediction[...,:7],dtype=torch.float32,device=device),current.to(device)).cpu().numpy()
        for k,i in enumerate(ids): values[i]={'prediction':prediction[k],'target':targets[k].numpy().astype(np.float64),'teacher_logits':teacher[k],'logits':logits[k]}
    report={'purpose':'held_out_validation_of_train_only_frozen_context_head_probe_not_robot_success',
        'checkpoint':args.checkpoint,'probe':args.probe,'optimized_on_validation':False,'test_targets_used':False,
        'diagnostic_steps':probe['steps'],'dataset_identity':probe['dataset_identity'],**group_report(ds,indices,values)}
    save_json(args.output_json,report)
    print(report['overall'],flush=True)


if __name__=='__main__': main()

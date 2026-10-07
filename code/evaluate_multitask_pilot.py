"""Evaluate both saved checkpoints once per seed, then group stored predictions."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
os.environ['HF_HUB_OFFLINE']='1'
import numpy as np
import torch
from transformers import CLIPTokenizer
from models import RobotAdapterModel
from train import bridge_plan_selection, bridge_plan_splits, collate_batch, set_seed, trainable_state_dict
from dataset import UnifiedRobotDataset, POSITION_SCALE_METERS


def metrics(prediction,target):
    p=torch.as_tensor(prediction); t=torch.as_tensor(target)
    pq=torch.nn.functional.normalize(p[...,3:7],dim=-1)
    tq=torch.nn.functional.normalize(t[...,3:7],dim=-1)
    true=t[...,7]>0; guess=p[...,7]>0
    recall=[float((guess[true==v]==v).float().mean()) for v in (False,True) if (true==v).any()]
    result=dict(windows=len(t),action_targets=len(t)*16,
        position_cm=float(torch.linalg.vector_norm(p[...,:3]-t[...,:3],dim=-1).mean())*POSITION_SCALE_METERS*100,
        rotation_deg=float((2*torch.acos((pq*tq).sum(-1).abs().clamp(max=1))*180/np.pi).mean()),
        gripper_accuracy=float((guess==true).float().mean()),balanced_accuracy=sum(recall)/len(recall),
        gripper_class_count=len(recall),open_targets=int(true.sum()),
        static_position_cm=float(torch.linalg.vector_norm(t[...,:3],dim=-1).mean())*POSITION_SCALE_METERS*100,
        static_rotation_deg=float((2*torch.acos(tq[...,3].abs().clamp(max=1))*180/np.pi).mean()),
        always_closed_accuracy=float((~true).float().mean()))
    for name,before,after in [('open_to_closed',True,False),('closed_to_open',False,True)]:
        mask=(true[:,:-1]==before)&(true[:,1:]==after)
        correct=(guess[:,:-1]==before)&(guess[:,1:]==after)
        result[name+'_pairs']=int(mask.sum())
        result[name+'_pair_accuracy']=float(correct[mask].float().mean()) if mask.any() else None
    return result


def main():
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    args=parser.parse_args()
    selected=bridge_plan_selection(args.manifest)
    dataset=UnifiedRobotDataset(data_dir=str(args.manifest.parent/'data'),chunk_size=16,stride=4,
        sources=['tfrecord'],min_trajectory_steps=17,exclude_path_parts=[],exclude_schemas=[],
        tfrecord_splits=['train'],bridge_gripper_policy='reverse_scan_valid_steps_v2',
        bridge_current_gripper='continuous',bridge_episode_selection=selected)
    splits=bridge_plan_splits(dataset)
    assert {k:len(v) for k,v in splits.items()}=={'train':207,'validation':40,'test':0}
    items=[dataset[i] for i in range(len(dataset))]
    targets=np.stack([x[3].numpy() for x in items])
    groups={}
    mapping={(r['shard'],r['record_index']):r for r in selected}
    for partition in ('train','validation'):
        groups[partition+'/overall']=splits[partition]
        task_groups=defaultdict(list); episode_groups=defaultdict(list)
        for i in splits[partition]:
            sample=dataset.samples[i]; key=Path(sample['file_path']).name,sample['record_index']
            task_groups[mapping[key]['instruction']].append(i)
            episode_groups[key[0]+'::'+str(key[1])].append(i)
        groups.update({partition+'/task/'+k:v for k,v in task_groups.items()})
        groups.update({partition+'/episode/'+k:v for k,v in episode_groups.items()})
    report=dict(trained_here=False,reserved_test_targets_read=False,training_seed=42,sampling_seeds=[0,1,2],
        checkpoint_selection='best by fixed-noise development loss; latest by fixed 20-epoch budget',
        transition_note='Overlapping window target pairs, not independent physical events',checkpoints={})
    for name in ('best','latest'):
        checkpoint=torch.load(args.run/(name+'.pt'),map_location='cpu',weights_only=False)
        assert checkpoint['split_indices']==splits
        model=RobotAdapterModel(checkpoint['config']).cuda().eval()
        assert set(checkpoint['trainable_state_dict'])==set(trainable_state_dict(model))
        model.load_state_dict(checkpoint['trainable_state_dict'],strict=False)
        tokenizer=CLIPTokenizer.from_pretrained(checkpoint['config']['model']['name'],local_files_only=True)
        predictions=np.empty((3,len(dataset),16,8),np.float32)
        with torch.no_grad():
            for start in range(0,len(items),2):
                language,images,current,_,_=collate_batch(items[start:start+2])
                tokens=tokenizer(language,padding=True,truncation=True,return_tensors='pt')
                context=model.get_context_vector(images.cuda(),tokens['input_ids'].cuda(),tokens['attention_mask'].cuda())
                for seed in range(3):
                    set_seed(seed*10000+start)
                    predictions[seed,start:start+len(language)]=model.sample(context,current.cuda()).cpu().numpy()
        assert np.isfinite(predictions).all()
        values={key:[metrics(predictions[s,indices],targets[indices]) for s in range(3)] for key,indices in groups.items()}
        report['checkpoints'][name]={'epoch':checkpoint['epoch'],'groups':values}
        np.savez_compressed(args.run/(name+'-predictions.npz'),predictions=predictions,targets=targets)
        print('FINAL',name,'train',values['train/overall'],'development',values['validation/overall'],flush=True)
        del model,checkpoint
        torch.cuda.empty_cache()
    (args.run/'full-evaluation.json').write_text(json.dumps(report,indent=2)+'\n')
    print('FIXED BUDGET PILOT AND FULL EVALUATION: COMPLETE',flush=True)


if __name__=='__main__': main()

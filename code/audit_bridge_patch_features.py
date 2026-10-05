"""Read-only CLS/patch diagnostics; no fitting, future-based neighbor selection or test data."""
import os
os.environ.setdefault('HF_HUB_OFFLINE','1')
import argparse
import hashlib
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image,ImageDraw,ImageFont
from transformers import CLIPTokenizer
from models import RobotAdapterModel
from dataset import CLIP_IMAGE_MEAN,CLIP_IMAGE_STD
from train import collate_batch,tokenise
from diagnose_bridge_modules import reconstruct,save_json,metrics
from audit_bridge_input_neighbors import cosine,summary


def select_neighbors(ds,indices,similarity,targets,current,queries,candidates,within=False):
    lookup={i:k for k,i in enumerate(indices)}; rows=[]
    for i in queries:
        if within:
            allowed=[j for j in candidates if ds.group_key(i)==ds.group_key(j)
                and abs(ds.samples[i]['start_index']-ds.samples[j]['start_index'])>=ds.chunk_size]
        else:
            allowed=[j for j in candidates if ds.group_key(i)!=ds.group_key(j)]
        if not allowed: continue
        scores=similarity[lookup[i],[lookup[j] for j in allowed]]
        rank=np.argsort(-scores,kind='stable'); j=allowed[int(rank[0])]
        m=metrics([targets[j]],[targets[i]])
        rows.append({'query_index':i,'neighbor_index':j,'similarity':float(scores[rank[0]]),
            'query_group':ds.group_key(i),'neighbor_group':ds.group_key(j),
            'query_start':ds.samples[i]['start_index'],'neighbor_start':ds.samples[j]['start_index'],
            'measurement_abs_gap':abs(current[i]-current[j]),
            'position_target_difference_cm':m['position_error_cm'],'rotation_target_difference_deg':m['rotation_error_deg'],
            'gripper_target_agreement':m['gripper_accuracy'],'query_target':targets[i].tolist(),'neighbor_target':targets[j].tolist(),
            'top5_input_neighbors':[{'index':allowed[int(k)],'similarity':float(scores[k])} for k in rank[:5]]})
    return rows


def heat_sheet(images,patches,indices,pair,path):
    lookup={i:k for k,i in enumerate(indices)}; i,j=pair
    a=cosine(patches[lookup[i]].float().numpy()); b=cosine(patches[lookup[j]].float().numpy())
    difference=np.clip(1-(a*b).sum(-1),0,2).reshape(16,16)
    # Fixed scale across pairs, not per-image min/max contrast.
    intensity=np.clip(difference/.5,0,1)
    heat=np.zeros((16,16,3),dtype=np.uint8); heat[:,:,0]=(255*intensity).astype('uint8')
    heat[:,:,2]=(255*(1-intensity)).astype('uint8')
    h=Image.fromarray(heat).resize((448,448),Image.Resampling.NEAREST)
    canvas=Image.new('RGB',(1380,570),'white'); draw=ImageDraw.Draw(canvas)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',17)
    draw.text((12,8),f'Actual CLIP inputs: index {i} vs {j}; final-layer aligned patch difference',font=font,fill='black')
    for k,index in enumerate((i,j)):
        im=Image.fromarray(images[lookup[index]]).resize((448,448))
        canvas.paste(im,(12+460*k,50))
        draw.text((12+460*k,27),f'Input {index}',font=font,fill='black')
    canvas.paste(h,(932,50)); draw.text((932,27),'Blue: 0; red: >=0.5 (1-cos)',font=font,fill='black')
    draw.text((12,510),'Not attention, object localization or causal saliency; each patch has global self-attention context.',font=font,fill='black')
    draw.text((12,537),f'Mean 1-cos={difference.mean():.4f}; max={difference.max():.4f}; corresponding grid cells only.',font=font,fill='black')
    canvas.save(path)
    return {'query_index':i,'neighbor_index':j,'mean_aligned_patch_distance':float(difference.mean()),
        'max_aligned_patch_distance':float(difference.max()),'patch_distance_grid':difference.tolist(),'image':str(path.resolve())}


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',default='results/bridge_single_task_full_validation_v1/latest.pt')
    parser.add_argument('--features',default='results/bridge_input_neighbors_v1/feature_cache.pt')
    parser.add_argument('--neighbors',default='results/bridge_input_neighbors_v1/neighbors.json')
    parser.add_argument('--output-dir',default='results/bridge_patch_features_v1')
    parser.add_argument('--cache-dir',default='D:/ntu_related/dissertation/hf_cache')
    args=parser.parse_args(); torch.set_num_threads(4)
    out=Path(args.output_dir)
    if out.exists() and any(out.iterdir()): raise ValueError('Use a fresh output directory')
    source=Path(args.checkpoint); source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
    ckpt=torch.load(source,map_location='cpu',weights_only=False)
    cache=torch.load(args.features,map_location='cpu',weights_only=False)
    if cache['dataset_identity']!=ckpt['dataset_identity'] or cache['checkpoint_sha256']!=source_hash:
        raise ValueError('Provenance mismatch')
    ds=reconstruct(ckpt); train=list(ckpt['split_indices']['train']); val=list(ckpt['split_indices']['validation']); indices=train+val
    if set(indices)!=set(cache['indices']): raise ValueError('Feature scope mismatch')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=RobotAdapterModel(ckpt['config'],cache_dir=args.cache_dir).to(device).eval()
    missing=model.load_state_dict(ckpt['trainable_state_dict'],strict=False)
    if missing.unexpected_keys or {n for n,p in model.named_parameters() if p.requires_grad}.intersection(missing.missing_keys):
        raise ValueError('Missing model weights')
    tokenizer=CLIPTokenizer.from_pretrained(ckpt['config']['model']['name'],cache_dir=args.cache_dir,local_files_only=True)
    out.mkdir(parents=True,exist_ok=True); descriptors=defaultdict(list); patch_tokens=[]; images=[]; targets={}; current={}; perturbation={}
    for start in range(0,len(indices),8):
        ids=indices[start:start+8]; texts,x,g,y,_=collate_batch([ds[i] for i in ids])
        hidden=model.vision_encoder(pixel_values=x.to(device)).last_hidden_state
        if hidden.shape[1]!=257: raise ValueError('Expected 16x16 patches')
        patch=hidden[:,1:].reshape(-1,16,16,hidden.shape[-1])
        grid=patch.reshape(-1,4,4,4,4,hidden.shape[-1]).mean((2,4))
        grid=F.normalize(grid,dim=-1).flatten(1)
        descriptors['clip_hidden_cls'].append(hidden[:,0].cpu())
        descriptors['patch_mean'].append(patch.mean((1,2)).cpu())
        descriptors['patch_grid4x4'].append(grid.cpu())
        # Fixed geometric region; NOT an automatically detected gripper/object mask.
        descriptors['patch_lower_center'].append(patch[:,8:16,4:12].mean((1,2)).cpu())
        patch_tokens.append(hidden[:,1:].cpu().half())
        rgb=((x*CLIP_IMAGE_STD+CLIP_IMAGE_MEAN).clamp(0,1)*255).round().byte().permute(0,2,3,1).numpy()
        images.extend(rgb)
        for k,i in enumerate(ids): targets[i]=y[k].numpy().astype(np.float64); current[i]=float(g[k])
        if start==0:
            text=tokenise(tokenizer,texts,device)
            language=model.text_encoder(input_ids=text['input_ids'],attention_mask=text['attention_mask']).last_hidden_state
            c=model.adapter(hidden,language,text['attention_mask'])
            changed=hidden.clone(); changed[:,1:]=0
            zero=model.adapter(changed,language,text['attention_mask'])
            shuffled=hidden.clone(); shuffled[:,1:]=hidden.flip(0)[:,1:]
            other=model.adapter(shuffled,language,text['attention_mask'])
            changed_cls=hidden.clone(); changed_cls[:,0]=0
            altered=model.adapter(changed_cls,language,text['attention_mask'])
            perturbation={'scope':'post-CLIP tokens, first 8 training windows; CLIP CLS held fixed for patch tests',
                'zero_patches_max_context_change':float((c-zero).abs().max()),
                'swap_patches_max_context_change':float((c-other).abs().max()),
                'zero_cls_max_context_change':float((c-altered).abs().max())}
        if start%64==0: print(f'[patch诊断] 特征 {min(start+8,len(indices))}/{len(indices)}',flush=True)
    descriptors={k:torch.cat(v).numpy() for k,v in descriptors.items()}
    cachemap={i:k for k,i in enumerate(cache['indices'])}
    expected=cache['clip_hidden_cls'][[cachemap[i] for i in indices]].numpy()
    cache_error=float(np.abs(descriptors['clip_hidden_cls']-expected).max())
    if cache_error>1e-4: raise ValueError('Recomputed CLS differs from previous audit')
    descriptors['adapter_context']=cache['adapter_context'][[cachemap[i] for i in indices]].numpy()
    report={'purpose':'read_only_patch_neighbor_diagnostic_not_policy_success','trained':False,'test_targets_used':False,
        'checkpoint_sha256':source_hash,'dataset_identity':ckpt['dataset_identity'],'indices':indices,
        'feature_definitions':{'patch_mean':'mean of all 256 raw final-layer patch tokens',
            'patch_grid4x4':'average each 4x4 block of raw patches; normalize each of 16 cells; concatenate; aligned-cell cosine',
            'patch_lower_center':'mean raw patches in fixed rows 8:16 and columns 4:12; no object annotations'},
        'post_clip_token_perturbation':perturbation,'recomputed_cls_max_error':cache_error,'feature_spaces':{},
        'limits':['Final-layer patch tokens already include global self-attention context.',
            'Aligned image cells are not physical or object correspondences across scenes.',
            'No object masks or location labels; fixed regions cannot prove gripper or object information retention.',
            'Cosine scales differ across descriptors; compare selected targets rather than raw scores.',
            'Three validation episodes and overlapping windows; not a statistical or causal root-cause proof.']}
    for name,values in descriptors.items():
        normalized=cosine(values).astype(np.float32); similarity=normalized@normalized.T
        report['feature_spaces'][name]={}
        for scope,queries,within in [('validation_to_train',val,False),('train_to_other_episode',train,False),('train_within_episode_nonoverlap',train,True)]:
            rows=select_neighbors(ds,indices,similarity,targets,current,queries,train,within)
            groups=defaultdict(list)
            for row in rows: groups[row['query_group']].append(row)
            report['feature_spaces'][name][scope]={'overall':summary(rows),'rows':rows,
                'per_query_episode':[{'group_key':k,**summary(v)} for k,v in groups.items()]}
        print('[patch诊断]',name,'validation',report['feature_spaces'][name]['validation_to_train']['overall']['position_error_cm'],flush=True)
    # Pair selection uses previous input-only CLS neighbors and feature similarity, never target discrepancies.
    import json
    prior_report=json.loads(Path(args.neighbors).read_text(encoding='utf-8'))
    if prior_report['dataset_identity']!=ckpt['dataset_identity'] or prior_report['source_sha256']!=source_hash:
        raise ValueError('Prior neighbor provenance mismatch')
    prior=prior_report['feature_spaces']['clip_hidden_cls']
    pairrows=[]
    for scope in ['validation_to_train','train_within_episode_nonoverlap']:
        groups=defaultdict(list)
        for row in prior[scope]['rows']: groups[row['query_group']].append(row)
        for group,rows in groups.items(): pairrows.append(max(rows,key=lambda r:r['similarity']))
    patches=torch.cat(patch_tokens); maps=out/'pairs'; maps.mkdir(); heatmaps=[]
    for row in pairrows:
        pair=(row['query_index'],row['neighbor_index'])
        heatmaps.append(heat_sheet(images,patches,indices,pair,maps/f'patch_difference_{pair[0]}_{pair[1]}.png'))
    report['input_selected_pair_maps']=heatmaps
    checks={}
    for name,spaces in report['feature_spaces'].items():
        checks[name]=all(r['neighbor_index'] in train and r['query_group']!=r['neighbor_group']
            for scope in ['validation_to_train','train_to_other_episode'] for r in spaces[scope]['rows']) and all(
            r['query_group']==r['neighbor_group'] and abs(r['query_start']-r['neighbor_start'])>=ds.chunk_size
            for r in spaces['train_within_episode_nonoverlap']['rows'])
    if not all(checks.values()): raise ValueError('Episode/overlap exclusions failed')
    if hashlib.sha256(source.read_bytes()).hexdigest()!=source_hash: raise ValueError('Source changed')
    report['verification']={'episode_and_overlap_exclusions':checks,'source_checkpoint_unchanged':True,
        'evaluated_windows':len(indices),'validation_windows':len(val),'all_descriptors_finite':all(np.isfinite(v).all() for v in descriptors.values())}
    if not report['verification']['all_descriptors_finite']: raise ValueError('Nonfinite descriptors')
    save_json(out/'patch_neighbors.json',report)
    torch.save({'indices':indices,'descriptors':{k:torch.from_numpy(v) for k,v in descriptors.items()},
        'dataset_identity':ckpt['dataset_identity'],'checkpoint_sha256':source_hash},out/'patch_descriptor_cache.pt')
    print('[完成]',out.resolve(),flush=True)


if __name__=='__main__': main()

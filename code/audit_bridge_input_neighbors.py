"""Read-only feature/action-neighborhood audit. No training or test targets.

Select neighbors using inputs only. Compare complete future target blocks after
selection. Near features with different targets are diagnostic evidence, not
proof of irreducible ambiguity. Cosine similarity is not a probability.
"""
import argparse
import hashlib
import io
import json
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault('HF_HUB_OFFLINE','1')
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import CLIPTokenizer

from diagnose_bridge_modules import reconstruct, save_json, metrics, group_report
from models import RobotAdapterModel
from train import collate_batch, tokenise, set_seed


def cosine(values):
    a=np.asarray(values,dtype=np.float64)
    return a/np.maximum(np.linalg.norm(a,axis=-1,keepdims=True),1e-12)


def summary(rows):
    if not rows: return None
    m=metrics(np.array([r['neighbor_target'] for r in rows]),np.array([r['query_target'] for r in rows]))
    m['similarity_quantiles']=np.quantile([r['similarity'] for r in rows],[0,.25,.5,.75,1]).tolist()
    m['current_measurement_abs_gap_median']=float(np.median([r['measurement_abs_gap'] for r in rows]))
    return m


def neighbors(ds,query_indices,candidates,features,targets,current,kind):
    ids=list(features)
    normalized=cosine([features[i] for i in ids]); rowmap={i:k for k,i in enumerate(ids)}
    rows=[]
    for i in query_indices:
        if kind=='within_episode_nonoverlap':
            allowed=[j for j in candidates if ds.group_key(i)==ds.group_key(j)
                and abs(ds.samples[i]['start_index']-ds.samples[j]['start_index'])>=ds.chunk_size]
        else: allowed=[j for j in candidates if ds.group_key(i)!=ds.group_key(j)]
        if not allowed: continue
        similarity=normalized[[rowmap[j] for j in allowed]] @ normalized[rowmap[i]]
        rank=np.argsort(-similarity,kind='stable'); j=allowed[int(rank[0])]
        diff=metrics([targets[j]],[targets[i]])
        top5=[{'index':allowed[int(k)],'similarity':float(similarity[k])} for k in rank[:5]]
        rows.append({'query_index':i,'neighbor_index':j,'similarity':float(similarity[rank[0]]),
            'query_group':ds.group_key(i),'neighbor_group':ds.group_key(j),
            'query_start':ds.samples[i]['start_index'],'neighbor_start':ds.samples[j]['start_index'],
            'measurement_abs_gap':float(abs(current[i]-current[j])),
            'position_target_difference_cm':diff['position_error_cm'],'rotation_target_difference_deg':diff['rotation_error_deg'],
            'gripper_target_agreement':diff['gripper_accuracy'],'query_target':targets[i].tolist(),
            'neighbor_target':targets[j].tolist(),'top5_input_neighbors':top5})
    return rows


def pair_sheet(ds,row,path,scope):
    images=[]
    for key in ('query_index','neighbor_index'):
        sample=ds.samples[row[key]]
        features=ds._load_tfrecord_example(sample['file_path'],sample['record_index']).features.feature
        image=Image.open(io.BytesIO(features['steps/observation/image_0'].bytes_list.value[sample['start_index']])).convert('RGB')
        image.thumbnail((540,410)); images.append(image)
    canvas=Image.new('RGB',(1140,540),'white'); draw=ImageDraw.Draw(canvas)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16)
    draw.text((12,8),f"{scope}: cosine={row['similarity']:.6f} (not probability)",font=font,fill='black')
    for k,(key,image) in enumerate(zip(('query_index','neighbor_index'),images)):
        i=row[key]; s=ds.samples[i]; x=12+k*570
        draw.text((x,37),f"{'query' if k==0 else 'train neighbor'} index={i}, t={s['start_index']}",font=font,fill='black')
        draw.text((x,60),f"{Path(s['file_path']).name[-25:]} / record {s['record_index']}",font=font,fill='black')
        canvas.paste(image,(x,90))
    draw.text((12,505),f"Target block difference: {row['position_target_difference_cm']:.2f} cm / {row['rotation_target_difference_deg']:.2f} deg; gripper agreement={row['gripper_target_agreement']:.1%}",font=font,fill='black')
    canvas.save(path)


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',default='results/bridge_single_task_full_validation_v1/latest.pt')
    parser.add_argument('--output-dir',default='results/bridge_input_neighbors_v1')
    parser.add_argument('--audit',default='results/bridge_module_diagnostic_v3/data_contract_audit.json')
    parser.add_argument('--cache-dir',default='D:/ntu_related/dissertation/hf_cache')
    args=parser.parse_args()
    output=Path(args.output_dir)
    if output.exists() and any(output.iterdir()): raise ValueError('Preserve existing report; use a new directory')
    torch.set_num_threads(4); set_seed(42)
    source=Path(args.checkpoint); source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
    checkpoint=torch.load(source,map_location='cpu',weights_only=False)
    ds=reconstruct(checkpoint)
    audit=json.loads(Path(args.audit).read_text(encoding='utf-8'))
    if audit['dataset_identity']!=checkpoint['dataset_identity'] or audit['windows']!=audit['passed_windows']:
        raise ValueError('Audit identity mismatch')
    partitions={p:list(checkpoint['split_indices'][p]) for p in ('train','validation')}
    indices=partitions['train']+partitions['validation']; output.mkdir(parents=True,exist_ok=True)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=RobotAdapterModel(checkpoint['config'],cache_dir=args.cache_dir).to(device).eval()
    missing=model.load_state_dict(checkpoint['trainable_state_dict'],strict=False)
    if missing.unexpected_keys or {n for n,p in model.named_parameters() if p.requires_grad}.intersection(missing.missing_keys):
        raise ValueError('Missing model parameters')
    tokenizer=CLIPTokenizer.from_pretrained(checkpoint['config']['model']['name'],cache_dir=args.cache_dir,local_files_only=True)
    visual_features={}; contexts={}; targets={}; current={}; instructions={}; states={}
    for start in range(0,len(indices),8):
        ids=indices[start:start+8]
        texts,images,grip,actions,masks=collate_batch([ds[i] for i in ids])
        text=tokenise(tokenizer,texts,device)
        visual=model.vision_encoder(pixel_values=images.to(device)).last_hidden_state
        language=model.text_encoder(input_ids=text['input_ids'],attention_mask=text['attention_mask']).last_hidden_state
        c=model.adapter(visual,language,text['attention_mask'])
        for k,i in enumerate(ids):
            visual_features[i]=visual[k,0].cpu().numpy(); contexts[i]=c[k].cpu().numpy()
            targets[i]=actions[k].numpy().astype(np.float64); current[i]=float(grip[k].item()); instructions[i]=texts[k]
        if start%64==0: print(f'[输入诊断] 提取特征 {min(start+8,len(indices))}/{len(indices)}',flush=True)
    report={'purpose':'input_only_neighbor_selection_then_action_difference_not_success',
        'checkpoint':args.checkpoint,'dataset_identity':checkpoint['dataset_identity'],'source_sha256':source_hash,
        'trained':False,'test_targets_used':False,'instruction_count':len(set(instructions.values())),
        'feature_spaces':{},'limits':['CLIP hidden CLS cosine is not a semantic match probability.',
            'Different targets for near features do not prove identical images or missing information.',
            'Neighbor transfer is an offline diagnostic, not a learned policy or a calibrated controller.',
            'Window statistics overlap in time; validation contains only 3 independent demonstrations.']}
    all_rows={}
    for space,features in (('clip_hidden_cls',visual_features),('adapter_context',contexts)):
        report['feature_spaces'][space]={}
        for name,queries,kind in (('validation_to_train',partitions['validation'],'different_episode'),
            ('train_to_other_episode',partitions['train'],'different_episode'),
            ('train_within_episode_nonoverlap',partitions['train'],'within_episode_nonoverlap')):
            rows=neighbors(ds,queries,partitions['train'],features,targets,current,kind)
            all_rows[space,name]=rows
            groups=defaultdict(list)
            for row in rows: groups[row['query_group']].append(row)
            report['feature_spaces'][space][name]={'overall':summary(rows),
                'per_query_episode':[{ 'group_key':key,**summary(v)} for key,v in groups.items()],
                'rows':rows}
    # Raw current EEF pose is deliberately NOT an input to the present pose head.
    # Describe whether nearby-feature pairs differ in that currently omitted state.
    raw_cache={}
    for i in indices:
        s=ds.samples[i]; key=(s['file_path'],s['record_index'])
        if key not in raw_cache:
            f=ds._load_tfrecord_example(*key).features.feature
            raw_cache[key]=np.asarray(f['steps/observation/state'].float_list.value,dtype=np.float64).reshape(-1,7)
        states[i]=raw_cache[key][s['start_index']]
    from diagnose_bridge_modules import matrices,matrix_angles
    import pybullet as bullet
    for key,rows in all_rows.items():
        for row in rows:
            a,b=states[row['query_index']],states[row['neighbor_index']]
            row['input_eef_position_gap_cm']=float(np.linalg.norm(a[:3]-b[:3])*100)
            qa=bullet.getQuaternionFromEuler(a[3:6].tolist()); qb=bullet.getQuaternionFromEuler(b[3:6].tolist())
            ra,rb=matrices([qa,qb])
            row['input_eef_rotation_gap_deg']=float(matrix_angles((ra.T@rb)[None])[0])
    report['input_pose_gap_summary']={space:{scope:{
        'median_input_eef_position_gap_cm':float(np.median([x['input_eef_position_gap_cm'] for x in v['rows']])),
        'median_input_eef_rotation_gap_deg':float(np.median([x['input_eef_rotation_gap_deg'] for x in v['rows']]))}
        for scope,v in s.items()} for space,s in report['feature_spaces'].items()}
    exclusion={space:all(x['query_group']!=x['neighbor_group'] for x in s['validation_to_train']['rows']+s['train_to_other_episode']['rows'])
        and all(abs(x['query_start']-x['neighbor_start'])>=ds.chunk_size and x['query_group']==x['neighbor_group']
            for x in s['train_within_episode_nonoverlap']['rows']) for space,s in report['feature_spaces'].items()}
    if not all(exclusion.values()): raise ValueError('Neighbor exclusion contract violated')
    hashes=defaultdict(list)
    for row in audit['rows']: hashes[row['input_image_sha256']].append(row['index'])
    report['verification']={'episode_exclusion_checks':exclusion,'evaluated_windows':len(indices),
        'validation_windows':len(partitions['validation']),'encoded_image_duplicate_groups':sum(len(v)>1 for v in hashes.values()),
        'duplicate_note':'Only checks byte-identical encoded images; absence does not prove pixel uniqueness or no ambiguity.'}
    save_json(output/'neighbors.json',report)
    torch.save({'indices':indices,'clip_hidden_cls':torch.tensor(np.array([visual_features[i] for i in indices])),
        'adapter_context':torch.tensor(np.array([contexts[i] for i in indices])),
        'dataset_identity':checkpoint['dataset_identity'],'checkpoint_sha256':source_hash},output/'feature_cache.pt')
    plots=output/'pairs'; plots.mkdir()
    for space in ('clip_hidden_cls','adapter_context'):
        for row in all_rows[space,'validation_to_train']: pair_sheet(ds,row,plots/f'{space}_validation_{row["query_index"]}.png',space)
        # One selected pair per training episode; no target-based ranking.
        groups=defaultdict(list)
        for row in all_rows[space,'train_within_episode_nonoverlap']: groups[row['query_group']].append(row)
        for v in groups.values():
            row=max(v,key=lambda r:r['similarity'])
            pair_sheet(ds,row,plots/f'{space}_within_{row["query_index"]}.png',space)
    if hashlib.sha256(source.read_bytes()).hexdigest()!=source_hash: raise ValueError('Source checkpoint changed')
    print('[完成] 不训练的输入近邻诊断',output.resolve(),flush=True)
    for space,r in report['feature_spaces'].items():
        for scope,s in r.items():
            m=s['overall']; print(space,scope,{k:m[k] for k in ('position_error_cm','rotation_error_deg','gripper_accuracy','similarity_quantiles')},flush=True)


if __name__=='__main__': main()

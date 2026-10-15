"""Read-only range and nearest-image diagnostics for the frozen 201663 run."""
import csv
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np

os.environ.update(USE_TF='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT/'cluster_packages/bridge_growth128_v1'
DATA = ROOT/'reports/bridge-growth128-fit-201663'
OUT = ROOT/'reports/bridge_growth128_generalization_201663_20261013'
CLIP = Path('D:/ntu_related/dissertation/hf_cache/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41')
sys.path.insert(0,str(ROOT/'scripts'))


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def unit(a): return a/np.maximum(np.linalg.norm(a,axis=-1,keepdims=True),1e-12)
def mean_pose(a):
    result = a.mean(0)
    q = a[...,3:7].copy(); q *= np.where(q[...,3:4]<0,-1,1)
    result[...,3:7] = unit(q.mean(0))
    return result
def write_csv(name,rows):
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def inputs():
    manifest = json.loads((PACK/'manifest.json').read_text())
    assert json.loads((ROOT/'reports/bridge_growth128_analysis_201663_20261013/analysis.json').read_text())['verified']
    paths=[ROOT/'reports/bridge_batch_input_reference_20261012/arrays.npz',
           ROOT/'cluster_packages/bridge_growth64_v1/added_reference/arrays.npz',PACK/'new_reference/arrays.npz']
    refs=[]
    for path in paths:
        with np.load(path,allow_pickle=False) as z: refs.append({k:z[k].copy() for k in z.files})
    arrays={k:np.concatenate([refs[0][k][:32],refs[1][k],refs[2][k],refs[0][k][32:]],axis=0) for k in refs[0]}
    assert arrays['images'].shape==(168,3,224,224)
    with np.load(DATA/'reduced_u8192_train.npz') as z:
        assert np.max(np.abs(arrays['targets'][:128]-z['targets']))<=1e-6
    with np.load(DATA/'reduced_u8192_development.npz') as z:
        assert np.max(np.abs(arrays['targets'][128:]-z['targets']))<=1e-6
    return manifest,arrays,paths


def encode():
    import torch
    from transformers import CLIPModel
    manifest,arrays,paths=inputs(); started=time.monotonic()
    assert torch.cuda.is_available()
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    model=CLIPModel.from_pretrained(str(CLIP),local_files_only=True).eval().cuda()
    descriptors=[]
    with torch.inference_mode():
        for i in range(0,168,2):
            v=model.vision_model(pixel_values=torch.from_numpy(arrays['images'][i:i+2]).cuda()).last_hidden_state
            descriptor=torch.cat([torch.nn.functional.normalize(v[:,0],dim=-1),
                                  torch.nn.functional.normalize(v[:,1:].mean(1),dim=-1)],dim=-1)
            descriptors.append(descriptor.cpu().numpy())
            if (i+2)%32==0: print('FROZEN_CLIP_IMAGES',i+2,'/168',flush=True)
    features=np.concatenate(descriptors);assert features.shape==(168,2048) and np.isfinite(features).all()
    OUT.mkdir(exist_ok=True)
    np.savez_compressed(OUT/'frozen_visual_descriptors.npz',features=features)
    weights=CLIP/'model.safetensors'; assert weights.is_file()
    metadata=dict(stage='frozen_visual_encoding_only',trained=False,weights_modified=False,
        weights_path=str(weights),weights_sha256=sha(weights),torch_version=torch.__version__,
        feature_definition='Concatenate separately L2-normalized last_hidden_state CLS and mean patch tokens; comparison descriptor, not full token sequence',
        feature_shape=list(features.shape),manifest_sha256=sha(PACK/'manifest.json'),
        reference_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths},
        processed_image_sha256=[hashlib.sha256(a.tobytes()).hexdigest() for a in arrays['images']],
        features_sha256=sha(OUT/'frozen_visual_descriptors.npz'),elapsed_seconds=time.monotonic()-started,
        peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,model_downloaded=False)
    (OUT/'encoding.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps({k:metadata[k] for k in ['stage','elapsed_seconds','peak_reserved_gib','feature_shape']}))


def analyze():
    from analyze_bridge_training_trace_200039 import errors
    manifest,arrays,paths=inputs(); rows=manifest['train']+manifest['development']
    meta=json.loads((OUT/'encoding.json').read_text())
    assert meta['manifest_sha256']==sha(PACK/'manifest.json')
    assert meta['features_sha256']==sha(OUT/'frozen_visual_descriptors.npz')
    assert meta['reference_sha256']=={str(p.relative_to(ROOT)):sha(p) for p in paths}
    assert meta['processed_image_sha256']==[hashlib.sha256(a.tobytes()).hexdigest() for a in arrays['images']]
    with np.load(OUT/'frozen_visual_descriptors.npz') as z: visual=unit(z['features'].astype(float))
    with np.load(DATA/'reduced_u8192_train.npz') as z: train={k:z[k].copy() for k in z.files}
    with np.load(DATA/'reduced_u8192_development.npz') as z: dev={k:z[k].copy() for k in z.files}
    truth=np.concatenate([train['targets'],dev['targets']]); context=unit(np.concatenate([train['contexts'],dev['contexts']]).astype(float))
    assert len(rows)==len(truth)==len(visual)==168
    tasks=np.array([r['task'] for r in rows]); episodes=np.array([r['shard']+'::'+str(r['record_index']) for r in rows])
    predictions=np.concatenate([train['predictions'],dev['predictions']],axis=1)
    model_p,model_r=errors(predictions,truth)
    static=np.zeros_like(truth);static[...,6]=1
    static_p,static_r=errors(static[None],truth)
    indexed = json.loads((ROOT/'reports/bridge-xyz-eval-197349.json').read_text())['window_selection']
    pool=defaultdict(list)
    for row in indexed: pool[(row['shard'],row['record_index'])].append(row['start_index'])
    phases=[]
    for row in rows:
        starts=pool[(row['shard'],row['record_index'])];t=row['start_index']
        assert t in starts and min(starts)<max(starts)
        phases.append('first' if t==min(starts) else 'last' if t==max(starts) else 'middle')
    ranges=[]; window_rows=[]; nearest_predictions={'frozen_visual':[],'learned_context':[],'task_mean':[],'truth_position_oracle':[]}
    nearest_indices={k:[] for k in ('frozen_visual','learned_context','truth_position_oracle')}
    similarities={k:[] for k in ('frozen_visual','learned_context')}
    for i,row in enumerate(rows):
        candidates=np.flatnonzero((tasks[:128]==tasks[i]) & (episodes[:128]!=episodes[i]))
        assert len(candidates)>0
        for key,features in [('frozen_visual',visual),('learned_context',context)]:
            sims=features[candidates]@features[i];j=int(candidates[sims.argmax()])
            nearest_indices[key].append(j);similarities[key].append(float(sims.max()))
            nearest_predictions[key].append(truth[j])
        gap=np.linalg.norm(truth[candidates,:,:3]-truth[i,None,:,:3],axis=-1).mean(-1)
        j=int(candidates[gap.argmin()]);nearest_indices['truth_position_oracle'].append(j)
        nearest_predictions['truth_position_oracle'].append(truth[j])
        nearest_predictions['task_mean'].append(mean_pose(truth[candidates]))
    distance_errors={k:errors(np.asarray(v)[None],truth) for k,v in nearest_predictions.items()}
    for task in sorted(set(tasks)):
        ti=np.flatnonzero(tasks[:128]==task);di=np.flatnonzero(tasks[128:]==task)+128
        txyz=truth[ti,:,:3].astype(float); dxyz=truth[di,:,:3].astype(float)
        lower=txyz.min((0,1));upper=txyz.max((0,1));outside=(dxyz<lower-1e-6)|(dxyz>upper+1e-6)
        # A marginal box does not establish conditional or joint trajectory coverage.
        dq=unit(truth[di,:,3:7].astype(float));tq=unit(truth[ti,:,3:7].astype(float))
        ta=np.degrees(2*np.arccos(np.clip(np.abs(tq[...,3]),0,1)))
        da=np.degrees(2*np.arccos(np.clip(np.abs(dq[...,3]),0,1)))
        ranges.append(dict(task=task,train_windows=len(ti),development_windows=len(di),
            train_xyz_min_cm=(lower*10).tolist(),train_xyz_max_cm=(upper*10).tolist(),
            development_xyz_min_cm=(dxyz.min((0,1))*10).tolist(),development_xyz_max_cm=(dxyz.max((0,1))*10).tolist(),
            targets_outside_same_task_xyz_box_fraction=float(outside.any(-1).mean()),
            per_axis_outside_fraction=outside.mean((0,1)).tolist(),
            train_relative_rotation_deg_range=[float(ta.min()),float(ta.max())],
            development_relative_rotation_deg_range=[float(da.min()),float(da.max())],
            development_relative_rotation_above_train_max_fraction=float((da>ta.max()+1e-4).mean())))
    for i,row in enumerate(rows):
        j=nearest_indices['frozen_visual'][i]
        result=dict(split='train_leave_same_episode_out' if i<128 else 'development',window=i,
            task=tasks[i],episode=episodes[i],start_index=row['start_index'],
            indexed_start_phase=phases[i],
            model_position_cm=float(model_p[:,i].mean()),model_rotation_deg=float(model_r[:,i].mean()),
            static_position_cm=float(static_p[:,i].mean()),static_rotation_deg=float(static_r[:,i].mean()),
            frozen_visual_cosine=similarities['frozen_visual'][i],learned_context_cosine=similarities['learned_context'][i],
            nearest_visual_train_window=j,nearest_visual_episode=episodes[j],nearest_visual_start_index=rows[j]['start_index'])
        for key,(p,r) in distance_errors.items():
            result[key+'_position_cm']=float(p[:,i].mean());result[key+'_rotation_deg']=float(r[:,i].mean())
        window_rows.append(result)
    episode_rows=[]
    metric_names=[k for k in window_rows[0] if k.endswith(('_cm','_deg','_cosine'))]
    for split in ('train_leave_same_episode_out','development'):
        for ep in sorted({r['episode'] for r in window_rows if r['split']==split}):
            group=[r for r in window_rows if r['split']==split and r['episode']==ep]
            episode_rows.append(dict(split=split,episode=ep,task=group[0]['task'],windows=len(group),
                **{k:float(np.mean([r[k] for r in group])) for k in metric_names}))
    task_rows=[]
    for task in sorted(set(tasks)):
        group=[r for r in episode_rows if r['split']=='development' and r['task']==task]
        task_rows.append(dict(task=task,episodes=len(group),**{k:float(np.mean([r[k] for r in group])) for k in metric_names}))
    write_csv('windows.csv',window_rows);write_csv('episodes.csv',episode_rows);write_csv('development_tasks.csv',task_rows)
    phase_rows=[]
    for split,lo,hi in [('train',0,128),('development',128,168)]:
        for phase in ('first','middle','last'):
            ix=[i for i in range(lo,hi) if phases[i]==phase]
            phase_rows.append(dict(split=split,phase=phase,windows=len(ix),episodes=len(set(episodes[ix])),
                model_position_cm=float(model_p[:,ix].mean()),model_rotation_deg=float(model_r[:,ix].mean()),
                static_position_cm=float(static_p[:,ix].mean()),static_rotation_deg=float(static_r[:,ix].mean())))
    write_csv('indexed_phase_coverage.csv',phase_rows)
    # Contact sheet chosen by input similarity only, with one row per episode.
    selected=[];seen=set()
    for i in sorted(range(128,168),key=lambda i:similarities['frozen_visual'][i],reverse=True):
        if episodes[i] not in seen: selected.append(i);seen.add(episodes[i])
        if len(selected)==4:break
    import matplotlib.pyplot as plt
    mean=np.array([.48145466,.4578275,.40821073])[:,None,None];std=np.array([.26862954,.26130258,.27577711])[:,None,None]
    fig,axes=plt.subplots(4,3,figsize=(12,13),gridspec_kw={'width_ratios':[1,1,1.4]})
    fig.subplots_adjust(left=.025,right=.985,bottom=.05,top=.955,hspace=.50,wspace=.36)
    for line,i in enumerate(selected):
        j=nearest_indices['frozen_visual'][i]
        for col,k in enumerate((i,j)):
            img=np.clip(arrays['images'][k]*std+mean,0,1).transpose(1,2,0)
            axes[line,col].imshow(img);axes[line,col].axis('off')
            axes[line,col].set_title(('Development' if col==0 else 'Nearest train')+' '+rows[k]['shard'].split('-')[2]+'::'+str(rows[k]['record_index'])+' t'+str(rows[k]['start_index'])+'\n'+tasks[k],fontsize=9)
        ax=axes[line,2]
        for trajectory,label,color in [(truth[i],'Dev truth','#1565c0'),(truth[j],'Nearest train truth','#e0641d'),(predictions[:,i].mean(0),'Model mean','#2e7d32')]:
            ax.plot(trajectory[:,0]*10,trajectory[:,1]*10,label=label,color=color)
            ax.scatter(trajectory[-1,0]*10,trajectory[-1,1]*10,color=color,s=16)
        ax.set_title(f"Image cosine {similarities['frozen_visual'][i]:.3f}; target gap {distance_errors['frozen_visual'][0][0,i].mean():.2f}cm",fontsize=9)
        ax.set_xlabel('Relative X (cm)');ax.set_ylabel('Relative Y (cm)');ax.legend(fontsize=7);ax.grid(alpha=.2)
    fig.savefig(OUT/'nearest_visual_cases.png',dpi=130);plt.close(fig)
    result=dict(stage='read_only_exploratory_generalization_diagnosis',trained=False,new_inference_of_policy=False,
        new_frozen_clip_encoding=True,unseen_tasks=sorted(set(tasks[128:])-set(tasks[:128])),
        train_episodes=64,development_episodes=10,train_windows=128,development_windows=40,
        ranges=ranges,indexed_phase_coverage=phase_rows,
        development_target_xyz_outside_same_task_box_fraction=float(sum(r['targets_outside_same_task_xyz_box_fraction']*r['development_windows'] for r in ranges)/40),
        episode_weighted_summaries={split:{k:float(np.mean([r[k] for r in episode_rows if r['split']==split])) for k in metric_names} for split in ('train_leave_same_episode_out','development')},
        contact_sheet_selection='Four highest frozen-visual similarities, unique development episodes; selected without prediction error',
        contact_sheet_windows=selected,source_sha256=sha(Path(__file__)),
        limitations=['Development data reused for diagnosis, not untouched test','Overlapping windows are not independent; summaries use episode means',
            'Nearest-neighbor retrieval uses same task, excludes same episode; a diagnostic, not a trained policy',
            'Truth-position oracle chooses by true targets, not deployable; position-nearest does not mean rotation-nearest',
            'Pooled frozen CLIP descriptor is not full tokens or a metric of robot state equality',
            'Marginal xyz/rotation bounds cannot establish joint/conditional coverage','Learned-context similarity is post-training and may reflect memorization',
            'First/middle/last refer to indexed eligible input starts, not semantic task phases; groups differ in episode composition',
            'No causal fault localization from correlations or nearest neighbors','No gripper or closed-loop success evaluated'])
    (OUT/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


def poses():
    # Decode only the four selected input pairs, verifying their raw identities.
    os.environ.update(CUDA_VISIBLE_DEVICES='-1',TF_CPP_MIN_LOG_LEVEL='3')
    sys.path.insert(0,str(PACK))
    from dataset import UnifiedRobotDataset, euler_xyz_to_quaternion, rotate_vector
    from bridge_input_contract import raw_identity
    manifest,arrays,_=inputs();rows=manifest['train']+manifest['development']
    analysis=json.loads((OUT/'analysis.json').read_text())
    window_rows=list(csv.DictReader((OUT/'windows.csv').open(encoding='utf-8-sig')))
    report=json.loads((DATA/'report.json').read_text());cache={};pairs=[]
    def raw_state(i):
        row=rows[i];key=(row['shard'],row['record_index'])
        if key not in cache:
            cache[key]=UnifiedRobotDataset._load_tfrecord_example(str(ROOT/'training_cache/exports/bridge_expansion_training_v1/data'/row['shard']),row['record_index'])
        example=cache[key]
        audit_index=i if i<32 else i+40 if i<128 else i-96
        assert raw_identity(example,row['start_index'])==report['data_checks'][audit_index]['raw_identity']
        fields=example.features.feature;n=len(fields['steps/language_instruction'].bytes_list.value)
        states=np.asarray(fields['steps/observation/state'].float_list.value,dtype=np.float32).reshape(n,7)
        t=row['start_index'];ref=states[t];q=euler_xyz_to_quaternion(ref[3:6])
        rotation=np.column_stack([rotate_vector(v,q) for v in np.eye(3,dtype=np.float32)])
        ix=np.minimum(np.arange(t+1,t+17),n-1)
        delta=states[ix,:3]-ref[:3]
        local=delta@rotation
        assert np.max(np.abs(local/.1-arrays['targets'][i,:,:3]))<1e-5
        return dict(window=i,episode=key[0]+'::'+str(key[1]),start_index=t,
            reference_xyz_m=ref[:3].tolist(),reference_rpy_deg=np.degrees(ref[3:6]).tolist(),
            reference_rotation=rotation.tolist(),base_frame_xyz_trajectory_cm=(delta*100).tolist(),
            local_xyz_trajectory_cm=(local*100).tolist())
    for i in analysis['contact_sheet_windows']:
        j=int(window_rows[i]['nearest_visual_train_window']);a,b=raw_state(i),raw_state(j)
        ra=np.asarray(a['reference_rotation']);rb=np.asarray(b['reference_rotation'])
        angle=float(np.degrees(np.arccos(np.clip((np.trace(ra.T@rb)-1)/2,-1,1))))
        gap=lambda key:float(np.linalg.norm(np.asarray(a[key])-np.asarray(b[key]),axis=-1).mean())
        pairs.append(dict(task=rows[i]['task'],development=a,nearest_train=b,
            input_frame_rotation_difference_deg=angle,base_frame_position_gap_cm=gap('base_frame_xyz_trajectory_cm'),
            local_position_gap_cm=gap('local_xyz_trajectory_cm')))
    result=dict(stage='raw_reference_pose_check',raw_identity_verified=True,local_xyz_reconstruction_verified=True,
        pairs=pairs,limitations=['Base-coordinate comparisons assume consistent robot bases; camera calibration was not supplied',
            'Reference-frame differences alone do not establish missing conditioning as a causal root',
            'Only four input-similarity-selected pairs inspected; no training'])
    (OUT/'raw_pose_pairs.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps([{k:p[k] for k in ('task','input_frame_rotation_difference_deg','base_frame_position_gap_cm','local_position_gap_cm')} for p in pairs],indent=2))


if __name__=='__main__':
    assert len(sys.argv)==2 and sys.argv[1] in ('encode','analyze','poses')
    globals()[sys.argv[1]]()

"""Read-only same-task train/development observation audit; no new model inference."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np

os.environ.update(CUDA_VISIBLE_DEVICES='-1', USE_TF='0', HF_HUB_OFFLINE='1',
                  TRANSFORMERS_OFFLINE='1', TF_CPP_MIN_LOG_LEVEL='3')
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'reports/bridge_broccoli_observation_audit_203699'
TASK = 'take broccoli out of pan'

def read(p):
    return json.loads(p.read_text(encoding='utf-8'))

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def load():
    p = read(ROOT/'cluster_packages/bridge_author_reference_v1/training_protocol.json')
    rows = p['rows']
    prior = ROOT/'reports/bridge_condition_input_audit_20261013'
    raw = read(prior/'raw_audit.json')
    assert raw['passed'] and raw['input_windows_sha256'] == sha(prior/'input_windows.csv')
    assert raw['observed_states_sha256'] == sha(prior/'observed_states.npz')
    records = list(csv.DictReader((prior/'input_windows.csv').open(encoding='utf-8-sig')))
    for i, row in enumerate(rows):
        assert int(records[i]['input_index']) == row['start_index'] and records[i]['task'] == row['task']
        assert records[i]['episode'] == row['shard']+'::'+str(row['record_index'])
    with np.load(prior/'observed_states.npz', allow_pickle=False) as z:
        state, future = z['current_state'].copy(), z['future_state'].copy()
    visual_dir = ROOT/'reports/bridge_growth128_generalization_201663_20261013'
    encoding = read(visual_dir/'encoding.json')
    assert encoding['features_sha256'] == sha(visual_dir/'frozen_visual_descriptors.npz')
    with np.load(visual_dir/'frozen_visual_descriptors.npz', allow_pickle=False) as z:
        visual = z['features'].astype(float)
    visual /= np.linalg.norm(visual, axis=-1, keepdims=True)
    ref_dir = ROOT/'cluster_packages'
    sys.path.insert(0, str(ref_dir/'bridge_author_preflight_v1'))
    from bridge_author_io import assemble_references
    refs = assemble_references(ref_dir)
    assert encoding['processed_image_sha256'] == [hashlib.sha256(x.tobytes()).hexdigest() for x in refs['images']]
    author_dir = ROOT/'reports/bridge-author-reference-203699'
    analysis = read(ROOT/'reports/bridge_author_reference_analysis_203699/analysis.json')
    assert analysis['audit_passed'] and all(sha(author_dir/n) == h for n,h in analysis['files_sha256'].items())
    pred = {}
    for arm in ('raw','ema'):
        blocks = []
        for split in ('train','development'):
            with np.load(author_dir/(arm+'_'+split+'.npz'), allow_pickle=False) as z:
                blocks.append(z['predictions'].copy())
        pred[arm] = np.concatenate(blocks, axis=1)
    ti = np.array([i for i,r in enumerate(rows[:128]) if r['task'] == TASK])
    di = np.array([i for i,r in enumerate(rows) if i>=128 and r['task'] == TASK])
    assert len(ti)==24 and len(di)==9
    return rows,records,state,future,visual,refs,pred,ti,di

def analyze():
    OUT.mkdir(exist_ok=True)
    rows,records,state,future,visual,refs,pred,ti,di = load()
    sys.path.insert(0, str(ROOT/'scripts'))
    from analyze_bridge_author_reference_203699 import errors, basic
    from audit_bridge_condition_inputs import quaternion
    q = quaternion(state[:,3:6])
    # Preserve the previously fixed input-neighbor distance; no target/error selection.
    xyz_std = state[:128,:3].std(0)
    target = refs['targets'][...,:7]
    train_min,train_max = target[:128].min((0,1)),target[:128].max((0,1))
    task_min,task_max = target[ti].min((0,1)),target[ti].max((0,1))
    state_min,state_max = state[ti,:3].min(0),state[ti,:3].max(0)
    items=[]
    for i in np.r_[ti,di]:
        i=int(i); row=rows[i]; ep=records[i]['episode']; n=int(records[i]['episode_steps'])
        candidates=ti[[records[j]['episode']!=ep for j in ti]]
        assert len(candidates)>0
        visual_neighbor=int(candidates[np.argmax(visual[candidates]@visual[i])])
        angle=2*np.arccos(np.clip(np.abs(q[candidates]@q[i]),0,1))
        distance=np.mean(((state[candidates,:3]-state[i,:3])/xyz_std)**2,axis=-1)+(angle/np.pi)**2
        state_neighbor=int(candidates[np.argmin(distance)])
        item=dict(window=i,partition='train' if i<128 else 'development',episode=ep,start=row['start_index'],
            steps=n,index_fraction=row['start_index']/(n-1),
            future_fraction=16/(n-1),current_xyz_m=state[i,:3].tolist(),
            current_measured_gripper=float(state[i,6]),current_outside_same_task_train_box=bool(
                ((state[i,:3]<state_min-1e-6)|(state[i,:3]>state_max+1e-6)).any()),
            target_xyz_outside_same_task_train_count=int(((target[i,:,:3]<task_min[:3]-1e-6)|
                (target[i,:,:3]>task_max[:3]+1e-6)).any(-1).sum()),
            target_xyz_outside_all_train_count=int(((target[i,:,:3]<train_min[:3]-1e-6)|
                (target[i,:,:3]>train_max[:3]+1e-6)).any(-1).sum()),neighbors={})
        for method,j in [('visual',visual_neighbor),('state',state_neighbor)]:
            po,ro,_=errors(target[j][None,None],target[i:i+1])
            item['neighbors'][method]=dict(window=j,episode=records[j]['episode'],start=rows[j]['start_index'],
                index_fraction=rows[j]['start_index']/(int(records[j]['episode_steps'])-1),
                visual_cosine=float(visual[j]@visual[i]),current_position_gap_cm=float(np.linalg.norm(state[i,:3]-state[j,:3])*100),
                current_rotation_gap_deg=float(np.degrees(2*np.arccos(np.clip(abs(q[i]@q[j]),0,1)))),
                future_local_position_gap_cm=float(po.mean()),future_local_rotation_gap_deg=float(ro.mean()))
        po,ro,_=errors(pred['ema'][:,i:i+1],target[i:i+1])
        item['ema']=basic(po,ro)
        for method,j in [('visual',visual_neighbor),('state',state_neighbor)]:
            cp,cr,_=errors(pred['ema'][:,i:i+1],target[j:j+1])
            item['neighbors'][method]['ema_prediction_to_neighbor_target_position_cm']=float(cp.mean())
            item['neighbors'][method]['ema_prediction_to_neighbor_target_rotation_deg']=float(cr.mean())
        item['current_outside_all_train_box']=bool(((state[i,:3]<state[:128,:3].min(0)-1e-6)|
            (state[i,:3]>state[:128,:3].max(0)+1e-6)).any())
        items.append(item)
    episodes=[]
    for ep in dict.fromkeys(x['episode'] for x in items):
        group=[x for x in items if x['episode']==ep];indices=[x['window'] for x in group]
        po,ro,_=errors(pred['ema'][:,indices],target[indices]); static=np.zeros_like(target[indices]);static[...,6]=1
        sp,sr,_=errors(static[None],target[indices])
        episodes.append(dict(episode=ep,partition=group[0]['partition'],steps=group[0]['steps'],
            starts=[x['start'] for x in group],input_index_fractions=[x['index_fraction'] for x in group],
            ema=basic(po,ro),static=basic(sp,sr),
            outside_current_windows=sum(x['current_outside_same_task_train_box'] for x in group),
            outside_task_targets=sum(x['target_xyz_outside_same_task_train_count'] for x in group),
            outside_global_targets=sum(x['target_xyz_outside_all_train_count'] for x in group)))
    result=dict(passed=True,stage='broccoli_shared_observation_target_suitability_audit',
        trained=False,new_model_loaded=False,new_visual_encoding=False,
        fixed_scope='All24 retained same-task training windows and9 development windows;12 train and2 development episodes',
        task_selected_after_observing_errors=True,independent_test=False,reserved_test_targets_read=False,
        timestamps_available=False,physical_prediction_duration_verified=False,
        label='16 subsequent recorded states relative to the current tool frame, not16 executed control commands',
        visual_neighbor_description='Previously saved frozenCLIP descriptor; not the author trainable visual encoder',
        input_neighbor_selection_uses_future_targets_or_prediction_errors=False,
        state_train_same_task_min_m=state_min.tolist(),state_train_same_task_max_m=state_max.tolist(),
        task_target_xyz_min_cm=(task_min[:3]*10).tolist(),task_target_xyz_max_cm=(task_max[:3]*10).tolist(),
        episodes=episodes,windows=items)
    for split in ('train','development'):
        group=[x for x in items if x['partition']==split]
        result[split]=dict(windows=len(group),episodes=len([e for e in episodes if e['partition']==split]),
            current_outside_same_task_train_box_windows=sum(x['current_outside_same_task_train_box'] for x in group),
            current_outside_all_train_box_windows=sum(x['current_outside_all_train_box'] for x in group),
            target_xyz_outside_same_task_train_count=sum(x['target_xyz_outside_same_task_train_count'] for x in group),
            target_xyz_outside_all_train_count=sum(x['target_xyz_outside_all_train_count'] for x in group),
            total_action_targets=16*len(group),neighbor_diagnostics={m:{k:float(np.mean([x['neighbors'][m][k] for x in group]))
                for k in ('visual_cosine','current_position_gap_cm','current_rotation_gap_deg','future_local_position_gap_cm','future_local_rotation_gap_deg')}
                for m in ('visual','state')})
    result['input_sha256']={str(p.relative_to(ROOT)):sha(p) for p in (
        ROOT/'cluster_packages/bridge_author_reference_v1/training_protocol.json',
        ROOT/'reports/bridge_condition_input_audit_20261013/raw_audit.json',
        ROOT/'reports/bridge_condition_input_audit_20261013/input_windows.csv',
        ROOT/'reports/bridge_condition_input_audit_20261013/observed_states.npz',
        ROOT/'reports/bridge_growth128_generalization_201663_20261013/frozen_visual_descriptors.npz')}
    (OUT/'analysis.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    render(rows,refs,ti,di,items)
    render_specific_case(rows,refs,pred)
    print(json.dumps({k:result[k] for k in ('train','development')},indent=2))

def render_specific_case(rows,refs,pred):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'font.size':9})
    i,j=146,114
    mean=np.array([.48145466,.4578275,.40821073])[:,None,None]
    std=np.array([.26862954,.26130258,.27577711])[:,None,None]
    fig,axes=plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
    for ax,k,label in [(axes[0,0],i,'留出00011::24，第8帧'),(axes[0,1],j,'训练00061::25，第0帧')]:
        ax.imshow(np.clip(refs['images'][k]*std+mean,0,1).transpose(1,2,0));ax.axis('off');ax.set_title(label)
    axes[0,2].axis('off')
    axes[0,2].text(0,.9,'相同任务指令\nCLIP视觉相似度：0.934\n当前末端位置相差：10.07 cm\n\n作者预测到真实目标：15.39 cm\n作者预测到训练近邻目标：1.64 cm\n\n这是固定样本的诊断线索，\n不是作者采用最近邻算法的证明。',va='top',linespacing=1.8)
    for k,ax in enumerate(axes[1]):
        ax.plot(np.arange(1,17),refs['targets'][i,:,k]*10,color='#c35349',label='留出真实目标')
        ax.plot(np.arange(1,17),refs['targets'][j,:,k]*10,color='#808080',linestyle='--',label='训练近邻真实目标')
        for draw in range(3):ax.plot(np.arange(1,17),pred['ema'][draw,i,:,k]*10,color='#247bb4',alpha=.65,
            label='作者EMA，3次采样' if draw==0 else None)
        ax.set(xlabel='未来目标序号',ylabel='相对当前工具坐标的'+('X','Y','Z')[k]+'位移（cm）')
        ax.grid(alpha=.2);ax.set_xticks([1,4,8,12,16])
    axes[1,0].legend(fontsize=8)
    fig.suptitle('一个范围内的失败窗口：预测接近另一条训练轨迹｜使用已有预测，无新推理')
    fig.savefig(OUT/'in_range_failure_case.png',dpi=140);plt.close(fig)

def render(rows,refs,ti,di,items):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'font.size':9})
    mean=np.array([.48145466,.4578275,.40821073])[None,:,None,None]
    std=np.array([.26862954,.26130258,.27577711])[None,:,None,None]
    images=np.clip(refs['images']*std+mean,0,1).transpose(0,2,3,1)
    for split,indices,ncols in [('train',ti,6),('development',di,3)]:
        nrows=(len(indices)+ncols-1)//ncols
        fig,axes=plt.subplots(nrows,ncols,figsize=(ncols*2.5,nrows*2.5),squeeze=False,constrained_layout=True)
        for ax in axes.flat:ax.axis('off')
        for ax,i in zip(axes.flat,indices):
            row=rows[int(i)];ax.imshow(images[i]);ax.set_title(row['shard'].split('tfrecord-')[1].split('-of')[0]+'::'+str(row['record_index'])+'  t='+str(row['start_index']))
        fig.suptitle(('训练：12条演示、24幅实际输入' if split=='train' else '留出：2条演示、9幅实际输入')+'｜均为取出西兰花任务，无新模型推理')
        fig.savefig(OUT/(split+'_input_images.png'),dpi=135);plt.close(fig)
    dev=[x for x in items if x['partition']=='development']
    fig,axes=plt.subplots(len(dev),2,figsize=(7,len(dev)*2.4),constrained_layout=True)
    for pair,item in zip(axes,dev):
        for ax,i,label in [(pair[0],item['window'],'留出输入'),(pair[1],item['neighbors']['visual']['window'],'同任务视觉近邻训练输入')]:
            row=rows[i];ax.imshow(images[i]);ax.axis('off');ax.set_title(label+' '+row['shard'].split('tfrecord-')[1].split('-of')[0]+'::'+str(row['record_index'])+' t='+str(row['start_index']))
    fig.suptitle('全部9个留出窗口的固定CLIP视觉近邻｜不按预测误差选择近邻')
    fig.savefig(OUT/'all_development_visual_neighbors.png',dpi=110);plt.close(fig)

def raw_episode_audit():
    OUT.mkdir(exist_ok=True)
    rows,records,state,future,visual,refs,pred,ti,di=load()
    sys.path.insert(0,str(ROOT/'cluster_packages/bridge_growth128_v1'))
    import torch
    torch.set_num_threads(4)
    from dataset import UnifiedRobotDataset
    from PIL import Image
    import io
    data=ROOT/'training_cache/exports/bridge_expansion_training_v1'
    manifest=read(data/'manifest.json')
    protected={(r['shard'],r['record_index']) for r in manifest['reserved_test_identity_only']}
    keys=list(dict.fromkeys((rows[int(i)]['shard'],rows[int(i)]['record_index']) for i in np.r_[ti,di]))
    assert not set(keys)&protected and len(keys)==14
    checked=set();output=[];positions={}
    for shard,record_index in keys:
        path=data/'data'/shard
        if shard not in checked:
            assert sha(path)==manifest['shard_sha256'][shard];checked.add(shard)
        example=UnifiedRobotDataset._load_tfrecord_example(str(path),record_index)
        fields=example.features.feature
        images=fields['steps/observation/image_0'].bytes_list.value
        language=fields['steps/language_instruction'].bytes_list.value
        n=len(images);states=np.array(fields['steps/observation/state'].float_list.value,dtype='<f4').reshape(n,7)
        action=np.array(fields['steps/action'].float_list.value,dtype='<f4').reshape(n,7)
        indices=[int(i) for i in np.r_[ti,di] if rows[int(i)]['shard']==shard and rows[int(i)]['record_index']==record_index]
        assert len(language)==n and np.isfinite(states).all() and np.isfinite(action).all()
        assert {int(records[i]['episode_steps']) for i in indices}=={n}
        for i in indices:
            t=rows[i]['start_index'];assert np.array_equal(states[t],state[i].astype('<f4'))
            assert np.array_equal(states[t+1:t+17],future[i].astype('<f4'))
        # Save fixed full-episode phase frames; these are diagnostic future views, never model inputs.
        frame_indices=sorted(set([0,n//4,n//2,3*n//4,n-1]+[rows[i]['start_index'] for i in indices]))
        selected=[]
        for t in frame_indices:
            Image.open(io.BytesIO(images[t])).convert('RGB').save(OUT/(shard.split('tfrecord-')[1].split('-of')[0]+'_'+str(record_index)+'_t'+str(t)+'.png'))
            selected.append(t)
        ep=shard+'::'+str(record_index)
        positions[ep]=states
        output.append(dict(episode=ep,partition='train' if indices[0]<128 else 'development',steps=n,
            input_indices=[rows[i]['start_index'] for i in indices],saved_frame_indices=selected,
            unique_canonical_instructions=sorted({' '.join(s.decode('utf-8').lower().split()) for s in language}),
            explicit_timestamp_fields=sorted(k for k in fields if 'timestamp' in k.lower() or 'stamp' in k.lower() or '/time' in k.lower()),
            state_xyz_min_m=states[:,:3].min(0).tolist(),state_xyz_max_m=states[:,:3].max(0).tolist(),
            recorded_position_path_length_cm=float(np.linalg.norm(np.diff(states[:,:3].astype(float),axis=0),axis=1).sum()*100),
            measured_gripper_min_max=[float(states[:,6].min()),float(states[:,6].max())],
            raw_state_and_future_identity_verified=True))
        print('BROCCOLI_RAW_EPISODE',len(output),'/14',ep,flush=True)
    np.savez_compressed(OUT/'raw_episode_states.npz',**positions)
    result=dict(passed=True,trained=False,model_loaded=False,new_policy_inference=False,
        selected_episodes=14,selected_shards=len(checked),reserved_test_targets_read=False,
        all_canonical_instruction_equal=all(x['unique_canonical_instructions']==[TASK] for x in output),
        future_frames_used_as_policy_inputs=False,physical_timestamps_verified=False,episodes=output)
    assert result['all_canonical_instruction_equal']
    (OUT/'raw_episode_audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['analyze','raw'])
    args=ap.parse_args()
    analyze() if args.mode=='analyze' else raw_episode_audit()

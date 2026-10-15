"""Read-only168-window current-state/index audit; reuse saved CLIP descriptors."""
import ast
from collections import defaultdict
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time
import numpy as np

os.environ.update(CUDA_VISIBLE_DEVICES='-1',USE_TF='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TF_CPP_MIN_LOG_LEVEL='3')
ROOT=Path(__file__).resolve().parents[1]
PACK=ROOT/'cluster_packages/bridge_growth128_v1'
DATA=ROOT/'reports/bridge-growth128-fit-201663'
VISUAL=ROOT/'reports/bridge_growth128_generalization_201663_20261013'
OUT=ROOT/'reports/bridge_condition_input_audit_20261013'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def unit(v):return v/np.maximum(np.linalg.norm(v,axis=-1,keepdims=True),1e-15)
def save_csv(name,records):
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
def quaternion(rpy):
    r,p,y=np.moveaxis(np.asarray(rpy,dtype=float)/2,-1,0)
    cr,cp,cy=np.cos(r),np.cos(p),np.cos(y);sr,sp,sy=np.sin(r),np.sin(p),np.sin(y)
    return unit(np.stack([sr*cp*cy-cr*sp*sy,cr*sp*cy+sr*cp*sy,cr*cp*sy-sr*sp*cy,cr*cp*cy+sr*sp*sy],axis=-1))
def rotation(rpy):
    r,p,y=np.asarray(rpy,dtype=float);cr,cp,cy=np.cos([r,p,y]);sr,sp,sy=np.sin([r,p,y])
    return np.array([[cy*cp,cy*sp*sr-sy*cr,cy*sp*cr+sy*sr],[sy*cp,sy*sp*sr+cy*cr,sy*sp*cr-cy*sr],[-sp,cp*sr,cp*cr]])
def relative_q(q0,q1):
    a=-q0[:3];s=q0[3];b=q1[...,:3];t=q1[...,3:4]
    return unit(np.concatenate([s*b+t*a+np.cross(a,b),s*t-np.sum(a*b,axis=-1,keepdims=True)],axis=-1))
def geometry_checks():
    assert np.allclose(rotation([0,0,np.pi/2])@np.array([1,0,0]),[0,1,0],atol=1e-12)
    q=quaternion([.2,-.3,.4]);assert np.allclose(relative_q(q,q),[0,0,0,1],atol=1e-12)
    assert np.allclose(relative_q(quaternion([0,0,0]),quaternion([0,0,np.pi/2])),[0,0,2**-.5,2**-.5],atol=1e-12)

def original_inputs():
    m=read(PACK/'manifest.json');r=read(DATA/'report.json')
    assert read(ROOT/'reports/bridge_growth128_analysis_201663_20261013/analysis.json')['verified']
    assert r['manifest_sha256']==sha(PACK/'manifest.json') and r['source_sha256']==m['source_sha256']
    assert all(sha(PACK/n)==h for n,h in m['source_sha256'].items())
    paths=[ROOT/'reports/bridge_batch_input_reference_20261012/arrays.npz',ROOT/'cluster_packages/bridge_growth64_v1/added_reference/arrays.npz',PACK/'new_reference/arrays.npz']
    refs=[]
    for p in paths:
        with np.load(p,allow_pickle=False) as z:refs.append({k:z[k].copy() for k in z.files})
    arrays={k:np.concatenate([refs[0][k][:32],refs[1][k],refs[2][k],refs[0][k][32:]]) for k in refs[0]}
    with np.load(DATA/'reduced_u8192_train.npz') as z:assert np.max(np.abs(z['targets']-arrays['targets'][:128]))<=1e-6
    with np.load(DATA/'reduced_u8192_development.npz') as z:assert np.max(np.abs(z['targets']-arrays['targets'][128:]))<=1e-6
    return m,r,arrays,paths

def extract():
    started=time.monotonic();geometry_checks()
    sys.path.insert(0,str(PACK))
    import torch
    from PIL import Image
    from dataset import UnifiedRobotDataset,prepare_image
    from bridge_input_contract import raw_identity
    torch.set_num_threads(4)
    m,report,arrays,refs=original_inputs();rows=m['train']+m['development'];assert len(rows)==168
    export=ROOT/'training_cache/exports/bridge_expansion_training_v1';inventory=read(export/'manifest.json')
    protected={(r['shard'],r['record_index']) for r in inventory['reserved_test_identity_only']}
    assert not protected&{(r['shard'],r['record_index']) for r in rows}
    checked=set();cache={};records=[];current=[];future=[];maxima=dict(image=0.,current_gripper=0.,relative_xyz_component=0.,relative_quaternion_component=0.)
    timestamp_fields=set();episode_lengths={}
    for i,row in enumerate(rows):
        key=(row['shard'],row['record_index']);path=export/'data'/row['shard']
        if path.name not in checked:assert sha(path)==inventory['shard_sha256'][path.name];checked.add(path.name)
        if key not in cache:cache[key]=UnifiedRobotDataset._load_tfrecord_example(str(path),row['record_index'])
        example=cache[key];fields=example.features.feature;t=row['start_index']
        ai=i if i<32 else (i+40 if i<128 else i-96)
        assert raw_identity(example,t)==report['data_checks'][ai]['raw_identity']
        languages=fields['steps/language_instruction'].bytes_list.value;images=fields['steps/observation/image_0'].bytes_list.value;n=len(languages)
        states=np.asarray(fields['steps/observation/state'].float_list.value,dtype='<f4').reshape(n,7)
        actions=np.asarray(fields['steps/action'].float_list.value,dtype='<f4').reshape(n,7)
        assert len(images)==n and np.isfinite(states).all() and np.isfinite(actions).all()
        assert np.flatnonzero(fields['steps/is_first'].int64_list.value).tolist()==[0]
        assert np.flatnonzero(fields['steps/is_last'].int64_list.value).tolist()==[n-1]
        assert 0<=t and t+16<n  # All16 targets exist, with no tail padding.
        instruction=languages[t].decode('utf-8',errors='strict')
        assert ' '.join(instruction.lower().split())==' '.join(row['task'].lower().split())
        rgb=np.asarray(Image.open(io.BytesIO(images[t])).convert('RGB'))
        image_delta=float(np.max(np.abs(prepare_image(rgb).numpy()-arrays['images'][i])))
        current_delta=float(np.max(np.abs(np.asarray([2*states[t,6]-1],dtype=np.float32)-arrays['currents'][i])))
        assert image_delta<=1e-6 and current_delta==0
        ref=states[t].astype(float);later=states[t+1:t+17].astype(float);r=rotation(ref[3:6]);q=quaternion(ref[3:6])
        xyz=(later[:,:3]-ref[:3])@r/.1
        qrel=relative_q(q,quaternion(later[:,3:6]));stored=arrays['targets'][i,:,3:7].astype(float)
        xyz_delta=float(np.max(np.abs(xyz-arrays['targets'][i,:,:3])))
        q_delta=float(np.min(np.stack([np.max(np.abs(qrel-stored),axis=-1),np.max(np.abs(qrel+stored),axis=-1)]),axis=0).max())
        assert xyz_delta<=1e-5 and q_delta<=2e-6
        for k,v in [('image',image_delta),('current_gripper',current_delta),('relative_xyz_component',xyz_delta),('relative_quaternion_component',q_delta)]:maxima[k]=max(maxima[k],v)
        timestamp_fields.update(k for k in fields if 'timestamp' in k.lower() or 'stamp' in k.lower() or '/time' in k.lower())
        episode=key[0]+'::'+str(key[1]);episode_lengths[episode]=n
        record=dict(window=i,split='train' if i<128 else 'development',episode=episode,task=row['task'],input_index=t,
            image_index=t,instruction_index=t,current_state_index=t,target_first_index=t+1,target_last_index=t+16,
            episode_steps=n,raw_identity_passed=True,xyz_reconstruction_max_abs=xyz_delta,quaternion_reconstruction_max_abs=q_delta,
            current_x_m=float(ref[0]),current_y_m=float(ref[1]),current_z_m=float(ref[2]),
            current_roll_rad=float(ref[3]),current_pitch_rad=float(ref[4]),current_yaw_rad=float(ref[5]),current_measured_gripper=float(ref[6]))
        records.append(record);current.append(ref);future.append(later)
        if (i+1)%40==0:print('RAW_INPUT_AUDIT',i+1,'/168',flush=True)
    OUT.mkdir(exist_ok=True)
    np.savez_compressed(OUT/'observed_states.npz',current_state=np.asarray(current),future_state=np.asarray(future))
    save_csv('input_windows.csv',records)
    # Check the actual frozen pose-decoder call boundary, not a generic README.
    entry=ast.parse((PACK/'fit_bridge_growth128.py').read_text(encoding='utf-8'));model=ast.parse((PACK/'models.py').read_text(encoding='utf-8'))
    context=next(n for n in ast.walk(entry) if isinstance(n,ast.FunctionDef) and n.name=='context')
    assert {n.id for n in ast.walk(context) if isinstance(n,ast.Name)}=={'indices','now','model','visual','language','mask','repeated'}
    loss=next(n for n in ast.walk(model) if isinstance(n,ast.FunctionDef) and n.name=='diffusion_loss')
    decoder_call=next(n for n in ast.walk(loss) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='diffusion_decoder')
    assert [n.id for n in decoder_call.args]==['noisy_actions','timesteps','context']
    result=dict(stage='raw_input_index_and_current_state_audit',passed=True,trained=False,model_loaded=False,new_policy_inference=False,new_clip_encoding=False,
        windows=168,train_windows=128,development_windows=40,selected_episodes=len(cache),selected_shards=len(checked),raw_identities_verified=168,
        full16_targets_without_padding=168,explicit_timestamp_fields=sorted(timestamp_fields),
        physical_sensor_synchronization_verified=False,source_index_alignment_verified=True,
        independent_geometry_synthetic_checks_passed=True,max_absolute_differences=maxima,
        current_state_available=True,current_state_shape=[168,7],
        current_pose_used_to_construct_targets=True,current_pose_numerically_supplied_to_pose_network=False,
        current_gripper_only_supplied_to_separate_gripper_head=True,pose_context='Current CLIP image/text tokens through Adapter; repeated current context, no real previous-frame input in this frozen runner',
        units='Project interpretation: xyz metres; rpy radians; measured gripper raw affine observation, not calibrated probability',
        source_sha256={n:sha(PACK/n) for n in ('dataset.py','models.py','adapter.py','fit_bridge_growth128.py')},
        manifest_sha256=sha(PACK/'manifest.json'),reference_sha256={str(p.relative_to(ROOT)):sha(p) for p in refs},
        observed_states_sha256=sha(OUT/'observed_states.npz'),input_windows_sha256=sha(OUT/'input_windows.csv'),
        source_script_sha256=sha(Path(__file__)),elapsed_seconds=time.monotonic()-started,reserved_test_targets_read=False,
        limits=['Index alignment is not independent proof of hardware sensor synchronization','Physical calibration/base-frame consistency across episodes not independently supplied',
            'Source inspection applies to frozen201663 runner, not every project training mode','A missing explicit pose input is a hypothesis, not proof of faulty vision or unique root cause'])
    (OUT/'raw_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('passed','windows','selected_episodes','max_absolute_differences','explicit_timestamp_fields','elapsed_seconds')},indent=2))

def analyze():
    from analyze_bridge_training_trace_200039 import errors
    m,report,arrays,refs=original_inputs();rows=m['train']+m['development']
    raw=read(OUT/'raw_audit.json');encoding=read(VISUAL/'encoding.json')
    assert raw['passed'] and raw['observed_states_sha256']==sha(OUT/'observed_states.npz')
    assert raw['manifest_sha256']==encoding['manifest_sha256']==sha(PACK/'manifest.json')
    assert encoding['reference_sha256']==raw['reference_sha256']=={str(p.relative_to(ROOT)):sha(p) for p in refs}
    assert encoding['features_sha256']==sha(VISUAL/'frozen_visual_descriptors.npz')
    assert encoding['processed_image_sha256']==[hashlib.sha256(v.tobytes()).hexdigest() for v in arrays['images']]
    with np.load(VISUAL/'frozen_visual_descriptors.npz') as z:visual=unit(z['features'].astype(float))
    with np.load(OUT/'observed_states.npz') as z:state=z['current_state'];later=z['future_state']
    assert state.shape==(168,7) and np.isfinite(state).all()
    q=quaternion(state[:,3:6]);truth=arrays['targets'];tasks=np.asarray([r['task'] for r in rows]);episodes=np.asarray([r['shard']+'::'+str(r['record_index']) for r in rows])
    mean=state[:128,:3].mean(0);std=state[:128,:3].std(0);assert (std>1e-8).all()
    pool=defaultdict(list)
    for r in read(ROOT/'reports/bridge-xyz-eval-197349.json')['window_selection']:pool[(r['shard'],r['record_index'])].append(r['start_index'])
    phases=[]
    for r in rows:
        ss=pool[(r['shard'],r['record_index'])];t=r['start_index']
        phases.append('first' if t==min(ss) else ('last' if t==max(ss) else 'middle'))
    records=[];retrieved={k:[] for k in ('visual','state')}
    for i,row in enumerate(rows):
        candidates=np.flatnonzero((tasks[:128]==tasks[i])&(episodes[:128]!=episodes[i]));assert len(candidates)
        sim=visual[candidates]@visual[i];v=int(candidates[sim.argmax()])
        dxyz=np.mean(((state[candidates,:3]-state[i,:3])/std)**2,axis=-1)
        angle=2*np.arccos(np.clip(np.abs(q[candidates]@q[i]),0,1))
        state_distance=dxyz+(angle/np.pi)**2  # Predetermined; no development fitting.
        s=int(candidates[state_distance.argmin()])
        for kind,j in [('visual',v),('state',s)]:
            p,r=errors(truth[j][None,None],truth[i][None]);retrieved[kind].append(truth[j])
            pose_distance=float(np.degrees(2*np.arccos(np.clip(abs(q[i]@q[j]),0,1))))
            records.append(dict(window=i,split='train_leave_same_episode_out' if i<128 else 'development',episode=episodes[i],task=tasks[i],phase=phases[i],neighbor_method=kind,
                neighbor_window=j,neighbor_episode=episodes[j],neighbor_start_index=rows[j]['start_index'],
                visual_cosine=float(visual[i]@visual[j]),current_position_gap_cm=float(np.linalg.norm(state[i,:3]-state[j,:3])*100),
                current_rotation_gap_deg=pose_distance,future_local_position_gap_cm=float(p.mean()),future_local_rotation_gap_deg=float(r.mean()),
                future_absolute_base_position_gap_cm=float(np.linalg.norm(later[i,:,:3]-later[j,:,:3],axis=-1).mean()*100)))
    save_csv('nearest_input_pairs.csv',records)
    episode_rows=[]
    fields=['visual_cosine','current_position_gap_cm','current_rotation_gap_deg','future_local_position_gap_cm','future_local_rotation_gap_deg','future_absolute_base_position_gap_cm']
    for split in ('train_leave_same_episode_out','development'):
        for method in ('visual','state'):
            for ep in sorted({r['episode'] for r in records if r['split']==split}):
                group=[r for r in records if r['split']==split and r['neighbor_method']==method and r['episode']==ep]
                episode_rows.append(dict(split=split,method=method,episode=ep,task=group[0]['task'],windows=len(group),**{k:float(np.mean([r[k] for r in group])) for k in fields}))
    save_csv('episode_neighbor_diagnostics.csv',episode_rows)
    summaries=[]
    for split in ('train_leave_same_episode_out','development'):
        for method in ('visual','state'):
            group=[r for r in episode_rows if r['split']==split and r['method']==method]
            summaries.append(dict(split=split,method=method,episodes=len(group),**{k:float(np.mean([r[k] for r in group])) for k in fields}))
    state_ranges=[]
    for task in sorted(set(tasks)):
        ti=np.flatnonzero(tasks[:128]==task);di=np.flatnonzero(tasks[128:]==task)+128
        lo=state[ti,:3].min(0);hi=state[ti,:3].max(0);outside=((state[di,:3]<lo-1e-6)|(state[di,:3]>hi+1e-6)).any(-1)
        state_ranges.append(dict(task=task,train_windows=len(ti),development_windows=len(di),train_xyz_min_m=lo.tolist(),train_xyz_max_m=hi.tolist(),
            development_xyz_min_m=state[di,:3].min(0).tolist(),development_xyz_max_m=state[di,:3].max(0).tolist(),
            development_current_xyz_outside_train_box_windows=int(outside.sum()),development_outside_window_indices=di[outside].tolist()))
    duplicate_groups=defaultdict(list);duplicate_pairs=[]
    for i,row in enumerate(rows):duplicate_groups[(row['task'],encoding['processed_image_sha256'][i])].append(i)
    for group in duplicate_groups.values():
        for offset,i in enumerate(group):
            for j in group[offset+1:]:
                p,r=errors(truth[j][None,None],truth[i][None])
                duplicate_pairs.append(dict(first_window=i,second_window=j,same_episode=bool(episodes[i]==episodes[j]),
                    task=tasks[i],current_position_gap_cm=float(np.linalg.norm(state[i,:3]-state[j,:3])*100),
                    future_position_gap_cm=float(p.mean()),future_rotation_gap_deg=float(r.mean())))
    # Six high-input-similarity cases, one per episode per split; no error-based selection.
    selected=[]
    for split in ('train_leave_same_episode_out','development'):
        seen=set()
        candidates=[r for r in records if r['split']==split and r['neighbor_method']=='visual']
        for r in sorted(candidates,key=lambda r:r['visual_cosine'],reverse=True):
            if r['episode'] in seen:continue
            selected.append(r);seen.add(r['episode'])
            if len(seen)==3:break
    save_csv('high_similarity_cases.csv',selected)
    result=dict(stage='observed_state_and_input_neighbor_diagnosis',passed=True,trained=False,new_policy_inference=False,new_clip_encoding=False,
        current_state_alignment_audit_passed=True,explicit_pose_input_missing_in_frozen_runner=True,
        train_only_state_reference=dict(xyz_mean_m=mean.tolist(),xyz_std_m=std.tolist(),fit_windows=128,
            diagnostic_state_distance='Mean squared xyz difference standardized by128 train-window std + squared geodesic rotation/pi; fixed, no development tuning'),
        state_ranges=state_ranges,episode_equal_neighbor_summaries=summaries,high_input_similarity_cases=selected,
        exact_same_task_processed_image_duplicate_pairs=duplicate_pairs,
        raw_audit_sha256=sha(OUT/'raw_audit.json'),source_script_sha256=sha(Path(__file__)),feature_source_sha256=sha(VISUAL/'frozen_visual_descriptors.npz'),
        development_target_used_as_neighbor_selection_input=False,same_episode_candidates_excluded=True,only_same_task_train_candidates=True,
        physical_base_frame_calibration_verified=False,limits=['Image descriptor pools CLS/patch tokens; similarity does not measure equality of all spatial information',
            'Nearest-state uses pose but lacks object state/task progress; retrieval is descriptive, not a trained policy',
            'Neighborhood correlations cannot prove state conditioning will improve a learned policy',
            'Raw future observations used only for retrospective label/retrieval evaluation, not candidate selection or model input',
            'Current-state axis boxes do not establish joint spatial/rotational coverage','No independent timestamp sensor synchronization or calibration supplied',
            'Within-split overlapping windows are correlated; summary gives each episode equal weight','Development set reused for diagnosis, not untouched test'])
    (OUT/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,summaries=summaries,current_xyz_outside={r['task']:r['development_current_xyz_outside_train_box_windows'] for r in state_ranges}),indent=2))

if __name__=='__main__':
    assert len(sys.argv)==2 and sys.argv[1] in ('extract','analyze')
    globals()[sys.argv[1]]()

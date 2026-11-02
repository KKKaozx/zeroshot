"""Freeze a one-time, two-checkpoint Bridge holdout evaluation before decoding labels."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
PACK=ROOT/'cluster_packages/bridge_heldout_evaluation_v1'

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n',encoding='utf-8',newline='\n')
def ep(v):return v['shard'],v['record_index']
def origin(v):return v['origin_file_path'].rstrip('/'),str(v['episode_id'])

def freeze():
    assert not (PACK/'protocol.json').exists(),'Do not alter an already frozen evaluation'
    plan_path=ROOT/'configs/bridge_multitask_pilot_plan.json';plan=read(plan_path)
    pool_path=ROOT/'training_cache/exports/bridge_expansion_training_v1/manifest.json';pool=read(pool_path)
    hold=plan['split_episodes']['reserved_test'];used=pool['partitions']['train']+pool['partitions']['validation']
    assert len(hold)==13 and not set(map(ep,hold))&set(map(ep,used))
    assert not set(map(origin,hold))&set(map(origin,used))
    assert len(set(map(ep,hold)))==len(set(map(origin,hold)))==13
    assert set(map(ep,hold))==set(map(ep,pool['reserved_test_identity_only']))
    for name in ['bridge_expansion_plan_v1.json','bridge_expansion_eligible_plan_v1.json']:
        assert read(ROOT/'configs'/name)['split_episodes']['reserved_test']==hold
    # Search available structured evidence, including episode-key strings; retain all matches.
    wanted=set(map(ep,hold));hits=[];scanned=[]
    def walk(v,path,jpath):
        if isinstance(v,dict):
            shard=v.get('shard',v.get('file_path',''));record=v.get('record_index')
            if isinstance(shard,str) and record is not None and (Path(shard).name,record) in wanted:
                hits.append(dict(file=path,json_path=jpath,shard=Path(shard).name,record_index=record))
            for k,x in v.items():walk(x,path,jpath+'/'+k)
        elif isinstance(v,list):
            for i,x in enumerate(v):walk(x,path,jpath+'/'+str(i))
        elif isinstance(v,str):
            for m in re.finditer(r'(bridge_data_v2-train\.tfrecord-\d+-of-01024)::(\d+)',v):
                if (m[1],int(m[2])) in wanted:hits.append(dict(file=path,json_path=jpath,shard=m[1],record_index=int(m[2])))
    for base in ['reports','configs']:
        for path in sorted((ROOT/base).rglob('*.json')):
            value=read(path);rel=path.relative_to(ROOT).as_posix();scanned.append(dict(file=rel,sha256=sha(path)))
            walk(value,rel,'')
    assert len(hits)==39 and all(h['file'].startswith('configs/') and h['json_path'].startswith('/split_episodes/reserved_test/') for h in hits), 'Unreviewed prior use of holdout identities'
    rows=[dict(shard=e['shard'],record_index=e['record_index'],start_index=t,task=e['instruction'],partition='sealed_test')
          for e in hold for t in range(0,e['steps']-16,4)]
    assert len(rows)==60
    replay_rows=read(ROOT/'cluster_packages/bridge_coverage_replication_v1/rows.json')['development'][:4]
    assert not set(map(ep,replay_rows))&wanted
    inherited=read(ROOT/'cluster_packages/bridge_rotation_weight_v1/protocol.json')['inherited_sha256']
    # Importing these utilities must not change the actual executed production classes.
    inherited['bridge_state_conditioning_v1/fit_state_arm.py']=sha(ROOT/'cluster_packages/bridge_state_conditioning_v1/fit_state_arm.py')
    inherited['bridge_coverage_replication_v1/protocol.json']=sha(ROOT/'cluster_packages/bridge_coverage_replication_v1/protocol.json')
    inherited['bridge_rotation_weight_v1/protocol.json']=sha(ROOT/'cluster_packages/bridge_rotation_weight_v1/protocol.json')
    models=[]
    for label,job,folder in [('coverage','207558','bridge-coverage-rep-fit-207558'),('rotation_weight','207746','bridge-rotation-weight-fit-207746')]:
        report_path=ROOT/'reports'/folder/'report.json';report=read(report_path)
        assert report['execution_checks_passed'] and report['clip_unchanged'] and not report['reserved_test_targets_read']
        check=report['checkpoints'][0]
        models.append(dict(label=label,job=job,report_directory=folder,report_sha256=sha(report_path),checkpoint=check,
            initial_shared_sha256=report['initial_shared_sha256'],augmented_initial_sha256=report['augmented_initial_sha256'],torch_version=report['torch_version']))
    PACK.mkdir(exist_ok=True)
    write(PACK/'selection.json',dict(episodes=hold,rows=rows,replay_rows=replay_rows))
    write(PACK/'metadata_audit.json',dict(passed=True,records=13,windows=60,train_episodes=405,development_episodes=10,
        raw_origin_plus_episode_id_overlap=0,record_identity_overlap=0,history_json_files=scanned,history_identity_hits=hits,
        earlier_model_scores_found=False,raw_targets_decoded=False,model_executed=False,
        limitation='Available structured records only; metadata identity/history evidence is not a guarantee about unrecorded activity or unseen physical scenes.'))
    write(PACK/'protocol.json',dict(date='2026-10-16',purpose='One-time same-task independent-demonstration evaluation of two frozen checkpoints; no model selection or optimization.',
        authorization='User explicitly approved beginning the new-demonstration assessment after explanation of one-time use.',
        selection_sha256=sha(PACK/'selection.json'),metadata_audit_sha256=sha(PACK/'metadata_audit.json'),
        original_plan_sha256=sha(plan_path),pool_manifest_sha256=sha(pool_path),inherited_sha256=inherited,models=models,
        counts=dict(episodes=13,windows=60,replay_windows=4,tasks=5),task_episode_counts=dict(Counter(e['instruction'] for e in hold)),
        window_rule='All valid16-step windows, stride4, start in range(0,steps-16,4); no target/error/range-based dropping or replacement.',
        sampling=dict(steps=100,seeds=[1101,1102,1103],batch_size=2,method='unchanged production DDPM'),
        image_processing='Frozen dataset.py prepare_image; no augmentation.',state_statistics='Original128 training observations only, frozen state protocol.',
        replay_guard=dict(context_atol=1e-4,prediction_atol=1e-4,rtol=0,windows='First4 of the already-scored40 development windows, same order and batch2; interface check only, not rescoring model selection.'),
        metrics=dict(primary='Episode-equal path position_cm and rotation_deg; also report original window-weighted metrics.',
            secondary='16th-target position/rotation, all5tasks, all13episodes, all16horizons, same-target no-motion baseline.',
            comparisons='Both fixed models versus static and each other; no winner chosen for subsequent tuning from this set.',
            mean_advantage='Descriptive only: BOTH episode-equal and window-weighted path position/rotation strictly below corresponding same-set static means. Report endpoint separately; do not claim significance, unseen-task ability or grasp success.'),
        test_use_policy='After first scoring this set is consumed. Do not tune hyperparameters, select seeds/checkpoints, drop episodes or repeatedly evaluate on it. Do not claim this is the final LIBERO/CLIPort or zero-shot benchmark.',
        training=False,optimizer_created=False,gripper_trained=False,simulation=False,downloads=False,
        local_gpu_required=False,resource_limits=dict(package_bytes=100*1024**2,new_cluster_files_bytes=256*1024**2,gpu_reserved_gib=7.5),
        stop_policy='Input/geometry/checkpoint/source/replay failures abort before test inference. One job performs guards then each frozen model once; no retraining or hyperparameter sweep. Preserve failure report.'))
    print('FROZEN BEFORE LABEL DECODING',sha(PACK/'protocol.json'))

def export(source):
    assert (PACK/'protocol.json').is_file() and not (PACK/'assets.json').exists()
    os.environ.update(CUDA_VISIBLE_DEVICES='-1',USE_TF='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TF_CPP_MIN_LOG_LEVEL='3')
    import numpy as np
    import torch
    import tensorflow as tf
    sys.path.insert(0,str(ROOT/'cluster_packages/bridge_growth128_v1'))
    sys.path.insert(0,str(ROOT/'cluster_packages/bridge_state_conditioning_v1'))
    from dataset import UnifiedRobotDataset
    from state_input_contract import state_features
    from bridge_input_contract import raw_identity
    # Independent source-to-world reconstruction, not the production conversion helper.
    sys.path.insert(0,str(ROOT/'scripts'))
    from audit_bridge_end_to_end_20261015 import matrix,qmatrix
    torch.set_num_threads(2)
    p=read(PACK/'protocol.json');s=read(PACK/'selection.json');assert sha(PACK/'selection.json')==p['selection_sha256']
    all_rows=s['replay_rows']+s['rows'];selected=defaultdict(set)
    for row in all_rows:selected[row['shard']].add(row['record_index'])
    records={};files={};source_hashes={};episode_audits=[];images=[];targets=[];currents=[];observed=[];row_checks=[]
    (PACK/'data').mkdir(exist_ok=False)
    for shard,indices in selected.items():
        source_hashes[shard]=sha(source/shard)
        for record in sorted(indices):
            records[(shard,record)]=UnifiedRobotDataset._load_tfrecord_example(str(source/shard),record)
        with tf.io.TFRecordWriter(str(PACK/'data'/shard)) as writer:
            for index in range(max(indices)+1):
                ex=records.get((shard,index));writer.write(ex.SerializeToString(deterministic=True) if ex is not None else b'')
        files['data/'+shard]=sha(PACK/'data'/shard)
    reader=UnifiedRobotDataset.__new__(UnifiedRobotDataset);reader.chunk_size=16
    reader.bridge_gripper_policy='reverse_scan_valid_steps_v2';reader.bridge_current_gripper='continuous'
    hold={ep(e):e for e in s['episodes']}
    for key,expected in hold.items():
        fields=records[key].features.feature;n=len(fields['steps/language_instruction'].bytes_list.value)
        assert fields['episode_metadata/file_path'].bytes_list.value[0].decode()==expected['origin_file_path']
        assert fields['episode_metadata/episode_id'].int64_list.value[0]==expected['episode_id'] and n==expected['steps']
        assert np.flatnonzero(fields['steps/is_first'].int64_list.value).tolist()==[0]
        assert np.flatnonzero(fields['steps/is_last'].int64_list.value).tolist()==[n-1]
        episode_audits.append(dict(**expected,payload_sha256=hashlib.sha256(records[key].SerializeToString(deterministic=True)).hexdigest()))
    for i,row in enumerate(all_rows):
        example=records[ep(row)];f=example.features.feature;t=row['start_index']
        n=len(f['steps/language_instruction'].bytes_list.value)
        states=np.asarray(f['steps/observation/state'].float_list.value,dtype='<f4').reshape(n,7)
        commands=np.asarray(f['steps/action'].float_list.value,dtype='<f4').reshape(n,7)
        assert np.isfinite(states).all() and np.isfinite(commands).all() and t+16<n
        reader._load_tfrecord_example=lambda *_:example
        text,image,current,target=reader._get_tfrecord_bridge_state_action(dict(file_path=row['shard'],record_index=row['record_index'],start_index=t))
        assert ' '.join(text.lower().split())==row['task'] and target.shape==(16,8) and image.shape==(3,224,224)
        y=target.numpy();reference=matrix(states[t,3:6]);max_p=max_r=0.
        for k in range(16):
            max_p=max(max_p,float(np.abs(states[t,:3]+reference@(y[k,:3]*.1)-states[t+k+1,:3]).max()))
            max_r=max(max_r,float(np.abs(reference@qmatrix(y[k,3:7])-matrix(states[t+k+1,3:6])).max()))
        assert max_p<1e-5 and max_r<1e-5
        arrays=[image.numpy(),y,current.numpy()];assert all(np.isfinite(a).all() for a in arrays)
        images.append(arrays[0]);targets.append(y);currents.append(arrays[2]);observed.append(states[t,:6])
        row_checks.append(dict(index=i,row=row,text=text,raw_identity=raw_identity(example,t),payload_sha256=hashlib.sha256(example.SerializeToString(deterministic=True)).hexdigest(),
            source_to_world_position_error_m=max_p,source_to_world_matrix_error=max_r,
            processed_sha256={k:hashlib.sha256(a.tobytes()).hexdigest() for k,a in zip(['image','target','current'],arrays)}))
    state_protocol=read(ROOT/'cluster_packages/bridge_state_conditioning_v1/protocol.json')
    features=state_features(np.stack(observed),state_protocol['state_features'])
    np.savez_compressed(PACK/'input_reference.npz',images=np.stack(images),targets=np.stack(targets),currents=np.stack(currents),observed=np.stack(observed),state_features=features)
    for model in p['models']:
        with np.load(ROOT/'reports'/model['report_directory']/'reduced_u8192_development.npz',allow_pickle=False) as z:
            assert np.abs(z['targets'][:4]-np.stack(targets)[:4]).max()<1e-6
            np.savez_compressed(PACK/(model['label']+'_replay.npz'),predictions=z['predictions'][:,:4],contexts=z['contexts'][:4],targets=z['targets'][:4])
    for name in ['input_reference.npz','coverage_replay.npz','rotation_weight_replay.npz']:files[name]=sha(PACK/name)
    write(PACK/'assets.json',dict(files=files,source_shard_sha256=source_hashes,episodes=episode_audits,row_checks=row_checks,
        source_contract_passed=True,targets_now_decoded_for_frozen_evaluation=True,model_executed=False,local_gpu_used=False,
        no_episodes_or_windows_excluded=True,protocol_frozen_before_decoding_sha256=sha(PACK/'protocol.json')))
    print('CPU INPUT AUDIT COMPLETE',len(all_rows),'observations; test60windows/13episodes; no neural execution')

def package(replace_draft=False):
    assert (PACK/'run_evaluation.py').is_file() and (PACK/'assets.json').is_file()
    files=sorted(p for p in PACK.rglob('*') if p.is_file() and p.name!='SHA256SUMS' and '__pycache__' not in p.parts)
    assert not any(p.suffix=='.pt' for p in files)
    (PACK/'SHA256SUMS').write_text(''.join(sha(p)+'  '+p.relative_to(PACK).as_posix()+'\n' for p in files),encoding='utf-8',newline='\n')
    archive=PACK.with_suffix('.zip')
    assert not archive.exists() or (replace_draft and not (PACK/'evaluation_claim.json').exists() and not (ROOT/'reports/bridge_heldout_submission.json').exists()),'Never replace a submitted package'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for path in files+[PACK/'SHA256SUMS']:z.write(path,path.relative_to(PACK.parent).as_posix())
    assert archive.stat().st_size<100*1024**2
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for path in files+[PACK/'SHA256SUMS']:assert z.read(path.relative_to(PACK.parent).as_posix())==path.read_bytes()
    PACK.with_suffix('.sha256').write_text(sha(archive)+'  '+archive.name+'\n',encoding='utf-8')
    print('PACKAGE',archive,'bytes',archive.stat().st_size,'SHA256',sha(archive))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['freeze','export','package'])
    parser.add_argument('--source',type=Path,default=Path('E:/dataset/bridge_v2_0.0.1/0.0.1'))
    parser.add_argument('--replace-unsubmitted-draft',action='store_true')
    a=parser.parse_args()
    if a.mode=='freeze':freeze()
    elif a.mode=='export':export(a.source)
    else:package(a.replace_unsubmitted_draft)

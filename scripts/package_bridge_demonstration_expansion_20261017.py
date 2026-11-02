"""CPU-only806-pool transfer preparation; export only401 audited additions.

Archives contain at most60MiB of uncompressed record files each. Verified
intermediate TFRecords are removed only after zip contents reproduce their hash;
the original E: dataset, prior packages and all model weights remain untouched.
"""
from collections import defaultdict
import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import zipfile

os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
ROOT=Path(__file__).resolve().parents[1]
PACK=ROOT/'cluster_packages/bridge_demonstration_expansion_v1'
BASE=ROOT/'cluster_packages/bridge_coverage_replication_v1'
DATA=Path('E:/dataset/bridge_v2_0.0.1/0.0.1')
sys.path.insert(0,str(ROOT/'cluster_packages/bridge_growth128_v1'))
sys.path.insert(0,str(ROOT/'cluster_packages/bridge_state_conditioning_v1'))
import numpy as np
import torch
import tensorflow as tf
from dataset import UnifiedRobotDataset
from state_input_contract import state_features


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8',newline='\n')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    assert not (PACK/'protocol.json').exists(),'Preserve frozen transfer preparation'
    assert shutil.disk_usage(ROOT).free>15*1024**3
    torch.set_num_threads(2)
    plan_path=ROOT/'configs/bridge_demonstration_expansion_eligible_20261017.json'
    plan=read(plan_path)
    audit_path=ROOT/'reports/bridge_additional_inventory_20261017/source_audit.json'
    audit=read(audit_path)
    assert audit['passed'] and len(audit['episodes'])==401 and not audit['errors']
    assert sha(audit_path)==plan['new_training_source_audit_sha256']
    assert sha(ROOT/'cluster_packages/bridge_growth128_v1/dataset.py')==audit['dataset_source_sha256']
    shutil.copyfile(ROOT/'reports/bridge_additional_inventory_20261017/eligible_window_rows.json',PACK/'rows.json')
    shutil.copyfile(ROOT/'reports/bridge_additional_inventory_20261017/eligible_schedule.npz',PACK/'schedule.npz')
    shutil.copyfile(audit_path,PACK/'new_source_audit.json')
    shutil.copyfile(BASE/'run_training_coverage.py',PACK/'run_training_coverage.py')
    shutil.copyfile(ROOT/'cluster_packages/bridge_heldout_evaluation_v1/coverage_replay.npz',PACK/'old4_reference.npz')
    all_rows=read(PACK/'rows.json');extra=all_rows['train'][1836:]
    assert len(extra)==1788 and len(all_rows['development'])==40
    by_episode=defaultdict(list)
    for i,row in enumerate(extra):by_episode[(row['shard'],row['record_index'])].append(i)
    audit_at={(e['shard'],e['record_index']):e for e in audit['episodes']}
    grouped=defaultdict(list)
    for key in by_episode:grouped[key[0]].append(key)
    data_dir=PACK/'data';data_dir.mkdir(exist_ok=True)
    assert not list(data_dir.iterdir())
    targets=np.zeros((1788,16,8),np.float32);currents=np.zeros((1788,1),np.float32)
    observed=np.zeros((1788,6),np.float32)
    pixels=np.linspace(0,3*224*224-1,96,dtype=np.int64)
    sketch=np.zeros((1788,96),np.float32)
    references=[None]*1788;files={}
    reader=UnifiedRobotDataset.__new__(UnifiedRobotDataset)
    reader.chunk_size=16;reader.bridge_gripper_policy='reverse_scan_valid_steps_v2';reader.bridge_current_gripper='continuous'
    for count,(shard,keys) in enumerate(sorted(grouped.items())):
        payloads={}
        for key in keys:
            example=UnifiedRobotDataset._load_tfrecord_example(str(DATA/shard),key[1])
            payload=example.SerializeToString(deterministic=True)
            assert hashlib.sha256(payload).hexdigest()==audit_at[key]['payload_sha256']
            payloads[key[1]]=payload
            states=np.asarray(example.features.feature['steps/observation/state'].float_list.value,dtype='<f4').reshape(-1,7)
            reader._load_tfrecord_example=lambda *_:example
            for i in by_episode[key]:
                row=extra[i];t=row['start_index']
                item=reader._get_tfrecord_bridge_state_action(dict(file_path=shard,record_index=key[1],start_index=t))
                assert ' '.join(item[0].lower().split())==row['task']
                targets[i]=item[3].numpy();currents[i]=item[2].numpy();observed[i]=states[t,:6]
                sketch[i]=item[1].numpy().reshape(-1)[pixels]
                encoded=example.features.feature['steps/observation/image_0'].bytes_list.value[t]
                references[i]=dict(row=row,encoded_image_sha256=hashlib.sha256(encoded).hexdigest(),
                    normalized_image_sha256_local=hashlib.sha256(item[1].numpy().tobytes()).hexdigest())
        path=data_dir/shard
        with tf.io.TFRecordWriter(str(path)) as writer:
            for i in range(max(payloads)+1):writer.write(payloads.get(i,b''))
        assert path.stat().st_size<60*1024**2
        files[shard]=dict(sha256=sha(path),bytes=path.stat().st_size,
                         exported_record_indices=sorted(payloads),records=len(payloads))
        if (count+1)%40==0:print('EXPORT_NEW401',count+1,'/',len(grouped),'subset shards',flush=True)
    stats=read(ROOT/'cluster_packages/bridge_state_conditioning_v1/protocol.json')['state_features']
    np.savez_compressed(PACK/'new_numeric_reference.npz',targets=targets,currents=currents,observed=observed,
        state_features=state_features(observed,stats),pixel_indices=pixels,pixel_sketch=sketch)
    write(PACK/'new_input_reference.json',dict(rows=references,image_comparison='Encoded image SHA plus frozen preprocessing and96 fixed normalized pixels within1e-6; not a full1GiB image reference.',
        numeric_atol=1e-6,full_targets_and_current_observations=True,new_test_targets_read=False))
    assert sum(v['records'] for v in files.values())==401
    # Zip partition bounds use raw file sizes so every generated archive stays
    # comfortably below64MiB without guessing the compression ratio.
    groups=[];current=[];size=0
    for name,v in sorted(files.items()):
        if current and size+v['bytes']>60*1024**2:groups.append(current);current=[];size=0
        current.append(name);size+=v['bytes']
    if current:groups.append(current)
    archives=[]
    for i,names in enumerate(groups,1):
        path=PACK.parent/f'bridge_demo806_data_{i:02d}.zip'
        assert not path.exists()
        members=[PACK.name+'/data/'+name for name in names]
        with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
            for name,member in zip(names,members):z.write(data_dir/name,member)
        assert path.stat().st_size<64*1024**2
        with zipfile.ZipFile(path) as z:
            assert z.testzip() is None
            for name,member in zip(names,members):
                h=hashlib.sha256()
                with z.open(member) as stream:
                    for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
                assert h.hexdigest()==files[name]['sha256']
        archives.append(dict(file=path.name,sha256=sha(path),bytes=path.stat().st_size,members=members))
        # All removed files were generated above and verified in this archive.
        for name in names:
            target=(data_dir/name).resolve();assert target.is_relative_to(PACK.resolve())
            assert sha(target)==files[name]['sha256'];target.unlink()
        print('VERIFIED_DATA_PART',i,'/',len(groups),'MiB',round(path.stat().st_size/2**20,1),flush=True)
    write(PACK/'transfer_data.json',dict(files=files,archives=archives,only_added_training_records=True,
                                       future_test_records_exported=False,old_dataset_overwritten=False))
    baseline=read(BASE/'protocol.json');p=copy.deepcopy(baseline)
    control=read(ROOT/'reports/bridge-state-rep-fit-202463/real/report.json')
    base_report=read(ROOT/'reports/bridge-coverage-rep-fit-207558/report.json')
    old_reference=read(BASE/'episode_reference.json')
    write(PACK/'episode_reference.json',dict(episodes=old_reference['episodes']+audit['episodes']))
    p.update(date='2026-10-17',objective='Fixed8192-update comparison:405 to806 training demonstrations;401 audited additions; no new task or loss change.',
        baseline_protocol_sha256=sha(BASE/'protocol.json'),baseline_report_sha256=sha(ROOT/'reports/bridge-coverage-rep-fit-207558/report.json'),
        baseline_checkpoint_sha256=base_report['checkpoints'][0]['sha256'],
        old_cache_provenance_protocol_sha256=baseline['cache_provenance_protocol_sha256'],
        rows_sha256=sha(PACK/'rows.json'),schedule_file_sha256=sha(PACK/'schedule.npz'),
        episode_reference_sha256=sha(PACK/'episode_reference.json'),
        expected_initial_shared_sha256=control['initial_shared_sha256'],expected_augmented_initial_sha256=control['augmented_initial_sha256'],
        train_episodes=806,train_windows=3624,development_episodes=10,development_windows=40,
        total_input_windows=3664,new_training_episodes=401,new_training_windows=1788,
        additional_plan_sha256=sha(plan_path),new_source_audit_sha256=sha(audit_path),
        future_test_candidates_identity_only=plan['future_test_candidates_identity_and_metadata_only'],
        consumed208056_test_excluded=True,new_test_targets_exported=False,
        development_decision=plan['development_decision'],development_reference=plan['development_reference'],
        final_test_policy=plan['final_test_policy'],exposure_interpretation=plan['exposure_interpretation'],
        resource_gate=dict(gpu_reserved_gib=12,cache_budget_bytes=5*1024**3,estimated_formal_seconds=12600,
                           project_planning_limit_gib=200,minimum_planning_headroom_gib=15))
    for name in ('run_training_coverage.py','coverage_inputs.py','protocol.json','rows.json','schedule.npz','episode_reference.json'):
        p['inherited_sha256']['bridge_coverage_replication_v1/'+name]=sha(BASE/name)
    with np.load(PACK/'schedule.npz',allow_pickle=False) as z:
        p['schedule_tensor_sha256']=hashlib.sha256(z['batches'].tobytes()).hexdigest()
        draws=np.bincount(z['batches'].ravel(),minlength=3624)
    p['schedule'].update(minimum_window_draws=int(draws.min()),maximum_window_draws=int(draws.max()),mean_window_draws=float(draws.mean()))
    p['replication']={'mode':'new806pool_fixed_seed42','baseline_405_job':'207558','no_baseline_retraining':True}
    p['assessment']='Same old40 development windows; training metrics diagnostic only; no new25 or consumed13 evaluation in this runner.'
    p['screen']=dict(decision=plan['development_decision'],reference=plan['development_reference'])
    p['counts']=dict(train_windows=3624,train_episodes=806,development_windows=40,development_episodes=10)
    p['existing_data']='Existing405-episode subset plus isolated added401-record data directory; no overwrite or external download.'
    p['input_proof']='Reuse old415-episode geometry audit and complete1876-row cache; append audited401 episodes/1788 windows. Check old input identity, new raw payloads, full numeric targets/state and96 normalized image pixels.'
    p['local_preflight']='CPU-only export/source/AST checks. No actual model or GPU executed; cannot authorize formal.'
    p['cluster_preflight']='All816 episodes/3664 input windows; old4 frozen checkpoint replay, complete cache,16 operational updates, no candidate development or new-test scoring.'
    write(PACK/'protocol.json',p)
    spec=importlib.util.spec_from_file_location('size_adapter',PACK/'run_expansion.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    original=(BASE/'run_training_coverage.py').read_text()
    effective,changes=mod.adapted_source(original)
    old_tree=ast.parse(original);new_tree=ast.parse(effective)
    def nested(tree,name):return next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name)
    for name in ('advance','context'):
        assert ast.dump(nested(old_tree,name),include_attributes=False)==ast.dump(nested(new_tree,name),include_attributes=False)
    write(PACK/'adaptation_contract.json',dict(original_runner_sha256=sha(BASE/'run_training_coverage.py'),
        effective_runner_sha256=hashlib.sha256(effective.encode()).hexdigest(),literal_replacements=changes,
        permitted_changes=['Input helper import/path and806-pool dimensions','12GiB/12600s resource gates','Preflight-only old4 checkpoint replay before optimizer creation'],
        training_advance_ast_identical=True,context_ast_identical=True,original_files_modified=False,
        local_actual_model_or_gpu_executed=False))
    print('EXPORT_AND_PROTOCOL_COMPLETE',sum(x['bytes'] for x in archives),'compressed bytes',len(archives),'parts')


if __name__=='__main__':main()

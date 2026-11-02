"""CPU audit only NEW late train inputs and new development qualification.

No policy, CLIP feature encoding, training or final25 payload access. Existing
405 training records are reused; no E-drive dataset mutation. New10 development
records are exported selectively with placeholders preserving original indices.
"""
import os
os.environ.update(CUDA_VISIBLE_DEVICES="-1",USE_TF="0",TF_CPP_MIN_LOG_LEVEL="3")
import ast
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'cluster_packages/bridge_growth128_v1'
OUT=ROOT/'reports/bridge_prefix_input_audit_20261017'
sys.path.insert(0,str(BASE))
from dataset import prepare_image,relative_pose_action,euler_xyz_to_quaternion,bridge_reverse_scan_gripper
from bridge_partial_future_contract_v1 import build_numeric_sample
from bridge_prefix_input_contract_v1 import sha,records_at,episode_arrays,image_contract,GRID


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def ep(e):return e['shard'],e['record_index']


def official_check(payload,values):
    from tensorflow.core.example.example_pb2 import Example
    f=Example.FromString(payload).features.feature
    for name,expected in values.items():
        field=f[name]
        kind=field.WhichOneof('kind')
        actual=list(getattr(field,kind).value)
        if kind=='float_list':assert np.array_equal(np.asarray(actual,dtype='<f4'),np.asarray(expected,dtype='<f4'))
        else:assert actual==expected


def rotation(rpy):
    r,p,y=map(float,rpy);cr,sr,cp,sp,cy,sy=np.cos(r),np.sin(r),np.cos(p),np.sin(p),np.cos(y),np.sin(y)
    return np.array([[cy*cp,cy*sp*sr-sy*cr,cy*sp*cr+sy*sr],
        [sy*cp,sy*sp*sr+cy*cr,sy*sp*cr-cy*sr],[-sp,cp*sr,cp*cr]])


def main():
    assert not OUT.exists(),'Preserve successful or failed audit; do not silently overwrite'
    begin=time.monotonic();torch.set_num_threads(2);assert not torch.cuda.is_available()
    protocol_path=ROOT/'experiments/bridge_prefix_coverage_pair_v1/protocol.json'
    p=read(protocol_path)
    for path,h in p['source_sha256'].items():assert sha(ROOT/path)==h,path
    pool_path=ROOT/'training_cache/exports/bridge_expansion_training_v1/manifest.json'
    pool=read(pool_path)
    delta_rows=read(ROOT/'reports/bridge_partial_future_contract_20261017/candidate_rows.json')['train'][1836:]
    assert len(delta_rows)==1558
    final=set(map(ep,p['final_assessment']['metadata_only_episodes']))
    train={ep(e):e for e in pool['partitions']['train']}
    validation={ep(e):e for e in p['validation']['episodes']}
    assert not (set(train)|set(validation))&final
    state_source=ROOT/'cluster_packages/bridge_state_conditioning_v1/state_input_contract.py'
    fn=next(n for n in ast.parse(state_source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='state_features')
    ns={'np':np};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(state_source),'exec'),ns)
    stats=read(state_source.with_name('protocol.json'))['state_features']
    references={ep(e):e for e in read(ROOT/'cluster_packages/bridge_coverage_replication_v1/episode_reference.json')['episodes']}
    OUT.mkdir();(OUT/'validation_data').mkdir()
    report=dict(passed=False,protocol_sha256=sha(protocol_path),training=False,policy_loaded=False,
        GPU_used=False,CLIP_encoded=False,validation_scored=False,final25_payloads_read=False,
        old1836_redecoded=False,training_episodes_verified=0,added_windows_verified=0,
        validation_episodes_verified=0,validation_windows_verified=0,official_parser_checks=[],episodes=[],rows={},errors=[])
    def persist():report['elapsed_seconds']=time.monotonic()-begin;write(OUT/'report.json',report)
    def process_role(role,meta,group_rows,data_root):
        by_shard=defaultdict(list)
        for identity,e in meta.items():by_shard[e['shard']].append((identity,e))
        numeric=[];current=[];features=[];grids=[];raw=[];masks=[];ordered=[]
        row_groups=defaultdict(list)
        for i,row in enumerate(group_rows):row_groups[ep(row)].append((i,row))
        slots={}
        for shard,entries in sorted(by_shard.items()):
            path=data_root/shard
            source_hash=sha(path)
            if role=='train_delta':assert source_hash==pool['shard_sha256'][shard]
            selected={identity[1] for identity,_ in entries}
            exported={}
            for index,payload,values in records_at(path,selected):
                identity=shard,index;e=meta[identity]
                try:
                    if role=='validation' or len(report['official_parser_checks'])<2:
                        official_check(payload,values)
                        report['official_parser_checks'].append(dict(role=role,shard=shard,record_index=index))
                    digest=hashlib.sha256(payload).hexdigest()
                    if role=='train_delta':assert digest==references[identity]['payload_sha256']
                    states,actions,grip=episode_arrays(values,e,bridge_reverse_scan_gripper)
                    ep_check=dict(role=role,shard=shard,record_index=index,payload_sha256=digest,
                        source_file_sha256=source_hash,steps=e['steps'],windows=0,
                        state_sha256=hashlib.sha256(states.tobytes()).hexdigest(),
                        action_sha256=hashlib.sha256(actions.tobytes()).hexdigest())
                    for row_index,row in row_groups[identity]:
                        t=row['start_index'];sample=build_numeric_sample(states,grip,t,relative_pose_action,euler_xyz_to_quaternion)
                        length=int(sample['valid'].sum())
                        assert length==min(16,e['steps']-t-1)
                        if role=='train_delta':assert 1<=length<16
                        else:assert length==16
                        assert np.isfinite(sample['target'][:length]).all() and np.isnan(sample['target'][length:]).all()
                        # Independent placement of newly constructed partial labels.
                        world=states[t,:3].astype(float)+(rotation(states[t,3:6])@(sample['target'][:length,:3].astype(float)*.1).T).T
                        assert np.max(abs(world-states[t+1:t+length+1,:3]))<1e-6
                        assert np.array_equal(sample['target'][:length,7],2*grip[t:t+length]-1)
                        image,raw_check,grid=image_contract(values,t,prepare_image)
                        state=ns['state_features'](states[t:t+1,:6],stats)[0]
                        raw_check.update(row=row,valid_future_length=length,
                            raw_current_state_sha256=hashlib.sha256(states[t,:6].tobytes()).hexdigest())
                        slots[row_index]=(sample['target'],sample['current_gripper'],state,grid,raw_check,sample['valid'])
                        ep_check['windows']+=1
                    report['episodes'].append(ep_check)
                    report['training_episodes_verified' if role=='train_delta' else 'validation_episodes_verified']+=1
                    report['added_windows_verified' if role=='train_delta' else 'validation_windows_verified']+=ep_check['windows']
                    if role=='validation':exported[index]=payload
                except Exception as error:
                    report['errors'].append(dict(role=role,shard=shard,record_index=index,error=repr(error)))
            if role=='validation' and exported:
                import tensorflow as tf
                # Empty records preserve indices, not fabricated demonstration data.
                with tf.io.TFRecordWriter(str(OUT/'validation_data'/shard)) as writer:
                    for index in range(max(exported)+1):writer.write(exported.get(index,b''))
            if role=='train_delta' and report['training_episodes_verified']%50< len(entries):
                print('PREFIX_INPUT',report['training_episodes_verified'],'/405 episodes',report['added_windows_verified'],'/1558 new starts',flush=True)
            persist()
        if report['errors']:raise ValueError('Source/input eligibility failed; stop before further preparation')
        assert set(slots)==set(range(len(group_rows)))
        for i in range(len(group_rows)):
            target,cur,state,grid,raw_check,mask=slots[i]
            numeric.append(target);current.append(cur);features.append(state);grids.append(grid);raw.append(raw_check);masks.append(mask)
        np.savez_compressed(OUT/f'{role}_reference.npz',targets=np.stack(numeric),current=np.stack(current),
            state_features=np.stack(features),normalized_image_grid=np.stack(grids),validity=np.stack(masks))
        report['rows'][role]=raw
    try:
        validation_rows=[]
        for e in p['validation']['episodes']:
            for t in range(0,e['steps']-16,4):validation_rows.append(dict(shard=e['shard'],record_index=e['record_index'],
                start_index=t,task=e['instruction'],partition='new_development'))
        assert len(validation_rows)==50
        process_role('validation',validation,validation_rows,Path('E:/dataset/bridge_v2_0.0.1/0.0.1'))
        print('PREFIX_INPUT newdevelopment10/10 records qualified; no model scores',flush=True)
        process_role('train_delta',train,delta_rows,pool_path.parent/'data')
        assert report['added_windows_verified']==1558 and report['training_episodes_verified']==405
        assert report['validation_episodes_verified']==10 and report['validation_windows_verified']==50
        report.update(passed=True,normalized_grid_shape=[3,8,8],normalized_grid_indices=GRID.tolist(),
            numeric_tolerances=dict(targets=1e-6,state_features=1e-6,normalized_image_grid=1e-6,current=0.,rtol=0.),
            image_identity='Full encoded JPEG SHA and decoded RGB SHA exact; same frozen preprocessing source; compact fixed numeric grid, not full normalized-tensor comparison',
            versions=dict(torch=str(torch.__version__),numpy=np.__version__),
            files_sha256={x.name:sha(x) for x in OUT.glob('*_reference.npz')},
            validation_export_sha256={x.name:sha(x) for x in (OUT/'validation_data').iterdir()},
            source_sha256={str(x.relative_to(ROOT)):sha(x) for x in [Path(__file__),
                ROOT/'scripts/bridge_prefix_input_contract_v1.py',ROOT/'scripts/bridge_partial_future_contract_v1.py',
                BASE/'dataset.py',state_source,protocol_path]})
        write(OUT/'validation_rows.json',validation_rows)
        report['validation_rows_sha256']=sha(OUT/'validation_rows.json')
    except Exception as error:report['error']=repr(error);raise
    finally:persist()
    print(json.dumps({k:v for k,v in report.items() if k not in ('rows','episodes')},indent=2),flush=True)


if __name__=='__main__':main()

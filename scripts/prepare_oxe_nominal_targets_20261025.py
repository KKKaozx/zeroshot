"""New nominal-clock labels for the already qualified training pool only.

Reads each admitted raw payload once, without decoding images. Preserves the
old row-offset labels, frozen role assignments and original source files.
"""
from collections import Counter, defaultdict
from contextlib import ExitStack
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ['HF_HUB_OFFLINE']=os.environ['TRANSFORMERS_OFFLINE']='1'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(1024**2),b''):h.update(part)
    return h.hexdigest()


def write_json(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def extract(example,row):
    import numpy as np
    # Reuse the complete previously qualified source checks and causal inputs.
    from audit_bounded_oxe_train_targets_20261025 import source_arrays, values
    from dataset import (bridge_reverse_scan_gripper,rt1_relative_gripper_commands,
        euler_xyz_to_quaternion,rotation_vector_to_quaternion)
    from oxe_nominal_timing import normalize_quaternions
    p,features=source_arrays(example,row)
    f=example.features.feature;n=row['steps'];source=row['source']
    if source=='Bridge':
        state=values(f,'steps/observation/state',7,n)
        q=np.asarray([euler_xyz_to_quaternion(x) for x in state[:,3:6]])
        commands=2*bridge_reverse_scan_gripper(values(f,'steps/action',7,n)[:-1,6])-1
    elif source=='BC-Z':
        angles=values(f,'steps/observation/present/axis_angle',3,n)
        q=np.asarray([rotation_vector_to_quaternion(x) for x in angles])
        commands=np.where(values(f,'steps/action/future/target_close',10,n)[:-1,0]<.5,1.,-1.)
    elif source=='Fractal':
        q=values(f,'steps/observation/base_pose_tool_reached',7,n)[:,3:7]
        commands=rt1_relative_gripper_commands(values(f,'steps/action/gripper_closedness_action',1,n)[:,0])[:-1]
    else:return p,None,None,features
    return p,normalize_quaternions(q),np.asarray(commands,np.float32),features


def verify_chunk(result,source,p,q,commands,features,start):
    """New label checks, with direct old encoding only for equal-rate sources."""
    import numpy as np
    from cartesian_action_contract import encode_source_pose,encode_source_planar
    a=result['actions'];m=result['supervision_mask'];meta=result['layout'];count=meta['valid_targets']
    assert np.isfinite(a).all() and np.isin(m,[0,1]).all()
    assert np.array_equal(result['current_features'],features[start])
    assert not np.any(m[count:])
    if count<16:assert np.array_equal(a[count:],np.repeat(a[count-1:count],16-count,axis=0))
    if source=='Language Table':
        assert m[:count,:2].all() and not np.any(m[:,2:])
    else:
        assert m[:count].all() and np.isin(a[:,7],[-1,1]).all()
        assert np.max(np.abs(np.linalg.norm(a[:,3:7].astype(np.float64),axis=1)-1))<1e-6
        assert np.array_equal(a[:count,7],commands[np.asarray(meta['command_rows'][:count])])
    if source in ('Bridge','Language Table'):
        for j in range(count):
            target=start+j+1
            expected=(encode_source_planar(p[start],p[target])[0] if source=='Language Table' else
                encode_source_pose(p[start],q[start],p[target],q[target],commands[target-1]))
            if not np.allclose(a[j],expected,rtol=0,atol=1e-7):raise ValueError('5Hz label changed')
    elif source=='BC-Z':
        for j in range(count):
            target=start+2*(j+1)
            assert meta['left_rows'][j]==meta['right_rows'][j]==target
            assert meta['command_rows'][j]==target-1
            expected=encode_source_pose(p[start],q[start],p[target],q[target],commands[target-1])
            if not np.allclose(a[j],expected,rtol=0,atol=1e-7):raise ValueError('BC-Z nominal row/command changed')


def main():
    started=time.monotonic()
    import numpy as np
    import torch
    torch.set_num_threads(1)
    import tensorflow as tf
    tf.config.set_visible_devices([],'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(1);tf.config.threading.set_inter_op_parallelism_threads(1)
    from oxe_nominal_timing import chunk,layout,protocol,protocol_sha256
    numeric=ROOT/'reports/oxe_bounded_numeric_20261025'
    prior=json.loads((numeric/'report.json').read_text())
    eligible_path=numeric/'episode_eligibility.json'
    text_path=numeric/'training_original_instructions.jsonl'
    if file_sha(eligible_path)!=prior['episode_eligibility_sha256'] or file_sha(text_path)!=prior['training_original_instructions_sha256']:
        raise ValueError('Frozen eligibility/text changed')
    for name,expected in prior['parser_sha256'].items():
        if file_sha(ROOT/'code'/name)!=expected:raise ValueError('Qualified source parser changed: '+name)
    records_dir=ROOT/'reports/oxe_bounded_records_20261025'
    record_report=json.loads((records_dir/'report.json').read_text())
    for name,expected in [('episode_groups.json',record_report['group_manifest_sha256']),
                          ('episode_roles.json',prior['frozen_role_manifest_sha256'])]:
        if file_sha(records_dir/name)!=expected:raise ValueError('Frozen record roles changed')
    records=json.loads((records_dir/'episode_groups.json').read_text(encoding='utf-8'))['records']
    addresses={(r['source'],r['episode_key']):r for r in records}
    eligibility=json.loads(eligible_path.read_text(encoding='utf-8'))
    rows=[]
    for e in eligibility:
        if not e['numeric_eligible']:continue
        r=addresses[(e['source'],e['episode_key'])]
        if r['partition']!='train' or any(e[k]!=r[k] for k in ('steps','payload_sha256','group_id','stage_relative_path')):
            raise ValueError('Qualified train identity differs')
        rows.append(r)
    if len(rows)!=prior['eligible_train_records']:raise ValueError('Qualified pool count differs')
    plan_path=ROOT/'reports/oxe_bounded_pool_plan_20261025/candidate_shards.json'
    plan=json.loads(plan_path.read_text(encoding='utf-8'))
    paths={r['stage_relative_path']:Path(r['source_path']) for r in plan['files']}
    windows=[];omitted=Counter();episode_info=[];offsets=[0]
    for i,row in enumerate(rows):
        first=len(windows)
        for start in range(0,row['steps']-1,4):
            count=layout(row['source'],start,row['steps'])['valid_targets']
            if count:windows.append((i,start,count))
            else:omitted[row['source']]+=1
        episode_info.append(dict({k:row[k] for k in ('source','episode_key','stage_relative_path','record_index',
            'payload_sha256','payload_offset','payload_bytes','group_id','partition','steps')},
            row_offset=offsets[-1],window_first=first,window_stop=len(windows),
            observed_dimensions=['x','y'] if row['source']=='Language Table' else ['x','y','z','qx','qy','qz','qw','open_command']))
        offsets.append(offsets[-1]+row['steps'])
    out=ROOT/'reports/oxe_nominal_targets_20261025';out.mkdir(exist_ok=False)
    write_json(out/'protocol.json',protocol())
    write_json(out/'episodes.json',episode_info)
    np.save(out/'episode_offsets.npy',np.asarray(offsets,np.int64),allow_pickle=False)
    np.save(out/'window_rows.npy',np.asarray(windows,np.int64),allow_pickle=False)
    arrays={
        'recorded_pose':np.lib.format.open_memmap(out/'recorded_pose.npy',mode='w+',dtype=np.float64,shape=(offsets[-1],7)),
        'current_features':np.lib.format.open_memmap(out/'current_features.npy',mode='w+',dtype=np.float32,shape=(offsets[-1],2)),
        'interval_commands':np.lib.format.open_memmap(out/'interval_commands.npy',mode='w+',dtype=np.float32,shape=(offsets[-1],)),
        'actions':np.lib.format.open_memmap(out/'actions.npy',mode='w+',dtype=np.float32,shape=(len(windows),16,8)),
        'supervision_masks':np.lib.format.open_memmap(out/'supervision_masks.npy',mode='w+',dtype=np.uint8,shape=(len(windows),16,8)),
    }
    counts=defaultdict(Counter);maxima=defaultdict(float);probe_counts=Counter();payload_bytes=0
    with ExitStack() as stack:
        streams={}
        for i,row in enumerate(rows):
            path=paths[row['stage_relative_path']]
            if path not in streams:streams[path]=stack.enter_context(path.open('rb'))
            f=streams[path];f.seek(row['payload_offset']);raw=f.read(row['payload_bytes']);payload_bytes+=len(raw)
            if len(raw)!=row['payload_bytes'] or hashlib.sha256(raw).hexdigest()!=row['payload_sha256']:
                raise ValueError('Frozen payload changed: '+row['episode_key'])
            p,q,commands,features=extract(tf.train.Example.FromString(raw),row)
            lo,hi=offsets[i:i+2];source=row['source']
            arrays['recorded_pose'][lo:hi]=0.;arrays['recorded_pose'][lo:hi,6]=1.
            arrays['recorded_pose'][lo:hi,:p.shape[1]]=p
            if q is not None:arrays['recorded_pose'][lo:hi,3:7]=q
            arrays['current_features'][lo:hi]=features
            # Last row has no following source interval; terminal placeholder is invalid.
            arrays['interval_commands'][lo:hi]=0.
            if commands is not None:arrays['interval_commands'][lo:hi-1]=commands
            info=episode_info[i];first,last=info['window_first'],info['window_stop']
            counts[source]['episodes']+=1;counts[source]['source_rows']+=len(p)
            for wi in range(first,last):
                _,start,count=windows[wi]
                result=chunk(source,p,q,commands,features,start)
                if result['layout']['valid_targets']!=count:raise ValueError('Nominal layout count differs')
                a=result['actions'];m=result['supervision_mask']
                arrays['actions'][wi]=a;arrays['supervision_masks'][wi]=m
                if wi in (first,last-1):
                    verify_chunk(result,source,p,q,commands,features,start);probe_counts[source]+=1
                c=counts[source];c['windows']+=1;c['valid_targets']+=count
                c['unknown_current_gripper_windows']+=int(features[start,1]==0)
                position=a[:count,:p.shape[1]]
                maxima[source]=max(maxima[source],float(np.abs(position).max()))
                c['position_components']+=position.size
                c['position_components_outside_normalized3']+=int(np.count_nonzero(np.abs(position)>3))
                c['position_components_outside_candidate10']+=int(np.count_nonzero(np.abs(position)>10))
            if (i+1)%500==0 or i+1==len(rows):print('NOMINAL_TRAIN_TARGETS',i+1,'/',len(rows),'windows',last,flush=True)
    for arr in arrays.values():arr.flush()
    del arrays
    # Original texts are reused by exact input-row identity; no new language generation.
    window_lookup={(rows[e]['source'],rows[e]['episode_key'],s):i for i,(e,s,_) in enumerate(windows)}
    seen=set();text_rows=0
    with text_path.open(encoding='utf-8') as src,(out/'training_original_instructions.jsonl').open('x',encoding='utf-8') as dst:
        for line in src:
            entry=json.loads(line);key=(entry['source'],entry['episode_key'],entry['start_index'])
            wi=window_lookup.get(key)
            if wi is None:continue # Only predeclared zero-valid nominal tails are omitted.
            if wi in seen:raise ValueError('Duplicate original instruction identity')
            e=rows[windows[wi][0]]
            if entry['payload_sha256']!=e['payload_sha256'] or entry['group_id']!=e['group_id'] or entry['partition']!='train':
                raise ValueError('Original instruction role/content differs')
            seen.add(wi);text_rows+=1
            dst.write(json.dumps(dict(entry,window_index=wi),ensure_ascii=False,separators=(',',':'))+'\n')
    if len(seen)!=len(windows):raise ValueError('Incomplete original instruction coverage')
    artifacts={p.name:{'bytes':p.stat().st_size,'sha256':file_sha(p)} for p in sorted(out.iterdir()) if p.is_file()}
    global_max=max(maxima.values())
    core={'format':'oxe-nominal-training-target-artifacts-v1','scope':'frozen_training_roles_only',
        'protocol_sha256':protocol_sha256(),'artifacts':artifacts,
        'role_manifest_sha256':prior['frozen_role_manifest_sha256'],
        'eligibility_sha256':prior['episode_eligibility_sha256'],
        'candidate_plan_sha256':file_sha(plan_path),
        'code_sha256':{name:file_sha(ROOT/name) for name in ('code/oxe_nominal_timing.py',
            'scripts/prepare_oxe_nominal_targets_20261025.py','scripts/audit_bounded_oxe_train_targets_20261025.py')},
        'source_parser_sha256':prior['parser_sha256'],
        'terminal_interval_placeholder_valid':False,
        'planar_pose_placeholders_are_measured':False,
        'formal_training_ready':False}
    identity=hashlib.sha256(json.dumps(core,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    write_json(out/'manifest.json',dict(core,dataset_identity=identity))
    report={'complete':True,'dataset_identity':identity,'protocol_sha256':protocol_sha256(),
        'clock_is_nominal_assumption':True,'measured_timestamps_used':False,
        'eligible_training_episodes':len(rows),'raw_observation_rows':offsets[-1],
        'training_windows':len(windows),'old_training_windows':prior['eligible_train_windows'],
        'zero_valid_common_grid_tail_windows_omitted':dict(omitted),
        'by_source':dict(counts),'max_abs_normalized_position_component_by_source':dict(maxima),
        'max_abs_normalized_position_component':global_max,
        'ceil_train_max_only_bound_candidate':float(math.ceil(global_max)),
        'position_scale_physical_units_qualified':False,'position_bound_finally_frozen':False,
        'new_first_tail_numeric_checks_by_source':dict(probe_counts),
        'source_payload_bytes_read':payload_bytes,'images_decoded':0,
        'validation_test_quarantine_numeric_records_read':0,'roles_unchanged':file_sha(records_dir/'episode_roles.json')==prior['frozen_role_manifest_sha256'],
        'original_training_instruction_rows':text_rows,'new_artifact_bytes':sum(a['bytes'] for a in artifacts.values()),
        'source_files_and_old_labels_modified':False,'models_loaded':0,'training_updates':0,'new_cluster_jobs':0,
        'formal_training_ready':False,'elapsed_seconds':time.monotonic()-started}
    write_json(out/'report.json',report)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()

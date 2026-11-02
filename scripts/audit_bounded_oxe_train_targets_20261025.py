"""Train-role numeric eligibility after identity groups and roles are frozen.

Checks all used source pose/command arrays, plus first/tail windows through the
actual Dataset target parsers with pixel decoding stubbed. No heldout labels,
models, optimization, physical calibration or silent target repair.
"""
from collections import Counter,defaultdict
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ['HF_HUB_OFFLINE']=os.environ['TRANSFORMERS_OFFLINE']='1'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))


def sha(value):return hashlib.sha256(value).hexdigest()


def values(feature,key,width,n):
    import numpy as np
    field=feature[key];kind=field.WhichOneof('kind')
    array=np.asarray(getattr(field,kind).value,dtype=np.float32) if kind else np.asarray([])
    if array.size!=n*width:raise ValueError('source_field_shape:'+key)
    return array.reshape(n,width)


def source_arrays(example,row):
    import numpy as np
    from dataset import bridge_reverse_scan_gripper,rt1_relative_gripper_commands,euler_xyz_to_quaternion,rotation_vector_to_quaternion
    from oxe_source_semantics import command_features_before_rows
    f=example.features.feature;n=row['steps'];source=row['source']
    if source=='Bridge':
        state=values(f,'steps/observation/state',7,n)
        action=values(f,'steps/action',7,n)
        pos=state[:,:3];orientation=state[:,3:6]
        if not np.isfinite(pos).all() or not np.isfinite(orientation).all():raise ValueError('nonfinite_used_pose')
        for name,index in (('is_first',0),('is_last',n-1)):
            flags=list(f['steps/'+name].int64_list.value)
            if len(flags)!=n or any(x not in (0,1) for x in flags) or [i for i,x in enumerate(flags) if x]!=[index]:
                raise ValueError('bridge_invalid_RLDS_boundary')
        commands=action[:,6]
        if not np.isfinite(commands[:-1]).all():raise ValueError('nonfinite_used_command')
        labels=bridge_reverse_scan_gripper(commands[:-1])
        if not np.isin(labels,[0,1]).all():raise ValueError('bridge_unresolved_nonbinary_reverse_scan')
        features=command_features_before_rows('bridge',commands)
        quats=np.asarray([euler_xyz_to_quaternion(v) for v in orientation])
    elif source=='BC-Z':
        pos=values(f,'steps/observation/present/xyz',3,n)
        angles=values(f,'steps/observation/present/axis_angle',3,n)
        commands=values(f,'steps/action/future/target_close',10,n)
        # The current production parser validates the complete command table.
        if not np.isfinite(commands).all():raise ValueError('nonfinite_production_command_table')
        features=command_features_before_rows('bc_z',commands[:,0])
        if not np.isfinite(pos).all() or not np.isfinite(angles).all():raise ValueError('nonfinite_used_pose')
        quats=np.asarray([rotation_vector_to_quaternion(v) for v in angles])
    elif source=='Fractal':
        poses=values(f,'steps/observation/base_pose_tool_reached',7,n)
        commands=values(f,'steps/action/gripper_closedness_action',1,n)[:,0]
        pos=poses[:,:3];quats=poses[:,3:7]
        if not np.isfinite(poses).all():raise ValueError('nonfinite_used_pose')
        if np.any(np.linalg.norm(quats.astype(np.float64),axis=1)<1e-8):raise ValueError('missing_orientation_no_identity_substitution')
        rt1_relative_gripper_commands(commands) # Actual production target conversion.
        features=command_features_before_rows('fractal',commands)
    else:
        pos=values(f,'steps/observation/effector_translation',2,n)
        if not np.isfinite(pos).all():raise ValueError('nonfinite_used_planar_pose')
        features=np.zeros((n,2),np.float32);quats=None
    if quats is not None and (not np.isfinite(quats).all() or np.any(np.linalg.norm(quats.astype(np.float64),axis=1)<1e-8)):
        raise ValueError('invalid_converted_orientation')
    span=(pos.astype(np.float64).max(axis=0)-pos.astype(np.float64).min(axis=0))/.1
    if not np.isfinite(span).all() or np.any(span>np.finfo(np.float32).max):raise ValueError('position_target_float32_overflow')
    return pos,features


def instructions(example,row,starts):
    from dataset import UnifiedRobotDataset
    import numpy as np
    f=example.features.feature;n=row['steps']
    if row['source']=='Language Table':
        tokens=np.asarray(f['steps/observation/instruction'].int64_list.value,dtype=np.int64)
        if len(tokens)%n or not len(tokens):raise ValueError('instruction_row_shape')
        tokens=tokens.reshape(n,-1)
        result={}
        for i in starts:
            if np.any((tokens[i]<0)|(tokens[i]>255)):raise ValueError('instruction_nonbyte_token')
            text=bytes(int(v) for v in tokens[i] if v).decode('utf-8')
            result[i]=text.strip()
    else:
        key='steps/language_instruction' if row['source']=='Bridge' else 'steps/observation/natural_language_instruction'
        texts=f[key].bytes_list.value
        if len(texts)!=n:raise ValueError('instruction_row_shape')
        result={i:texts[i].decode('utf-8') for i in starts}
    if any(not t.strip() for t in result.values()):raise ValueError('empty_window_instruction')
    return result


def parser_probes(example,row,starts,features):
    import numpy as np
    import torch
    import dataset as module
    from dataset import UnifiedRobotDataset
    from oxe_window_index import OPTIONS,SOURCES
    data=object.__new__(UnifiedRobotDataset)
    for key,value in OPTIONS.items():setattr(data,key,value)
    data._language_table_groups={(row['shard'],row['record_index']):{'payload_sha256':row['payload_sha256']}}
    def cached(file_path,record_index,expected_payload_sha256=None):
        if record_index!=row['record_index'] or (expected_payload_sha256 is not None and expected_payload_sha256!=row['payload_sha256']):
            raise ValueError('Probe identity differs')
        return example
    data._load_tfrecord_example=cached
    fake=SimpleNamespace(convert=lambda mode:np.zeros((2,2,3),dtype=np.uint8))
    count=0
    with patch('PIL.Image.open',return_value=fake),patch.object(module,'prepare_image',return_value=torch.zeros(3,224,224)):
        for start in sorted({starts[0],starts[-1]}):
            data.samples=[dict(source=SOURCES[row['source']][1],file_path=row['stage_relative_path'],
                record_index=row['record_index'],start_index=start,trajectory_steps=row['steps'],payload_sha256=row['payload_sha256'])]
            text,pixels,current,target,mask=data[0]
            if not torch.isfinite(current).all() or not torch.isfinite(target).all():raise ValueError('nonfinite_main_parser_target')
            if not np.array_equal(current.numpy(),features[start]):raise ValueError('main_parser_command_phase_differs')
            valid=min(16,row['steps']-start-1)
            if mask[valid:].count_nonzero():raise ValueError('main_parser_padding_supervised')
            if row['source']=='Language Table':
                if mask[:,:2].sum()!=valid*2 or mask[:,2:].count_nonzero():raise ValueError('planar_missing_dimensions_supervised')
            else:
                if not torch.isin(target[:,7],torch.tensor([-1.,1.])).all():raise ValueError('nonbinary_main_gripper_target')
                if not torch.allclose(target[:,3:7].norm(dim=-1),torch.ones(16),rtol=0,atol=1e-5):raise ValueError('invalid_main_relative_quaternion')
            count+=1
    return count


def main():
    started=time.monotonic()
    out=ROOT/'reports/oxe_bounded_numeric_20261025';out.mkdir(exist_ok=False)
    import numpy as np
    import torch
    torch.set_num_threads(1)
    import tensorflow as tf
    tf.config.set_visible_devices([],'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(1);tf.config.threading.set_inter_op_parallelism_threads(1)
    groups_path=ROOT/'reports/oxe_bounded_records_20261025/episode_groups.json'
    roles_path=groups_path.with_name('episode_roles.json');prior=json.loads(groups_path.with_name('report.json').read_text())
    if sha(groups_path.read_bytes())!=prior['group_manifest_sha256'] or sha(roles_path.read_bytes())!=prior['role_manifest_sha256']:
        raise ValueError('Frozen candidate groups/roles changed')
    groups=json.loads(groups_path.read_text(encoding='utf-8'))
    plan=json.loads((ROOT/'reports/oxe_bounded_pool_plan_20261025/candidate_shards.json').read_text(encoding='utf-8'))
    paths={r['stage_relative_path']:Path(r['source_path']) for r in plan['files']}
    rows=[r for r in groups['records'] if r['partition']=='train']
    rules={'scope':'frozen_train_roles_only','minimum_episode_steps':10,'stride':4,'horizon':16,
        'arrays':'finite used reached positions/orientations, source-declared commands; reject missing orientation, do not repair',
        'instructions':'strict UTF8 and nonempty at every existing stride4 input start',
        'bridge_terminal_command':'excluded as in reverse_scan_valid_steps_v2',
        'pixel_decoding':'stubbed only during numeric parser probes; image bytes were hashed in identity phase',
        'probe_scope':'actual Dataset first and tail window per eligible train episode',
        'large_motion':'reported against existing normalized3 bound, never an exclusion criterion',
        'validation_test_quarantine_numeric_labels':'not read','source_units_or_time_guessed':False}
    (out/'rules.json').write_text(json.dumps(rules,indent=2)+'\n')
    results=[];counts=defaultdict(Counter);errors=defaultdict(Counter);probes=0;read_bytes=0
    text_path=out/'training_original_instructions.jsonl'
    with ExitStack() as stack:
        handles={};text_output=stack.enter_context(text_path.open('x',encoding='utf-8'))
        for number,row in enumerate(rows,1):
            source=row['source'];result={k:row[k] for k in ('source','episode_key','stage_relative_path','record_index','payload_sha256','group_id','partition','steps')}
            try:
                if row['steps']<10:raise ValueError('trajectory_too_short')
                if row['structure_errors']:raise ValueError('prior_structure:'+','.join(row['structure_errors']))
                path=paths[row['stage_relative_path']]
                if path not in handles:handles[path]=stack.enter_context(path.open('rb'))
                stream=handles[path];stream.seek(row['payload_offset']);raw=stream.read(row['payload_bytes']);read_bytes+=len(raw)
                if len(raw)!=row['payload_bytes'] or sha(raw)!=row['payload_sha256']:raise RuntimeError('Frozen train payload changed')
                example=tf.train.Example.FromString(raw)
                starts=list(range(0,row['steps']-1,4))
                texts=instructions(example,row,starts)
                pos,features=source_arrays(example,row)
                probes+=parser_probes(example,row,starts,features)
                start_array=np.asarray(starts);target_indices=start_array[:,None]+np.arange(1,17)
                valid=target_indices<row['steps'];target_indices=np.minimum(target_indices,row['steps']-1)
                delta=(pos[target_indices].astype(np.float64)-pos[start_array,None].astype(np.float64))/.1
                active=delta[valid]
                maximum=float(np.abs(active).max())
                outside=int(np.count_nonzero(np.abs(active)>3))
                unknown=int(np.count_nonzero(features[start_array,1]==0))
                result.update(numeric_eligible=True,windows=len(starts),valid_future_targets=int(valid.sum()),
                    unknown_current_gripper_windows=unknown,position_components=int(active.size),
                    position_components_outside_existing_normalized3_bound=outside,
                    maximum_absolute_normalized_displacement=maximum,
                    input_starts=starts,physical_units_or_timing_qualified=False)
                counts[source].update(records=1,windows=len(starts),valid_future_targets=int(valid.sum()),
                    unknown_current_gripper_windows=unknown,position_components=int(active.size),
                    position_components_outside_normalized3=outside)
                for start in starts:
                    text_output.write(json.dumps(dict(source=source,episode_key=row['episode_key'],
                        start_index=start,instruction=texts[start],payload_sha256=row['payload_sha256'],
                        group_id=row['group_id'],partition='train'),ensure_ascii=False,separators=(',',':'))+'\n')
            except (ValueError,UnicodeError,KeyError,IndexError) as exc:
                reason=type(exc).__name__+':'+str(exc)
                result.update(numeric_eligible=False,reason=reason)
                errors[source][reason]+=1
            results.append(result)
            if number%250==0 or number==len(rows):
                print('BOUNDED_TRAIN_NUMERIC',number,'/',len(rows),'eligible',sum(c['records'] for c in counts.values()),flush=True)
    (out/'episode_eligibility.json').write_text(json.dumps(results,separators=(',',':'),ensure_ascii=False)+'\n',encoding='utf-8')
    report={'complete':True,'frozen_train_records':len(rows),'eligible_train_records':sum(c['records'] for c in counts.values()),
        'eligible_train_windows':sum(c['windows'] for c in counts.values()),'by_source':dict(counts),'excluded_reasons':dict(errors),
        'actual_first_tail_loader_probes':probes,'train_payload_bytes_read':read_bytes,
        'source_validation_test_or_quarantine_numeric_records_read':0,'protected_bridge_records_interpreted':0,
        'pixels_decoded':False,'model_loaded':False,'training_updates':0,'physics_steps':0,'new_cluster_jobs':0,
        'numeric_eligibility_is_not_physical_unit_or_execution_qualification':True,
        'roles_unchanged':sha(roles_path.read_bytes())==prior['role_manifest_sha256'],
        'frozen_role_manifest_sha256':prior['role_manifest_sha256'],'formal_training_ready':False,
        'elapsed_seconds':time.monotonic()-started,
        'script_sha256':sha(Path(__file__).read_bytes()),
        'parser_sha256':{name:sha((ROOT/'code'/name).read_bytes()) for name in (
            'dataset.py','oxe_source_semantics.py','cartesian_action_contract.py')}}
    report['episode_eligibility_sha256']=sha((out/'episode_eligibility.json').read_bytes())
    report['training_original_instructions_sha256']=sha(text_path.read_bytes())
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()

"""Same declared numeric rules and nominal targets on frozen validation roles.

No training labels are rebuilt. Test/quarantine payloads and model outcomes are
not read. Qualification reasons are reported, never repaired silently.
"""
from collections import Counter, defaultdict
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))


def main():
    started=time.monotonic()
    import numpy as np
    import torch
    torch.set_num_threads(1)
    import tensorflow as tf
    tf.config.set_visible_devices([],'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(1);tf.config.threading.set_inter_op_parallelism_threads(1)
    from prepare_oxe_nominal_targets_20261025 import extract,verify_chunk,file_sha,write_json
    from audit_bounded_oxe_train_targets_20261025 import instructions
    from oxe_nominal_timing import chunk,layout,protocol,protocol_sha256
    records_dir=ROOT/'reports/oxe_bounded_records_20261025'
    train_dir=ROOT/'reports/oxe_nominal_targets_20261025'
    prior=json.loads((ROOT/'reports/oxe_bounded_numeric_20261025/report.json').read_text())
    record_report=json.loads((records_dir/'report.json').read_text())
    for name,expected in [('episode_groups.json',record_report['group_manifest_sha256']),
                          ('episode_roles.json',prior['frozen_role_manifest_sha256'])]:
        if file_sha(records_dir/name)!=expected:raise ValueError('Frozen roles changed')
    for name,expected in prior['parser_sha256'].items():
        if file_sha(ROOT/'code'/name)!=expected:raise ValueError('Previously qualified source parser changed')
    records=json.loads((records_dir/'episode_groups.json').read_text(encoding='utf-8'))['records']
    candidates=[r for r in records if r['partition']=='validation']
    train_groups={(r['source'],r['group_id']) for r in records if r['partition']=='train'}
    if any((r['source'],r['group_id']) in train_groups for r in candidates):raise ValueError('Frozen source group crosses train/validation')
    plan_path=ROOT/'reports/oxe_bounded_pool_plan_20261025/candidate_shards.json'
    paths={r['stage_relative_path']:Path(r['source_path']) for r in json.loads(plan_path.read_text(encoding='utf-8'))['files']}
    out=ROOT/'reports/oxe_nominal_validation_20261025';out.mkdir(exist_ok=False)
    rules=json.loads((ROOT/'reports/oxe_bounded_numeric_20261025/rules.json').read_text())
    rules.update(scope='frozen_validation_roles_only',training_rules_reused_without_new_selection=True,
        test_or_quarantine_payloads='not read',clock_is_nominal_assumption=True,
        zero_valid_new_grid_windows='omit exactly as training; no target extrapolation',
        model_or_outcome_based_exclusions=False)
    write_json(out/'qualification_rules.json',rules)
    offsets=[0];windows=[];episodes=[];poses=[];features_all=[];commands_all=[];actions=[];masks=[];texts=[]
    eligibility=[];counts=defaultdict(Counter);errors=defaultdict(Counter);omitted=Counter();probe_counts=Counter()
    maxima=defaultdict(float);payload_bytes=0;payload_records=0
    with ExitStack() as stack:
        streams={}
        for number,row in enumerate(candidates,1):
            source=row['source'];result={k:row[k] for k in ('source','episode_key','group_id','partition','steps','payload_sha256')}
            try:
                if row['steps']<10:raise ValueError('trajectory_too_short')
                if row['structure_errors']:raise ValueError('prior_structure:'+','.join(row['structure_errors']))
                path=paths[row['stage_relative_path']]
                if path not in streams:streams[path]=stack.enter_context(path.open('rb'))
                f=streams[path];f.seek(row['payload_offset']);raw=f.read(row['payload_bytes'])
                payload_bytes+=len(raw);payload_records+=1
                if len(raw)!=row['payload_bytes'] or hashlib.sha256(raw).hexdigest()!=row['payload_sha256']:
                    raise RuntimeError('Frozen validation payload changed')
                example=tf.train.Example.FromString(raw)
                starts=list(range(0,row['steps']-1,4))
                original=instructions(example,row,starts)
                p,q,commands,features=extract(example,row)
            except ValueError as exc:
                result.update(numeric_eligible=False,reason='ValueError:'+str(exc));errors[source][result['reason']]+=1
                eligibility.append(result);continue
            ei=len(episodes);first=len(windows);n=len(p)
            pose=np.zeros((n,7),np.float64);pose[:,6]=1.;pose[:,:p.shape[1]]=p
            if q is not None:pose[:,3:7]=q
            cmd=np.zeros(n,np.float32)
            if commands is not None:cmd[:-1]=commands
            poses.append(pose);features_all.append(features);commands_all.append(cmd)
            valid_starts=[s for s in starts if layout(source,s,n)['valid_targets']]
            omitted[source]+=len(starts)-len(valid_starts)
            for s in valid_starts:
                target=chunk(source,p,q,commands,features,s);count=target['layout']['valid_targets'];wi=len(windows)
                windows.append((ei,s,count));actions.append(target['actions']);masks.append(target['supervision_mask'].astype(np.uint8))
                if s in (valid_starts[0],valid_starts[-1]):
                    verify_chunk(target,source,p,q,commands,features,s);probe_counts[source]+=1
                texts.append(dict(source=source,episode_key=row['episode_key'],start_index=s,
                    instruction=original[s],payload_sha256=row['payload_sha256'],group_id=row['group_id'],
                    partition='validation',window_index=wi))
                c=counts[source];c['windows']+=1;c['valid_targets']+=count
                pos=target['actions'][:count,:p.shape[1]]
                maxima[source]=max(maxima[source],float(np.abs(pos).max()))
                c['position_components_outside_train_candidate8']+=int(np.count_nonzero(np.abs(pos)>8))
                c['unknown_current_gripper_windows']+=int(features[s,1]==0)
            episodes.append(dict({k:row[k] for k in ('source','episode_key','stage_relative_path','record_index',
                'payload_sha256','payload_offset','payload_bytes','group_id','partition','steps')},
                row_offset=offsets[-1],window_first=first,window_stop=len(windows),
                observed_dimensions=['x','y'] if source=='Language Table' else ['x','y','z','qx','qy','qz','qw','open_command']))
            offsets.append(offsets[-1]+n);counts[source]['episodes']+=1;counts[source]['source_rows']+=n
            result.update(numeric_eligible=True,nominal_windows=len(valid_starts));eligibility.append(result)
            if number%200==0 or number==len(candidates):print('NOMINAL_VALIDATION',number,'/',len(candidates),'eligible',len(episodes),flush=True)
    if not all(counts[s]['windows'] for s in ('Bridge','Language Table','BC-Z','Fractal')):
        raise ValueError('Validation is missing a source; do not silently resplit training')
    write_json(out/'protocol.json',protocol());write_json(out/'episodes.json',episodes)
    write_json(out/'episode_eligibility.json',eligibility)
    for name,value in [('episode_offsets',np.asarray(offsets,np.int64)),('window_rows',np.asarray(windows,np.int64)),
        ('recorded_pose',np.concatenate(poses)),('current_features',np.concatenate(features_all)),
        ('interval_commands',np.concatenate(commands_all)),('actions',np.stack(actions)),('supervision_masks',np.stack(masks))]:
        np.save(out/(name+'.npy'),value,allow_pickle=False)
    with (out/'validation_original_instructions.jsonl').open('x',encoding='utf-8') as f:
        for row in texts:f.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n')
    artifacts={p.name:{'bytes':p.stat().st_size,'sha256':file_sha(p)} for p in sorted(out.iterdir()) if p.is_file()}
    train_manifest=json.loads((train_dir/'manifest.json').read_text())
    core={'format':'oxe-nominal-validation-target-artifacts-v1','scope':'frozen_validation_roles_only',
        'protocol_sha256':protocol_sha256(),'artifacts':artifacts,
        'role_manifest_sha256':prior['frozen_role_manifest_sha256'],
        'train_target_identity_for_rule_matching':train_manifest['dataset_identity'],
        'candidate_plan_sha256':file_sha(plan_path),
        'code_sha256':{name:file_sha(ROOT/name) for name in ('code/oxe_nominal_timing.py',
            'scripts/prepare_oxe_nominal_validation_20261025.py','scripts/prepare_oxe_nominal_targets_20261025.py',
            'scripts/audit_bounded_oxe_train_targets_20261025.py')},
        'source_parser_sha256':prior['parser_sha256'],'terminal_interval_placeholder_valid':False,
        'planar_pose_placeholders_are_measured':False,'formal_training_ready':False}
    identity=hashlib.sha256(json.dumps(core,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    write_json(out/'manifest.json',dict(core,dataset_identity=identity))
    report={'complete':True,'dataset_identity':identity,'protocol_sha256':protocol_sha256(),
        'frozen_validation_candidates':len(candidates),'eligible_validation_episodes':len(episodes),
        'validation_windows':len(windows),'by_source':dict(counts),'excluded_reasons':dict(errors),
        'zero_valid_tail_windows_omitted':dict(omitted),'first_tail_target_checks_by_source':dict(probe_counts),
        'max_abs_normalized_position_by_source':dict(maxima),
        'training_position_bound_not_refit_from_validation':True,
        'source_payload_bytes_read':payload_bytes,'source_payload_records_read':payload_records,
        'test_or_quarantine_payload_records_read':0,'training_targets_rebuilt':False,'source_files_modified':False,
        'roles_unchanged':file_sha(records_dir/'episode_roles.json')==prior['frozen_role_manifest_sha256'],
        'images_decoded':0,'models_loaded':0,'optimizer_updates':0,'new_cluster_jobs':0,
        'generalization_verdict':None,'formal_training_ready':False,'elapsed_seconds':time.monotonic()-started}
    write_json(out/'report.json',report);print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()

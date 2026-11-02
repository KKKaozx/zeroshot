"""Index bounded candidates once, group origins/JPEGs, preserve frozen roles.

No pose/action numeric values or decoded pixels are inspected in this phase.
Protected Bridge aliases are excluded immediately after origin identification.
Per-shard outputs support resuming an interrupted scan without repeated I/O.
"""
from collections import Counter,defaultdict
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ['HF_HUB_OFFLINE']=os.environ['TRANSFORMERS_OFFLINE']='1'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from oxe_candidate_groups import candidate_components,check_anchor_rules

IMAGE_FIELDS={'Bridge':'steps/observation/image_0','Language Table':'steps/observation/rgb',
              'BC-Z':'steps/observation/image','Fractal':'steps/observation/image'}


def sha(value):return hashlib.sha256(value).hexdigest()


def main():
    started=time.monotonic();check_anchor_rules()
    plan_path=ROOT/'reports/oxe_bounded_pool_plan_20261025/candidate_shards.json'
    plan=json.loads(plan_path.read_text(encoding='utf-8'))
    output=ROOT/'reports/oxe_bounded_records_20261025';output.mkdir(exist_ok=True)
    shards=output/'shards';shards.mkdir(exist_ok=True)
    contract={'plan_sha256':sha(plan_path.read_bytes()),'script_sha256':sha(Path(__file__).read_bytes()),
        'group_code_sha256':sha((ROOT/'code/oxe_candidate_groups.py').read_bytes())}
    contract_path=output/'scan_contract.json'
    if contract_path.exists():
        if json.loads(contract_path.read_text())!=contract:raise ValueError('Scan plan/code changed; do not reuse prior shard outputs')
    else:contract_path.write_text(json.dumps(contract,indent=2)+'\n')
    old_group_path=ROOT/'reports/oxe_origin_groups_20261020/episode_groups.json'
    old_groups=json.loads(old_group_path.read_text())
    old_role_path=ROOT/'reports/oxe_episode_roles_20261020/episode_roles.json'
    old_roles=json.loads(old_role_path.read_text())
    known={(r['source'],r['shard'],r['record_index']):r for r in old_groups['episode_rows']}
    role_map={(r['source'],r['episode_key']):r for r in old_roles['role_assignments']}
    protected_plan=ROOT/'configs/bridge_demonstration_expansion_eligible_20261017.json'
    protected=json.loads(protected_plan.read_text())
    blocked={}
    for name in ('development','consumed_test_identity_only','future_test_candidates_identity_and_metadata_only'):
        for r in protected[name]:blocked[r['origin_file_path'],r['episode_id']]=name
    import tensorflow as tf
    tf.config.set_visible_devices([],'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    options=tf.data.Options();options.threading.private_threadpool_size=1
    options.threading.max_intra_op_parallelism=1;options.autotune.enabled=False
    rows=[];excluded=[];file_reports=[]
    for number,file in enumerate(plan['files'],1):
        source=file['source'];path=Path(file['source_path'])
        state=path.stat();cache=shards/(file['stage_relative_path'].replace('/','__')+'.json')
        if cache.exists():
            result=json.loads(cache.read_text())
            if result['source_file_stat']!=[state.st_size,state.st_mtime_ns]:raise ValueError('Previously indexed source file changed')
        else:
            if state.st_size!=file['bytes']:raise ValueError('Candidate file size changed')
            part=[];skipped=[];offset=12;payload_chain=hashlib.sha256();count=0
            reader=tf.data.TFRecordDataset([str(path)]).with_options(options)
            for index,record in enumerate(reader):
                raw=bytes(record.numpy());payload=sha(raw)
                payload_chain.update(len(raw).to_bytes(8,'little')+raw)
                example=tf.train.Example.FromString(raw);f=example.features.feature
                row=dict(source=source,episode_key=f'{path.name}::{index}',shard=path.name,
                    record_index=index,payload_sha256=payload,payload_offset=offset,payload_bytes=len(raw),
                    stage_relative_path=file['stage_relative_path'],origin_key=None,origin_id_available=False)
                offset+=len(raw)+16;count+=1
                if source=='Bridge':
                    origins=f['episode_metadata/file_path'].bytes_list.value
                    ids=f['episode_metadata/episode_id'].int64_list.value
                    if len(origins)!=1 or len(ids)!=1:raise ValueError('Bridge origin fields changed')
                    origin=origins[0].decode('utf-8');eid=int(ids[0])
                    row.update(origin_file_path=origin,origin_episode_id=eid,
                        origin_key=sha(json.dumps([origin,eid]).encode()),origin_id_available=True)
                    if (origin,eid) in blocked:
                        skipped.append(row|{'reason':'protected_bridge_origin_alias','protected_role':blocked[origin,eid]})
                        continue
                elif source=='BC-Z':
                    values=f['episode_id'].bytes_list.value
                    if len(values)==1 and values[0]:
                        row.update(origin_episode_id=values[0].decode('utf-8'),origin_key=sha(values[0]),origin_id_available=True)
                images=f[IMAGE_FIELDS[source]].bytes_list.value
                n=len(images);errors=[]
                if not n:errors.append('missing_image_rows')
                old=known.get((source,path.name,index))
                if old is not None:
                    if old['payload_sha256']!=payload:raise ValueError('Known cached record differs from full source')
                    frames=[h for h in old['encoded_frame_sha256']]
                    if len(frames)!=n:raise ValueError('Reused image-hash evidence differs')
                else:frames=[sha(image) for image in images if image]
                starts=list(range(0,max(0,n-1),4))
                if any(not images[i] for i in starts):errors.append('empty_candidate_input_image')
                if source=='Language Table':
                    tokens=f['steps/observation/instruction'].int64_list.value
                    if not n or not len(tokens) or len(tokens)%n:errors.append('instruction_row_shape')
                    else:
                        width=len(tokens)//n
                        if any(not any(tokens[i*width:(i+1)*width]) for i in starts):errors.append('empty_candidate_instruction')
                    language_rows=n if n and len(tokens)%n==0 else 0
                else:
                    name='steps/language_instruction' if source=='Bridge' else 'steps/observation/natural_language_instruction'
                    texts=f[name].bytes_list.value;language_rows=len(texts)
                    if language_rows!=n:errors.append('instruction_row_shape')
                    elif any(not texts[i].strip() for i in starts):errors.append('empty_candidate_instruction')
                row.update(steps=n,language_rows=language_rows,encoded_frame_sha256=frames,
                    structure_errors=errors,old_frame_evidence_reused=old is not None)
                part.append(row)
            if count!=file['metadata_expected_episodes'] or offset-12!=state.st_size:
                raise ValueError('Source count/container size differs from release metadata')
            after=path.stat()
            if (after.st_size,after.st_mtime_ns)!=(state.st_size,state.st_mtime_ns):raise ValueError('Source changed during scan')
            result={'source':source,'stage_relative_path':file['stage_relative_path'],
                'source_file_stat':[state.st_size,state.st_mtime_ns],'container_records':count,
                'length_delimited_payload_sha256':payload_chain.hexdigest(),
                'tfrecord_crc_checks_completed':True,'records':part,'protected_excluded':skipped}
            cache.write_text(json.dumps(result,separators=(',',':'),ensure_ascii=False)+'\n',encoding='utf-8')
        rows.extend(result['records']);excluded.extend(result['protected_excluded'])
        file_reports.append({k:v for k,v in result.items() if k not in ('records','protected_excluded')})
        if number%8==0 or number==len(plan['files']):
            print('BOUNDED_RECORD_INDEX',number,'/',len(plan['files']),'records',len(rows),flush=True)
    components,membership=candidate_components(rows,old_groups['episode_rows'],role_map)
    # Preserve old role anchors; only genuinely new, unanchored groups are split.
    from train import split_by_trajectory
    fresh=[c for c in components if c['partition']=='unassigned']
    class NewGroups:
        def __len__(self):return len(fresh)
        def group_key(self,i):return fresh[i]['source']+':'+fresh[i]['group_id']
    for role,indices in zip(('train','validation','test'),split_by_trajectory(NewGroups(),42)):
        for i in indices:fresh[i]['partition']=role
    for row in rows:
        component=membership[row['source'],row['episode_key']]
        row['group_id']=component['group_id'];row['partition']=component['partition']
    roles=[{k:r[k] for k in ('source','episode_key','shard','record_index','payload_sha256','group_id','partition')} for r in rows]
    role_payload={'format':'bounded-oxe-frozen-candidate-roles-v1','role_seed':42,'assignments':roles,
        'previous_role_manifest_sha256':sha(old_role_path.read_bytes()),
        'previous_role_anchors_preserved':True,'conflicts_quarantined_not_reassigned':True,
        'split_function_sha256':sha(inspect.getsource(split_by_trajectory).encode()),
        'shared_across_training_seeds_and_all_thirteen_conditions':True,'training_ready':False}
    (output/'episode_roles.json').write_text(json.dumps(role_payload,separators=(',',':'))+'\n')
    (output/'episode_groups.json').write_text(json.dumps({'components':components,'records':rows,
        'protected_excluded':excluded,'training_ready':False},separators=(',',':'),ensure_ascii=False)+'\n',encoding='utf-8')
    (output/'file_checks.json').write_text(json.dumps(file_reports,indent=2)+'\n')
    summary=defaultdict(Counter)
    for row in rows:summary[row['source']][row['partition']]+=1
    report={'complete':True,'candidate_files':len(file_reports),'container_records':sum(f['container_records'] for f in file_reports),
        'candidate_records':len(rows),'protected_bridge_aliases_excluded':len(excluded),
        'candidate_content_groups':len(components),'by_source_role_records':dict(summary),
        'old_anchored_groups':sum(bool(c['previous_role_anchors']) for c in components),
        'conflicting_old_role_groups_quarantined':sum(c['partition'].startswith('quarantine') for c in components),
        'structure_error_records':sum(bool(r['structure_errors']) for r in rows),
        'numeric_target_values_read':False,'images_decoded':False,'model_loaded':False,
        'training_updates':0,'new_cluster_jobs':0,'training_ready':False,
        'role_manifest_sha256':sha((output/'episode_roles.json').read_bytes()),
        'group_manifest_sha256':sha((output/'episode_groups.json').read_bytes()),
        'candidate_plan_sha256':contract['plan_sha256'],'elapsed_seconds':time.monotonic()-started}
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()

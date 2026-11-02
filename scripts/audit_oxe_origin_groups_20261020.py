"""Remaining candidate origins/shared JPEGs; reuse the completed LT audit.

No pose/action decoding, image decoding, model, fit, partition assignment or
final benchmark access. Known protected Bridge aliases are checked before any
observation/label fields are interpreted. TFRecord reader checks record CRCs.
"""
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'reports/oxe_origin_groups_20261020'


def sha(data): return hashlib.sha256(data).hexdigest()


def group_rows(rows):
    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    owners, shared = {}, set()
    for i,row in enumerate(rows):
        tokens = [('jpeg',h) for h in set(row['encoded_frame_sha256'])]
        if row.get('origin_key'):
            tokens.append(('origin',row['origin_key']))
        for kind,token in tokens:
            key = row['source'],kind,token
            if key in owners:
                parent[find(i)] = find(owners[key]); shared.add(key)
            else: owners[key] = i
    components = defaultdict(list)
    for i,row in enumerate(rows): components[find(i)].append(row)
    groups=[]
    for values in components.values():
        members=sorted(row['episode_key'] for row in values)
        source=values[0]['source']
        identity=source+'\n'+'\n'.join(members)
        group={'source':source,'group_id':'oxe-origin-content-'+sha(identity.encode()),
               'episode_keys':members,'episodes':len(members)}
        groups.append(group)
        for row in values: row['group_id']=group['group_id']
    groups.sort(key=lambda g:(g['source'],g['episode_keys']))
    # No recorded identical origin or JPEG token may cross components.
    verification={}
    for row in rows:
        tokens=[('jpeg',h) for h in set(row['encoded_frame_sha256'])]
        if row.get('origin_key'):tokens.append(('origin',row['origin_key']))
        for kind,token in tokens:
            key=row['source'],kind,token
            assert verification.setdefault(key,row['group_id'])==row['group_id']
    return groups,Counter(source for source,kind,_ in shared if kind=='jpeg')


def self_check():
    fixture=[{'source':'A','episode_key':'a','origin_key':'same','encoded_frame_sha256':['1']},
             {'source':'A','episode_key':'b','origin_key':'same','encoded_frame_sha256':['2']},
             {'source':'A','episode_key':'c','origin_key':None,'encoded_frame_sha256':['2']},
             {'source':'B','episode_key':'d','origin_key':'same','encoded_frame_sha256':['2']}]
    a,_=group_rows(fixture)
    assert [g['episode_keys'] for g in a]==[['a','b','c'],['d']]
    assert group_rows(list(reversed(fixture)))[0]==a


def main():
    start=time.perf_counter();self_check();OUT.mkdir(exist_ok=False)
    manifest_path=ROOT/'reports/oxe_baseline_candidates_20261020/episode_candidates.json'
    inventory_path=ROOT/'reports/oxe_baseline_candidates_20261020/report.json'
    manifest=json.loads(manifest_path.read_text())
    inventory=json.loads(inventory_path.read_text())
    plan_path=ROOT/'configs/bridge_demonstration_expansion_eligible_20261017.json'
    plan=json.loads(plan_path.read_text())
    blocked={}
    for partition in ('development','consumed_test_identity_only','future_test_candidates_identity_and_metadata_only'):
        for row in plan[partition]:blocked[row['origin_file_path'],row['episode_id']]=partition
    rows=[];excluded=[];file_checks=[]
    lt_path=ROOT/'reports/language_table_overlap_20261020/episode_groups.json'
    lt_report=json.loads((lt_path.parent/'report.json').read_text())
    assert sha(lt_path.read_bytes())==lt_report['group_manifest_sha256']
    lt=json.loads(lt_path.read_text())
    for original in lt['episode_rows']:
        rows.append({**original,'source':'Language Table','origin_key':None,
                     'origin_id_available':False,'identity_eligibility':'previous_LT_structure_audit_reused'})
    image_fields={'Bridge':'steps/observation/image_0','BC-Z':'steps/observation/image','Fractal':'steps/observation/image'}
    for source in ('Bridge','BC-Z','Fractal'):
        expected={(r['shard'],r['record_index']):r for r in manifest['episode_rows'] if r['source']==source}
        for file in inventory['sources'][source]['cached_files']:
            path=Path(file['path']);assert path.stat().st_size==file['bytes']
            offset=12;count=0;used=0;payloads=hashlib.sha256()
            for index,raw in enumerate(tf.data.TFRecordDataset([str(path)])):
                data=bytes(raw.numpy());payloads.update(len(data).to_bytes(8,'little')+data)
                key=path.name,index
                old_offset=offset;offset+=len(data)+16;count+=1
                if key not in expected: continue  # Previous protected rows: no Example interpretation.
                candidate=expected[key]
                assert candidate['payload_offset']==old_offset and candidate['payload_bytes']==len(data)
                assert candidate['partition']=='unassigned_candidate'
                example=tf.train.Example();example.ParseFromString(data);f=example.features.feature
                row={'source':source,'episode_key':f'{path.name}::{index}',
                     'shard':path.name,'record_index':index,'payload_sha256':sha(data),
                     'partition':'unassigned_candidate','origin_key':None,'origin_id_available':False}
                if source=='Bridge':
                    path_values=f['episode_metadata/file_path'].bytes_list.value
                    id_values=f['episode_metadata/episode_id'].int64_list.value
                    assert len(path_values)==len(id_values)==1
                    origin=path_values[0].decode('utf-8');eid=int(id_values[0])
                    if (origin,eid) in blocked:
                        excluded.append({**row,'reason':'protected_bridge_origin_alias',
                                         'protected_partition':blocked[origin,eid]})
                        continue  # In particular: no pose, language or images of final aliases.
                    row.update(origin_file_path=origin,origin_episode_id=eid,
                               origin_key=sha(json.dumps([origin,eid]).encode()),origin_id_available=True)
                elif source=='BC-Z':
                    values=f['episode_id'].bytes_list.value
                    if len(values)==1 and values[0]:
                        row.update(origin_episode_id=values[0].decode('utf-8'),
                                   origin_key=sha(values[0]),origin_id_available=True)
                images=f[image_fields[source]].bytes_list.value
                text_key='steps/language_instruction' if source=='Bridge' else 'steps/observation/natural_language_instruction'
                texts=f[text_key].bytes_list.value
                row.update(steps=len(images),language_rows=len(texts),
                           nonempty_language_rows=sum(bool(t.strip()) for t in texts),
                           empty_encoded_images=sum(not image for image in images),
                           encoded_frame_sha256=[sha(image) for image in images if image],
                           identity_eligibility='not_a_numeric_or_window_eligibility_verdict')
                rows.append(row);used+=1
            assert count==file['episodes'] and offset-12==file['bytes']
            file_checks.append({'source':source,'shard':path.name,'container_records':count,
                                'candidate_records_used':used,'tfrecord_reader_crc_checks_completed':True,
                                'length_delimited_payload_sha256':payloads.hexdigest()})
        print('OXE_ORIGIN_GROUP',source,'complete',flush=True)
    groups,shared=group_rows(rows)
    # The reused LT component memberships must remain exactly unchanged.
    old_members={tuple(g['episode_keys']) for g in lt['groups']}
    assert {tuple(g['episode_keys']) for g in groups if g['source']=='Language Table'}==old_members
    summaries={}
    for source in ('Bridge','Language Table','BC-Z','Fractal'):
        subset=[r for r in rows if r['source']==source]
        components=[g for g in groups if g['source']==source]
        origins=Counter(r['origin_key'] for r in subset if r['origin_key'])
        summaries[source]={'records':len(subset),'groups':len(components),
                           'multi_record_groups':sum(g['episodes']>1 for g in components),
                           'records_in_multi_record_groups':sum(g['episodes'] for g in components if g['episodes']>1),
                           'max_group_records':max(g['episodes'] for g in components),
                           'distinct_origin_ids':len(origins),
                           'records_with_origin_id':sum(r['origin_id_available'] for r in subset),
                           'duplicate_origin_id_values':sum(v>1 for v in origins.values()),
                           'shared_encoded_frame_types':shared[source]}
    result={'format':'oxe-origin-and-encoded-frame-groups-v1','episode_rows':rows,'groups':groups,
            'excluded_origin_aliases':excluded,'partitions_assigned':False,'training_ready':False}
    output=OUT/'episode_groups.json';output.write_text(json.dumps(result,indent=2)+'\n')
    report={'complete':True,'sources':summaries,'records':len(rows),'groups':len(groups),
            'protected_bridge_origin_aliases_excluded':len(excluded),
            'LT_full_audit_reused_not_repeated':True,'LT_group_memberships_unchanged':True,
            'source_scoped_shared_origin_and_JPEG_never_cross_groups':True,
            'grouping_fixture_transitive_source_scoped_and_order_invariant':True,
            'candidate_manifest_sha256':sha(manifest_path.read_bytes()),
            'protected_plan_sha256':sha(plan_path.read_bytes()),'prior_LT_manifest_sha256':sha(lt_path.read_bytes()),
            'group_manifest_sha256':sha(output.read_bytes()),'script_sha256':sha(Path(__file__).read_bytes()),
            'file_checks':file_checks,'decoded_images':0,'pose_or_action_arrays_interpreted':False,
            'final_targets_interpreted':False,'models_loaded':0,'training_updates':0,
            'partitions_assigned':False,'physical_origin_independence_proven':False,
            'training_ready':False,'elapsed_seconds':time.perf_counter()-start,
            'limitations':['Encoded-frame equality misses recompressed copies; a static identical frame may overmerge',
                           'Bridge origin is exact stored file_path+episode_id; path aliases still need mapping',
                           'BC-Z episode_id semantics and persistence across re-export need upstream binding',
                           'LT and Fractal expose no physical session/episode identifiers in local schema',
                           'No numeric label validity, source physical units/time or fixed split verdict']}
    (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('file_checks','limitations')},indent=2))


if __name__=='__main__':main()

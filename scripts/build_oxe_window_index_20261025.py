"""Enumerate actual loader windows from already audited metadata and frozen roles.

No source payloads/images/labels are reread. This engineering index remains
prepare-only while physical source contracts and numeric eligibility are open.
"""
from collections import Counter,defaultdict
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from oxe_window_index import FORMAT,OPTIONS,SOURCES,PARSER_FILES,file_sha,seal,validate_manifest,splits


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def build(root):
    role_path=root/'reports/oxe_episode_roles_20261020/episode_roles.json'
    group_path=root/'reports/oxe_origin_groups_20261020/episode_groups.json'
    candidates_path=root/'reports/oxe_baseline_candidates_20261020/episode_candidates.json'
    inventory_path=candidates_path.with_name('report.json')
    role_report=read(role_path.with_name('report.json'))
    group_report=read(group_path.with_name('report.json'))
    inventory=read(inventory_path);roles=read(role_path);groups=read(group_path);candidates=read(candidates_path)
    assert file_sha(role_path)==role_report['frozen_role_manifest_sha256']
    assert file_sha(group_path)==group_report['group_manifest_sha256']==roles['origin_group_manifest_sha256']
    assert file_sha(candidates_path)==inventory['candidate_manifest_sha256']==roles['candidate_manifest_sha256']
    origin={(r['source'],r['shard'],r['record_index']):r for r in groups['episode_rows']}
    assigned={(r['source'],r['shard'],r['record_index']):r for r in roles['role_assignments']}
    assert len(origin)==len(assigned)==5158
    episodes=[];excluded=[];windows=[]
    for candidate in candidates['episode_rows']:
        key=(candidate['source'],candidate['shard'],candidate['record_index'])
        if key not in assigned:raise ValueError('Unassigned or extra candidate record')
        audit,role=origin[key],assigned[key]
        assert audit['payload_sha256']==role['payload_sha256'] and audit['group_id']==role['group_id']
        steps=audit['steps']
        if candidate['source']=='Language Table':
            if audit['errors']:raise ValueError('Prior LT audit failed')
            # Existing LT audit admitted every decoded instruction and retained
            # its exact16-start list; no new language or image interpretation.
            if audit['full16_candidate_starts_stride4']!=list(range(0,max(0,steps-16),4)):
                raise ValueError('LT original full16 indexing evidence differs')
            nonempty=steps
        else:
            nonempty=audit['nonempty_language_rows']
            if audit['language_rows']!=steps or nonempty not in (0,steps):
                raise ValueError('Need per-row language evidence for mixed/changed language availability')
        reason=('trajectory_too_short' if steps<10 else 'without_language' if nonempty==0 else None)
        if reason:
            excluded.append(dict(source=key[0],shard=key[1],record_index=key[2],
                partition=role['partition'],payload_sha256=role['payload_sha256'],reason=reason,steps=steps))
            continue
        subdir,schema=SOURCES[key[0]]
        index=len(episodes)
        episodes.append(dict(source=key[0],schema=schema,shard=key[1],record_index=key[2],
            relative_file=subdir+'/'+key[1],steps=steps,partition=role['partition'],
            payload_sha256=role['payload_sha256'],group_id=role['group_id'],
            language_eligibility='all_existing_observation_rows_nonempty_in_previous_audit'))
        for start in range(0,steps-1,4):
            windows.append(dict(episode_index=index,start_index=start,valid_targets=min(16,steps-start-1)))
    source_files=[]
    for source,info in inventory['sources'].items():
        for file in info['cached_files']:
            source_files.append(dict(source=source,relative_file=SOURCES[source][0]+'/'+file['shard'],
                bytes=file['bytes'],cached_episode_count=file['episodes'],header_stream_sha256=file['header_stream_sha256']))
    lt_path=root/'reports/language_table_overlap_20261020/episode_groups.json'
    core=dict(format=FORMAT,candidate_pool_only=True,training_ready=False,
        sample_options=OPTIONS,parser_files_sha256={n:file_sha(root/'code'/n) for n in PARSER_FILES},
        provenance=dict(candidate_manifest_sha256=file_sha(candidates_path),origin_group_manifest_sha256=file_sha(group_path),
            frozen_role_manifest_sha256=file_sha(role_path),language_table_groups_sha256=file_sha(lt_path),
            source_inventory_sha256=file_sha(inventory_path),
            units_time_evidence_sha256=file_sha(root/'reports/oxe_units_time_evidence_20261020/report.json')),
        source_files=source_files,episodes=episodes,windows=windows,excluded_episodes=excluded,
        physical_admission=dict(source_units_to_meters_resolved=False,export_target_timing_resolved=False,
            cross_robot_axes_tcp_mapping_qualified=False,numeric_target_pool_eligibility_checked=False),
        label_contract_note='Observed reached rows t+1..t+16; preceding observation command slot0 for3D; '
            'LT onlyXY; tail repeats carry zero supervision. Native command slots not used. '
            'Legacy scale0.1 is a declared arithmetic divisor, not proof of units. All target times remain unknown.')
    manifest=seal(core);validate_manifest(manifest,root/'code')
    partitions=splits(manifest)
    by_source=defaultdict(Counter);full=defaultdict(Counter)
    for i,row in enumerate(windows):
        ep=episodes[row['episode_index']]
        by_source[ep['source']][ep['partition']]+=1
        full[ep['source']]['full16' if row['valid_targets']==16 else 'partial_tail']+=1
    report=dict(complete=True,window_index_created=True,main_dataset_consumable=True,
        formal_training_admitted=False,source_payloads_reread=False,images_or_numeric_labels_decoded=False,
        candidate_episodes=5158,indexed_episodes=len(episodes),excluded_episodes=len(excluded),windows=len(windows),
        original_frozen_episode_roles_unchanged=True,role_policy='Filter within fixed roles; no reassignments across seeds/conditions',
        by_source_windows=dict(by_source),by_role_windows={k:len(v) for k,v in partitions.items()},
        by_source_horizon_counts=dict(full),excluded_reasons=dict(Counter((r['source']+':'+r['reason']) for r in excluded)),
        dataset_identity=manifest['dataset_identity'],target_protocol_sha256=manifest['target_binding']['sha256'],
        source_units_time_or_tcp_guessed=False,model_loaded=False,training_updates=0,simulation_steps=0,
        scope='Portable engineering window index from previously verified cached metadata; prepare-only, not numeric/physical training admission')
    return manifest,report


def main():
    manifest,report=build(ROOT)
    folder=ROOT/'reports/oxe_window_index_20261025';folder.mkdir(exist_ok=True)
    path=folder/'window_index.json'
    path.write_text(json.dumps(manifest,separators=(',',':'),ensure_ascii=False)+'\n',encoding='utf-8')
    report['window_manifest_sha256']=file_sha(path)
    (folder/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()

"""A reproducible storage-bounded candidate pool, not an admitted training set.

Uses only already-inventoried official train filenames and release metadata.
No copying, target/visual reads, split changes, or model-dependent selection.
"""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
ALIASES={'Bridge':'bridge_v2','Language Table':'language_table_real','BC-Z':'bc_z','Fractal':'rt1_fractal'}
SEED=20261025
PER_SOURCE_BYTES=4*1024**3


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()


def main():
    source_path=ROOT/'reports/oxe_full_source_availability_20261025/report.json'
    inventory=json.loads(source_path.read_text(encoding='utf-8'))
    if not inventory['all_train_filename_sets_present']:raise ValueError('Resolve missing source train files first')
    selected=[];summary={}
    for source,alias in ALIASES.items():
        info=inventory['sources'][source]
        if not info['metadata_matches_existing_audited_release']:raise ValueError('Release identity changed: '+source)
        release=Path(info['release_directory'])
        meta=json.loads((release/'dataset_info.json').read_text(encoding='utf-8'))
        split=next(s for s in meta['splits'] if s['name']=='train')
        rows=info['splits']['train']['files']
        rank=lambda row:hashlib.sha256(f'bounded-oxe-pool-v1:{SEED}:{source}:{row["relative_path"]}'.encode()).hexdigest()
        used=episodes=0;count=0
        for row in sorted(rows,key=rank):
            size=row['bytes']
            if used+size>PER_SOURCE_BYTES:continue
            index=row['index'];used+=size;count+=1;episodes+=int(split['shardLengths'][index])
            path=(release/row['relative_path']).resolve()
            if path.parent!=release.resolve() or path.stat().st_size!=size:
                raise ValueError('Inventoried file path/size changed')
            selected.append(dict(source=source,source_dataset=info['dataset'],version=info['version'],
                source_path=str(path),stage_relative_path=alias+'/'+path.name,shard_index=index,
                bytes=size,metadata_expected_episodes=int(split['shardLengths'][index]),
                selection_hash=rank(row),payload_sha256_verified=False))
        if count<1:raise ValueError('Empty source budget')
        summary[source]={'shards':count,'bytes':used,'gib':used/1024**3,
            'metadata_expected_episodes':episodes,'raw_byte_cap':PER_SOURCE_BYTES}
    portable=[{k:v for k,v in r.items() if k!='source_path'} for r in selected]
    plan={'format':'bounded-oxe-candidate-shards-v1','scope':'candidate_only_not_formal_training',
        'selection_seed':SEED,'ranking':'SHA256 of fixed protocol seed, source and train filename; greedy fit within each4GiB cap',
        'raw_byte_cap':4*PER_SOURCE_BYTES,'selected_bytes':sum(r['bytes'] for r in selected),
        'sources':summary,'files':selected,'portable_candidate_identity':digest(portable),
        'source_inventory_sha256':hashlib.sha256(source_path.read_bytes()).hexdigest(),
        'shared_by_all_thirteen_runs_if_later_admitted':True,
        'all_episodes_and_eligible_frames_within_selected_shards_to_be_considered':True,
        'episode_content_grouping_splits_and_target_qualification_complete':False,
        'physical_units_and_execution_mapping_resolved':False,
        'full_oxe_claim_allowed':False,'data_copied':False,'training_started':False,
        'selection_used_model_errors_task_success_or_heldout_labels':False}
    out=ROOT/'reports/oxe_bounded_pool_plan_20261025';out.mkdir(exist_ok=False)
    (out/'candidate_shards.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    # This is a planning envelope; the cluster allocation must be remeasured.
    budget={'project_limit_bytes':200_000_000_000,'last_known_project_allocated_bytes':152_091_529_216,
        'last_known_usage_is_live_measurement':False,'candidate_raw_bytes':plan['selected_bytes'],
        'proposed_combined_checkpoint_and_logs_envelope_bytes':24_000_000_000,
        'minimum_remaining_headroom_bytes':6_000_000_000,
        'frozen_models_and_environments_assumed_already_in_last_known_usage':True,
        'no_dense_all_frame_dino_token_cache_planned':True,
        'checkpoint_policy':'Latest resumable checkpoint per run; no accumulating every-save copies; exact cost still measured on formal config',
        'full_oxe_raw_train_bytes':inventory['train_file_bytes_present'],
        'formal_quota_admission':False}
    budget['illustrative_project_total_bytes']=sum(budget[k] for k in (
        'last_known_project_allocated_bytes','candidate_raw_bytes',
        'proposed_combined_checkpoint_and_logs_envelope_bytes','minimum_remaining_headroom_bytes'))
    budget['fits_only_under_last_known_usage_assumption']=budget['illustrative_project_total_bytes']<=budget['project_limit_bytes']
    (out/'storage_envelope.json').write_text(json.dumps(budget,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'sources':summary,'selected_gib':plan['selected_bytes']/1024**3,
        'shards':len(selected),'metadata_expected_episodes':sum(r['metadata_expected_episodes'] for r in selected),
        'illustrative_total_under_last_known_usage':budget['illustrative_project_total_bytes'],
        'formal_training_admitted':False},indent=2))


if __name__=='__main__':main()

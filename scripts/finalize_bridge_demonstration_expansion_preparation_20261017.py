"""Reconcile the new CPU source audit; prepare metadata and schedule, no training."""
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import random

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'reports/bridge_additional_inventory_20261017'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path,value):
    assert not path.exists(), ('Preserve prepared artifacts',str(path))
    path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')


def main():
    candidate_path = ROOT/'configs/bridge_demonstration_expansion_candidates_20261017.json'
    p = read(candidate_path)
    audit_path = OUT/'source_audit.json'
    audit = read(audit_path)
    assert audit['passed'] and not audit['errors']
    assert audit['candidate_plan_sha256'] == sha(candidate_path)
    assert audit['future_test_targets_read'] is False and audit['selected_heldout_records_decoded'] == 0
    valid = {(e['shard'],e['record_index']) for e in audit['episodes']}
    new = [e for e in p['additional_train_candidates'] if (e['shard'],e['record_index']) in valid]
    old = read(ROOT/'configs/bridge_expansion_eligible_plan_v1.json')
    final = copy.deepcopy(p)
    final.update(status='new_training_source_verified_no_gpu_job_prepared',
                 candidate_plan_sha256=sha(candidate_path),new_training_source_audit_sha256=sha(audit_path),
                 additional_train_candidates=new,train=old['split_episodes']['train']+new,
                 source_ineligible=audit['source_ineligible'])
    final['proposed_training_episodes'] = len(final['train'])
    final['additional_training_episodes'] = len(new)
    final['proposed_training_counts_by_task'] = dict(Counter(e['instruction'] for e in final['train']))
    final['additional_training_counts_by_task'] = dict(Counter(e['instruction'] for e in new))
    old_origins = {e['origin_file_path'] for e in old['split_episodes']['train']}
    new_origins = {e['origin_file_path'] for e in new}
    final['source_file_grouping'].update(new_train_source_files=len(new_origins),
                                      additional_source_files_not_in_original_train=len(new_origins-old_origins))
    base = ROOT/'cluster_packages/bridge_coverage_replication_v1'
    old_rows = read(base/'rows.json')
    rows = copy.deepcopy(old_rows['train'])
    assert len(rows) == 1836
    audited = {(e['shard'],e['record_index']):e for e in audit['episodes']}
    for e in new:
        for start in audited[(e['shard'],e['record_index'])]['starts']:
            rows.append(dict(shard=e['shard'],record_index=e['record_index'],start_index=start,
                             task=e['instruction'],partition='train'))
    assert len({(r['shard'],r['record_index'],r['start_index']) for r in rows}) == len(rows)
    blocked = {(r['shard'],r['record_index']) for r in p['future_test_candidates_identity_and_metadata_only']+p['consumed_test_identity_only']+p['development']}
    assert not {(r['shard'],r['record_index']) for r in rows} & blocked
    with np.load(base/'schedule.npz',allow_pickle=False) as z:
        old_batches = z['batches'].copy()
    old_tasks = np.asarray([r['task'] for r in old_rows['train']])
    tasks = np.asarray([r['task'] for r in rows])
    requested = old_tasks[old_batches]
    streams = {}
    cursors = Counter()
    for ti,task in enumerate(sorted(set(tasks))):
        indices = np.flatnonzero(tasks==task).tolist()
        needed = int((requested==task).sum())+8
        sequence = []
        cycle = 0
        while len(sequence) < needed:
            order = indices.copy()
            random.Random(460001+ti*100000+cycle).shuffle(order)
            sequence.extend(order)
            cycle += 1
        streams[task] = sequence
    batches = np.empty((8192,8),dtype=np.int64)
    for update,labels in enumerate(requested):
        used = set()
        for slot,task in enumerate(labels):
            sequence = streams[task]
            cursor = cursors[task]
            if sequence[cursor] in used:
                at = next(i for i in range(cursor+1,len(sequence)) if sequence[i] not in used)
                sequence[cursor],sequence[at] = sequence[at],sequence[cursor]
            batches[update,slot] = sequence[cursor]
            used.add(sequence[cursor])
            cursors[task] += 1
    assert np.array_equal(tasks[batches],requested)
    assert all(len(set(row)) == 8 for row in batches)
    draws = np.bincount(batches.ravel(),minlength=len(rows))
    assert draws.min() > 0
    row_path = OUT/'eligible_window_rows.json'
    write(row_path,dict(train=rows,development=old_rows['development']))
    schedule_path = OUT/'eligible_schedule.npz'
    assert not schedule_path.exists()
    np.savez_compressed(schedule_path,batches=batches)
    source_info = read(Path('E:/dataset/bridge_v2_0.0.1/0.0.1/dataset_info.json'))
    lengths = next(x['shardLengths'] for x in source_info['splits'] if x['name']=='train')
    caches = sorted((OUT/'shards').glob('*.json'))
    assert len(caches) == 768
    for path in caches:
        i = int(path.name.split('-')[-3])
        assert read(path)['scanned_episodes'] == int(lengths[i])
    final.update(proposed_training_windows_from_metadata=len(rows),
                 additional_training_windows_from_metadata=len(rows)-1836,
                 source_eligibility_verified=True,new_window_rows_sha256=sha(row_path),
                 new_schedule_file_sha256=sha(schedule_path),
                 schedule_tensor_sha256=hashlib.sha256(batches.tobytes()).hexdigest(),
                 every_original_task_slot_preserved=True,schedule_shape=[8192,8],
                 draw_counts=dict(minimum=int(draws.min()),maximum=int(draws.max()),mean=float(draws.mean()),
                                  old_mean=65536/1836,unique_training_windows_seen=len(rows)),
                 all768_new_shard_record_counts_match_tfds=True,
                 source_audit_next_step_completed=True,
                 next_step='Prepare a dynamic-size input/cache wrapper and bounded resource/replay preflight for this verified pool, preserving the frozen207558 recipe. No source exports, GPU job, checkpoint or token cache created locally yet.')
    # Reconcile storage estimate to the actual eligible additions.
    sizes = read(OUT/'selected_training_payload_sizes.json')
    estimated_export = 0
    for shard,v in sizes.items():
        records = {int(i):n for i,n in v['selected_record_payload_bytes'].items() if (shard,int(i)) in valid}
        if records:
            estimated_export += sum(records.values())+16*(max(records)+1)
    final['storage_estimates']['new_training_export_uncompressed_bytes'] = estimated_export
    row_bytes = 1.8767607063055038*2**30/1876
    final['storage_estimates']['total_token_cache_estimated_bytes'] = int((len(rows)+40)*row_bytes)
    final['storage_estimates']['additional_token_cache_estimated_bytes'] = int((len(rows)-1836)*row_bytes)
    final_path = ROOT/'configs/bridge_demonstration_expansion_eligible_20261017.json'
    write(final_path,final)
    print(json.dumps(dict(eligible_plan=str(final_path),training_episodes=len(final['train']),
                         training_windows=len(rows),new_episodes=len(new),new_windows=len(rows)-1836,
                         source_ineligible=len(audit['source_ineligible']),draw_counts=final['draw_counts'],
                         future_test_candidates=25,source_audit_passed=True,
                         export_mib=estimated_export/2**20,
                         cache_gib=final['storage_estimates']['total_token_cache_estimated_bytes']/2**30),indent=2))


if __name__=='__main__':
    main()

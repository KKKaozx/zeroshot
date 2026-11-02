"""Prepare metadata-only810-episode candidate plan and storage estimates.

No training, target decoding, image decoding, exports or model imports.
The original25 new-test candidates stay identity-only. This plan is provisional
until a separate source-contract audit verifies the new training candidates.
"""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import struct

ROOT = Path(__file__).resolve().parents[1]
DATA = Path('E:/dataset/bridge_v2_0.0.1/0.0.1')
OUT = ROOT/'reports/bridge_additional_inventory_20261017'
PLAN = ROOT/'configs/bridge_demonstration_expansion_candidates_20261017.json'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ep(row):
    return row['shard'], row['record_index']


def origin(row):
    return row['origin_file_path'], row['episode_id']


def priority(row, role):
    return hashlib.sha256(json.dumps([role, row['instruction'], *origin(row)], ensure_ascii=False).encode()).hexdigest(), ep(row)


def payload_estimate(rows):
    """Read only TFRecord length headers; do not read any selected payload."""
    grouped = defaultdict(set)
    for row in rows:
        grouped[row['shard']].add(row['record_index'])
    result = {}
    for shard, indices in sorted(grouped.items()):
        entries = {}
        with (DATA/shard).open('rb') as stream:
            for i in range(max(indices)+1):
                header = stream.read(12)
                assert len(header) == 12
                length = struct.unpack('<Q', header[:8])[0]
                if i in indices:
                    entries[str(i)] = length
                stream.seek(length+4, 1)
        assert len(entries) == len(indices)
        result[shard] = dict(selected_record_payload_bytes=entries,
                            selected_payload_bytes=sum(entries.values()),
                            placeholder_export_bytes=sum(entries.values())+16*(max(indices)+1))
    return result


def main():
    assert not PLAN.exists(), 'Preserve prior candidate selection'
    inventory_path = OUT/'inventory.json'
    inv = read(inventory_path)
    old_path = ROOT/'configs/bridge_expansion_eligible_plan_v1.json'
    old = read(old_path)
    base_protocol_path = ROOT/'cluster_packages/bridge_coverage_replication_v1/protocol.json'
    base_protocol = read(base_protocol_path)
    base_report_path = ROOT/'reports/bridge-coverage-rep-fit-207558/report.json'
    base_report = read(base_report_path)
    assert base_report['execution_checks_passed'] and base_report['clip_unchanged']
    base_analysis_path = ROOT/'reports/bridge_coverage_replication_analysis_207558/analysis.json'
    base_analysis = read(base_analysis_path)
    assert base_analysis['verified']
    pool = inv['candidates']
    # Exclude full-parser validation records from a future blind test role.
    parser_records = {ep(x) for x in inv['metadata_reader_verified_against_official_example']}
    counts = Counter(e['instruction'] for e in old['split_episodes']['train'])
    additional, future_test = [], []
    for task in old['tasks']:
        candidates = [e for e in pool if e['instruction'] == task]
        eligible_test = sorted((e for e in candidates if ep(e) not in parser_records and int(e['shard'].split('-')[-3]) >= 256), key=lambda e: priority(e,'future-test-20261017-v1'))
        assert len(eligible_test) >= 5
        test = eligible_test[:5]
        future_test.extend(test)
        held = {origin(e) for e in test}
        train = sorted((e for e in candidates if origin(e) not in held),key=lambda e: priority(e,'additional-training-20261017-v1'))
        assert len(train) >= counts[task]
        additional.extend(train[:counts[task]])
    train = old['split_episodes']['train']+additional
    dev = old['split_episodes']['development']
    consumed = old['split_episodes']['reserved_test']
    assert len(train) == 810 and len(additional) == 405 and len(future_test) == 25
    assert len({origin(e) for e in train}) == 810
    for key in (ep, origin):
        groups = [set(map(key, v)) for v in (train, dev, consumed, future_test)]
        assert all(not groups[i] & groups[j] for i in range(4) for j in range(i+1,4))
    windows = lambda data: sum(len(range(0,e['steps']-16,4)) for e in data)
    new_windows = windows(additional)
    assert windows(old['split_episodes']['train']) == 1836
    # Estimate persistent cache from the measured1876-row baseline cache, rather
    # than copy any full source shard, checkpoint or token tensors locally.
    row_bytes = 1.8767607063055038*2**30/1876
    estimates = payload_estimate(additional)
    new_export_bytes = sum(x['placeholder_export_bytes'] for x in estimates.values())
    old_origin_files = {e['origin_file_path'] for e in old['split_episodes']['train']}
    new_origin_files = {e['origin_file_path'] for e in additional}
    plan = dict(date='2026-10-17',status='metadata_candidates_only_pending_source_contract_audit',
        objective='At the same8192-update compute budget, does adding405 independent training demonstrations improve existing development pose prediction?',
        inventory_sha256=sha(inventory_path),old_eligible_plan_sha256=sha(old_path),
        baseline_job='207558',baseline_report_sha256=sha(base_report_path),
        baseline_analysis_sha256=sha(base_analysis_path),
        baseline_protocol_sha256=sha(base_protocol_path),
        baseline_training_episodes=405,baseline_training_windows=1836,
        proposed_training_episodes=810,proposed_training_windows_from_metadata=windows(train),
        additional_training_episodes=405,additional_training_windows_from_metadata=new_windows,
        development_episodes=10,development_windows=40,consumed_208056_episodes_excluded=13,
        additional_training_counts_by_task=dict(Counter(e['instruction'] for e in additional)),
        proposed_training_counts_by_task=dict(Counter(e['instruction'] for e in train)),
        train=train,development=dev,additional_train_candidates=additional,
        consumed_test_identity_only=[dict(shard=e['shard'],record_index=e['record_index'],origin_file_path=e['origin_file_path'],episode_id=e['episode_id']) for e in consumed],
        future_test_candidates_identity_and_metadata_only=future_test,
        future_test_count=25,future_test_selection='Five per task by deterministic SHA256 of role/task/source identity, before target inspection; source-schema eligibility still unverified. Never replace based on model error.',
        selection_policy='Keep all405 original training episodes; reserve25 metadata-only test candidates first, then add exactly the old per-task episode count using a separate deterministic source-identity hash ordering. Exclude all former435 selected records and their source identities including7 known rejects.',
        baseline_recipe=base_protocol['recipe'],baseline_model_config=base_protocol['config'],
        development_reference=dict(window_weighted=base_analysis['metrics']['development']['coverage'],
            episode_equal=base_analysis['metrics']['development']['episode_equal']['coverage']),
        baseline_choice='207558 is the already-verified seed42 standard-loss405-episode coverage control from the earlier development experiments. Do not choose the rotation-weight model, task weights or data composition based on208056 test scores.',
        controls=['Same five normalized instruction strings; no BC-Z or other dataset.',
                  'Reuse unchanged seed42 initialization and all original trainable/frozen modules; standard x0 pose loss, no rotation reweighting.',
                  'Batch8, total8192 updates, Adam, learning rate1e-4 for4096 then1e-5; no extra training or seed/weight sweep.',
                  'Keep every8192x8 task-label slot from207558 unchanged; generate the new window schedule only after source eligibility is final.',
                  'Keep original128-observation state statistics, image normalization,16-step targets, stride4, and DDPM100/seeds1101-1103 evaluation.',
                  'Reuse405-episode baseline evidence and cache; no baseline retraining or repeated old geometry audit. New candidate source audit only.',
                  'Retain original evaluation context execution bundle that reproduced0 difference; no inference-mode refactor.'],
        exposure_interpretation='A larger pool under the same65536 sample draws reduces average draws per window. This tests practical fixed-compute coverage, not the isolated effect of dataset size at equal per-window exposure.',
        development_decision='One final candidate on the original40 development windows. Require both episode-equal and window-weighted mean path position/rotation no higher than frozen207558, with at least one strictly lower; report all10 episodes and5 tasks. If failed, stop and report; do not extend updates or sweep parameters automatically.',
        final_test_policy='Only after the development rule passes and candidate weights/recipe are frozen, perform one new same-task independent-demonstration assessment of the baseline and candidate on the eligible future-test subset. No inference or target decoding of those25 records now. Do not use208056 for selection/tuning.',
        source_audit_policy='Before export or training, audit every selected new training record using unchanged loader/geometry, finite state/action, all valid windows, images and existing reverse_scan_valid_steps_v2 gripper-tail eligibility. Any incompatible record is reported; no replacement based on target magnitude or model score. Reconcile actual counts and freeze the final protocol before GPU work.',
        source_file_grouping=dict(original_train_source_files=len(old_origin_files),new_train_source_files=len(new_origin_files),
                                  additional_source_files_not_in_original_train=len(new_origin_files-old_origin_files),
                                  limitation='Source .npy file grouping is not certification of distinct physical scenes.'),
        storage_estimates=dict(new_training_export_uncompressed_bytes=new_export_bytes,
            total_token_cache_estimated_bytes=int((windows(train)+40)*row_bytes),
            additional_token_cache_estimated_bytes=int(new_windows*row_bytes),
            local_d_free_bytes=shutil.disk_usage(ROOT).free,source_e_free_bytes=shutil.disk_usage(DATA).free,
            current_preparation_local_bytes='Only metadata, script, and plan; no TFRecord export, model download or persistent token cache.',
            export_note='Estimate from source TFRecord length headers plus empty-record placeholders; later protobuf reserialization/compression may change size. Do not copy full shards.',
            gpu_note='Additional token rows can increase GPU memory; verify with one bounded preflight before formal work. Current7.5GiB gate must not be assumed to pass.'),
        training_started=False,new_gpu_job_prepared=False,new_test_predictions=False,
        next_step='CPU source-contract audit only the405 additional training candidates; preserve25 future-test candidates as metadata-only. Then reconcile eligibility and prepare one resource preflight and one formal run; no repeated fitting ladder.')
    # Metadata only: no target-derived selection fields are present.
    assert not any('position_error' in str(x) for x in additional)
    PLAN.write_text(json.dumps(plan,indent=2)+'\n',encoding='utf-8')
    (OUT/'selected_training_payload_sizes.json').write_text(json.dumps(estimates,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(plan=str(PLAN),additional_candidates_available=inv['additional_unique_candidates'],
        training_episodes=810,training_windows=windows(train),future_test_candidates=25,
        additional_source_files_not_in_original_train=len(new_origin_files-old_origin_files),
        export_mib=new_export_bytes/2**20,total_token_cache_gib=(windows(train)+40)*row_bytes/2**30),indent=2))


if __name__=='__main__':
    main()

"""Freeze one paired continuation design using EXISTING metadata/schedules only.

No raw data, target inspection, model loading, submission or execution of fitting.
"""
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import random
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"experiments/bridge_prefix_coverage_pair_v1"


def read(p):return json.loads(p.read_text(encoding="utf-8"))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def key(e):return e["shard"],e["record_index"]
def origin(e):return e["origin_file_path"],e["episode_id"]


def main():
    assert not OUT.exists(),"Keep frozen protocol; do not tune selection after results"
    active=read(ROOT/"configs/active_experiment_reference.json")
    for p,h in active["local_pinned_files"].items():assert sha(ROOT/p)==h
    package=ROOT/"cluster_packages/bridge_coverage_replication_v1"
    base=read(package/"protocol.json")
    rows=read(ROOT/"reports/bridge_partial_future_contract_20261017/candidate_rows.json")["train"]
    assert len(rows)==3394
    old=read(package/"rows.json")["train"]
    assert [(r['shard'],r['record_index'],r['start_index']) for r in rows[:1836]]==[
        (r['shard'],r['record_index'],r['start_index']) for r in old]
    with np.load(package/"schedule.npz",allow_pickle=False) as z:
        control=z["batches"][:4096].copy()
    assert sha(package/"schedule.npz")==base["schedule_file_sha256"]
    candidate=control.copy()
    horizons=np.where(np.arange(4096)%2==0,16,1).astype(np.int64)
    labels=np.array([r["task"] for r in rows])
    task_slots=labels[control]
    streams={};cursors=Counter()
    for index,task in enumerate(sorted(set(labels))):
        indices=np.flatnonzero(labels==task).tolist()
        wanted=int((task_slots[horizons==1]==task).sum())+8
        stream=[];cycle=0
        while len(stream)<wanted:
            order=indices.copy()
            random.Random(17102026+index*100000+cycle).shuffle(order)
            stream+=order;cycle+=1
        streams[task]=stream
    for u in np.flatnonzero(horizons==1):
        used=set()
        for j,task in enumerate(task_slots[u]):
            stream=streams[task];cursor=cursors[task]
            if stream[cursor] in used:
                other=next(k for k in range(cursor+1,len(stream)) if stream[k] not in used)
                stream[cursor],stream[other]=stream[other],stream[cursor]
            candidate[u,j]=stream[cursor];used.add(stream[cursor]);cursors[task]+=1
    assert np.array_equal(control[horizons==16],candidate[horizons==16])
    assert np.array_equal(labels[control],labels[candidate])
    assert all(len(set(v))==8 for v in candidate)
    counts=np.bincount(candidate[horizons==1].ravel(),minlength=3394)
    assert counts.min()>0,"One-step candidate must actually cover every candidate input"
    # New development selection is metadata-only, before any model/target view.
    # Use >=256 to exclude earlier diagnostic records, the full810 proposal
    # (including four subsequently rejected additions), and the already-reserved25.
    inventory=read(ROOT/"reports/bridge_additional_inventory_20261017/inventory.json")
    expansion=read(ROOT/"configs/bridge_demonstration_expansion_candidates_20261017.json")
    final=expansion["future_test_candidates_identity_and_metadata_only"]
    assert len(final)==25
    historical=expansion["train"]+expansion["development"]+expansion["consumed_test_identity_only"]+final
    excluded_keys=set(map(key,historical))
    excluded_origins=set(map(origin,historical))
    parser_records=set(map(key,inventory["metadata_reader_verified_against_official_example"]))
    validation=[]
    for task in sorted(set(labels)):
        available=[e for e in inventory["candidates"] if e["instruction"]==task
            and int(e["shard"].split('-')[-3])>=256 and key(e) not in excluded_keys|parser_records
            and origin(e) not in excluded_origins]
        def priority(e):
            return hashlib.sha256(json.dumps(["prefix-pair-development-20261017-v1",task,*origin(e)],
                ensure_ascii=False).encode()).hexdigest(),key(e)
        chosen=sorted(available,key=priority)[:2]
        assert len(chosen)==2
        validation+=chosen
    assert len(validation)==10
    assert not set(map(key,validation))&excluded_keys
    assert not set(map(origin,validation))&excluded_origins
    available_windows=sum(len(range(0,e["steps"]-16,4)) for e in validation)
    old_pool=read(ROOT/"training_cache/exports/bridge_expansion_training_v1/manifest.json")
    training_keys=set(map(key,old_pool["partitions"]["train"]))
    assert set(map(key,rows))==training_keys and len(training_keys)==405
    assert not training_keys&set(map(key,final+validation))
    sources=[Path(__file__),ROOT/"configs/active_experiment_reference.json",package/"protocol.json",
        package/"rows.json",package/"schedule.npz",
        ROOT/"reports/bridge_partial_future_contract_20261017/candidate_rows.json",
        ROOT/"reports/bridge_prefix_full_interface_20261017/report.json",
        ROOT/"scripts/bridge_prefix_model_interface_v1.py",
        ROOT/"scripts/bridge_requested_prefix_diffusion_v1.py",
        ROOT/"reports/bridge_additional_inventory_20261017/inventory.json",
        ROOT/"configs/bridge_demonstration_expansion_candidates_20261017.json"]
    assert read(sources[6])["full_size_CPU_interface_verified"]
    protocol=dict(date="2026-10-17",status="frozen_design_pending_new_input_and_cluster_checkpoint_checks",
        objective="Within the SAME requested-horizon model and compute, does adding late CURRENT observations improve NEW-demonstration full16 pose errors?",
        active_baseline=active,training_episodes=405,
        arms=dict(prefix_control=dict(K16_rows="old1836",K1_rows="old1836"),
            prefix_added=dict(K16_rows="same old1836",K1_rows="all3394 including1558 late observations")),
        comparison_meaning="Added-current-input coverage at fixed compute inside the new two-request training scheme. Compare to untouched207558 separately; not a pure architecture comparison to207558.",
        initialization=dict(source_job=207558,source_checkpoint=active["remote_checkpoint"],
            sha256=active["checkpoint_sha256"],base_parameters="Same trained207558 in BOTH arms, no random restart",
            request_projection="Zero-init,16384 new parameters, RNG protected, same in BOTH arms",
            optimizer="FreshAdam for BOTH arms; old final_trainable.pt contains no optimizer state, not a resume of the oldAdam trajectory"),
        recipe=dict(new_updates_per_arm=4096,arms=2,total_new_optimizer_updates=8192,
            batch_size=8,K16_updates_per_arm=2048,K1_updates_per_arm=2048,
            request_order="Odd1-indexed updatesK16, even updatesK1",
            learning_rate=1e-5,learning_rate_schedule="constant",weight_decay=0,gradient_clip=1.,
            context_dropout_seed_rule="620000+update",diffusion_noise_seed_rule="610000+update",
            draw_tensor_shape=[4096,8],task_slots="Original207558 first4096 batches; all slots matched exactly across arms",
            K16_schedule="Identical original indices in both arms",
            K1_control_schedule="Original indices",K1_added_schedule="Fixed task-matched per-task shuffled streams across3394 rows; no duplicate row in batch",
            early_stopping=False,checkpoint_selection="Final update4096 only",sampling_steps=100,
            diffusion_sampling_seeds=[1101,1102,1103]),
        frozen=dict(CLIP=True,gripper=True,state_statistics="Original128 training observations only",
            current_observation_history="Same-current double Adapter calls, unchanged207558",
            image_preprocessing="Pinned frozen dataset.py; no augmentation",
            action="Fixed-input tool frame XYZ/.1m +xyzw reached pose",target_prediction="x0/sample",
            pose_loss="K16 original flattenedMSE;K1 only first real7D pose"),
        sampler_checks=dict(candidate_K1_all3394_seen=True,K1_min_draws=int(counts.min()),
            K1_max_draws=int(counts.max()),K1_added_late_draws=int((candidate[horizons==1]>=1836).sum()),
            K16_across_arm_exact=True,all_task_slots_across_arm_exact=True,
            task_draw_counts=dict(Counter(labels[control].ravel()))),
        validation=dict(kind="New development; metadata-only qualifications pending",episodes=validation,
            episodes_per_task=2,full16_windows_from_metadata=available_windows,
            window_rule="All stride4 starts with16 real future observations",selection_uses_model_errors=False,
            excludes_known810_training_proposal_and_prior_dev_and_consumed13_and_reserved25=True,
            numeric_or_images_read=False,checkpoint_or_hyperparameter_selection=False),
        final_assessment=dict(kind="Previously reserved25 same-task episode candidates, NOT LIBERO/CLIPort benchmark",
            metadata_only_episodes=final,episode_count=25,targets_or_predictions_read=False,
            policy="Only after new-development pass and final weights/config frozen; evaluate BOTH paired arms and207558 once, same full16 protocol; no tuning/retest/drop-on-error"),
        evaluation=dict(primary="Full16 path pose errors, episode-equal; do not substituteK1 error",
            report_all="Window-weighted, all episodes/tasks, endpoints and first-target diagnostics; no cherry-picking",
            methods=["unchanged207558","prefix_control","prefix_added","static"],
            admission_rule=["Candidate episode-equal full16 path position strictly lower than control,207558,and static",
                "Candidate episode-equal full16 path rotation no higher than control,207558,and static",
                "At least6of10 development episodes improve full16 position versus control AND versus207558",
                "Window-weighted full16 path position/rotation no higher than control and207558"],
            K1="Separate auxiliary result; cannot satisfy full16 admission",old40_and_consumed13="Do not rescore for selection"),
        preflight_before_any_formal=dict(actual_source207558_hash_and_key_check=True,
            original_old_window_mode_and_prediction_replay="Pinned no_grad/eval semantics and saved original4-window references; no tolerance changes",
            added1558_inputs="Numeric, image/language, real-next-action and current-state features audited on training records; no padded supervision",
            development10="Qualify schema/real-label/image eligibility without policy scores before training; any incompatibility stops, no error-based replacement",
            one_resource_check="One fixed8-update-per-arm operational run, no development scoring or fitted-performance gate; verify matched initial base/new port, Adam/RNG restore, gradients/freeze, save/load, memory/time",
            resource_acceptance="<=7.5GiB peakreserved and <=2h estimated paired formal run; otherwise stop before formal; no automatic architecture/batch/seed changes"),
        stopping="One paired4096-new-update continuation per arm. If new-development admission fails, retain207558, close branch; no extra rounds/seeds/K/crop/freeze/budget sweep. Final set remains closed on failure.",
        not_done=["All1558 added real image/token inputs", "New10 numeric eligibility", "Actual207558 trainedcheckpoint replay and migration", "Clusterresource check", "Runnable training/evaluation launchers", "Training/generalization/grasp results"],
        training_authorized_or_submitted=False,new_GPU_job_prepared=False,
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources})
    OUT.mkdir(parents=True)
    np.savez_compressed(OUT/"matched_schedule.npz",control=control,candidate=candidate,requested_horizons=horizons)
    protocol["matched_schedule_sha256"]=sha(OUT/"matched_schedule.npz")
    (OUT/"protocol.json").write_text(json.dumps(protocol,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps(dict(protocol_path=str(OUT/"protocol.json"),new_updates_per_arm=4096,
        new_validation_episodes=len(validation),new_validation_full16_windows=available_windows,
        reserved_final_episodes=len(final),sampler_checks=protocol["sampler_checks"],GPU_submission=False),indent=2))


if __name__=="__main__":main()

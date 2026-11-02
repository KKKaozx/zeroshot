"""Read saved metadata/code only; no raw datasets, targets, model or GPU access."""
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/active_training_scheme_review_20261017"


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def episode_key(row):
    return row["shard"], row["record_index"]


def source_group(path):
    parts = path.replace("\\", "/").split("/")
    for i, part in enumerate(parts):
        if part in {"bridge_data_v1", "bridge_data_v2"}:
            return "/".join(parts[i:i + 3])
    return "unclassified"


def main():
    assert not OUT.exists(), "Preserve the completed review; do not overwrite"
    active = read("configs/active_experiment_reference.json")
    for path, expected in active["local_pinned_files"].items():
        assert sha(path) == expected, path
    package = "cluster_packages/bridge_coverage_replication_v1/"
    protocol = read(package + "protocol.json")
    rows = read(package + "rows.json")
    pool_path = "training_cache/exports/bridge_expansion_training_v1/manifest.json"
    pool = read(pool_path)
    references = read(package + "episode_reference.json")
    selection_path = "configs/bridge_expansion_eligible_plan_v1.json"
    selection = read(selection_path)
    assert sha(pool_path) == protocol["pool_manifest_sha256"]
    assert sha(package + "rows.json") == protocol["rows_sha256"]
    assert sha(package + "episode_reference.json") == protocol["episode_reference_sha256"]
    assert sha(package + "schedule.npz") == protocol["schedule_file_sha256"]
    audit = {episode_key(e): e for e in references["episodes"]}
    train_keys = set(map(episode_key, pool["partitions"]["train"]))
    dev_keys = set(map(episode_key, pool["partitions"]["validation"]))
    assert len(train_keys) == 405 and len(dev_keys) == 10 and not train_keys & dev_keys
    assert {episode_key(r) for r in rows["train"]} == train_keys
    assert {episode_key(r) for r in rows["development"]} == dev_keys
    reserved = set(map(episode_key, pool["reserved_test_identity_only"]))
    assert not (train_keys | dev_keys) & reserved
    origins = lambda es: {(e["origin_file_path"], e["episode_id"]) for e in es}
    assert not origins(pool["partitions"]["train"]) & origins(pool["partitions"]["validation"])
    with np.load(ROOT / (package + "schedule.npz"), allow_pickle=False) as z:
        batches = z["batches"].copy()
    assert batches.shape == (8192, 8)
    assert hashlib.sha256(batches.tobytes()).hexdigest() == protocol["schedule_tensor_sha256"]
    assert min(batches.ravel()) >= 0 and max(batches.ravel()) < 1836
    draws = np.bincount(batches.ravel(), minlength=1836)
    train_by_ep = defaultdict(list)
    for i, row in enumerate(rows["train"]):
        train_by_ep[episode_key(row)].append((i, row))
    episode_rows = []
    for e in pool["partitions"]["train"]:
        n = e["steps"]
        entries = train_by_ep[episode_key(e)]
        starts = sorted(row["start_index"] for _, row in entries)
        assert starts == list(range(0, n - 16, 4)) == audit[episode_key(e)]["starts"]
        # T observations give T-1 valid current -> next-observation transitions.
        possible_inputs = set(range(n - 1))
        complete_future_inputs = set(range(n - 16))
        late_inputs = possible_inputs - complete_future_inputs
        targets_seen = {t + k for t in starts for k in range(1, 17)}
        assert not set(starts) & late_inputs and len(late_inputs) == 15
        episode_rows.append(dict(
            shard=e["shard"], record_index=e["record_index"], task=e["instruction"],
            source_group=source_group(e["origin_file_path"]), observations=n,
            valid_current_inputs=n - 1, complete_future_inputs=n - 16,
            selected_inputs=len(starts), complete_future_inputs_skipped_by_stride=len(complete_future_inputs-set(starts)),
            late_inputs_excluded_by_horizon=len(late_inputs),
            future_observations_seen_as_targets=len(targets_seen),
            final_observation_seen_as_target=(n - 1 in targets_seen),
            last_selected_start=max(starts), last_selected_start_fraction=max(starts)/(n - 1),
            hypothetical_stride4_inputs_with_partial_future=len(range(0, n - 1, 4)),
            optimizer_draws=int(sum(draws[i] for i, _ in entries)),
        ))
    task_rows = []
    task_labels = [r["task"] for r in rows["train"]]
    for task in sorted(set(task_labels)):
        eps = [e for e in episode_rows if e["task"] == task]
        indices = [i for i, label in enumerate(task_labels) if label == task]
        d = draws[indices]
        task_rows.append(dict(task=task, episodes=len(eps), windows=len(indices),
            window_share=len(indices)/1836, optimizer_draws=int(d.sum()),
            optimizer_draw_share=float(d.sum()/batches.size),
            draws_per_window_min=int(d.min()), draws_per_window_mean=float(d.mean()),
            draws_per_window_max=int(d.max()),
            valid_current_inputs=sum(e["valid_current_inputs"] for e in eps),
            late_inputs_excluded_by_horizon=sum(e["late_inputs_excluded_by_horizon"] for e in eps)))
    assert {t["task"]: t["optimizer_draws"] for t in task_rows} == protocol["schedule"]["task_draw_counts"]
    # Independently count exposures without bincount and sum lengths from pool metadata.
    direct_draws = Counter(map(int, batches.ravel()))
    assert all(direct_draws[i] == int(draws[i]) for i in range(1836))
    n_valid = sum(e["steps"] - 1 for e in pool["partitions"]["train"])
    n_full = sum(e["steps"] - 16 for e in pool["partitions"]["train"])
    assert n_valid == sum(e["valid_current_inputs"] for e in episode_rows)
    assert n_full == sum(e["complete_future_inputs"] for e in episode_rows)
    assert n_valid - n_full == 405 * 15
    runner = (ROOT / "reports/bridge-coverage-rep-fit-207558/runner_snapshot.py").read_text()
    assert runner == (ROOT / (package + "run_training_coverage.py")).read_text()
    assert "loss=torch.nn.functional.mse_loss(guess,truth)" in runner
    assert "repeated=model.adapter(visual[indices],language[indices],mask[indices])" in runner
    runtime = read("reports/bridge-coverage-rep-fit-207558/report.json")
    assert runtime["execution_checks_passed"] and not runtime["gripper_trained"]
    assert not runtime["reserved_test_targets_read"]
    contract = (ROOT / "reports/official_pusht_195527/checkpoint_config.yaml").read_text()
    assert "pad_before: 1" in contract and "pad_after: 7" in contract
    sources = ["configs/active_experiment_reference.json", pool_path, selection_path,
        package + "protocol.json", package + "rows.json", package + "episode_reference.json",
        package + "schedule.npz", package + "run_training_coverage.py",
        "reports/bridge-coverage-rep-fit-207558/report.json",
        "reports/official_pusht_195527/checkpoint_config.yaml",
        "baselines/diffusion_policy_data_reference/diffusion_policy/common/sampler.py",
        "baselines/diffusion_policy_data_reference/diffusion_policy/dataset/pusht_image_dataset.py",
        "scripts/prepare_bridge_training_coverage_20261016.py",
        "cluster_packages/bridge_growth128_v1/models.py",
        "cluster_packages/bridge_growth128_v1/trace_bridge_batch_training.py",
        "cluster_packages/bridge_state_conditioning_v1/state_input_contract.py"]
    report = dict(
        review_type="saved_metadata_and_code_review_not_a_new_performance_experiment",
        verified=True, active_job=207558, model_loaded=False, raw_dataset_read=False,
        images_decoded=False, targets_loaded=False, new_predictions=False,
        gpu_used=False, jobs_submitted=False, production_code_changed=False,
        source_hashes={p: sha(p) for p in sources},
        selection_policy=selection["expansion_selection"],
        exclusions_for_unresolved_gripper_commands=len(selection["eligibility_revision"]["excluded_training_episodes"]),
        counts=dict(train_episodes=405, train_windows=1836, development_episodes=10,
            development_windows=40, consumed_independent_assessment_episodes=13,
            training_observations=sum(e["observations"] for e in episode_rows),
            valid_current_inputs=n_valid, complete_16_future_inputs=n_full,
            selected_current_inputs=1836, selected_fraction_of_valid_current_inputs=1836/n_valid,
            skipped_complete_future_inputs_due_to_stride=n_full-1836,
            excluded_late_current_inputs_due_to_full_horizon=n_valid-n_full,
            hypothetical_stride4_inputs_including_partial_future=sum(e["hypothetical_stride4_inputs_with_partial_future"] for e in episode_rows),
            future_observations_seen_as_targets=sum(e["future_observations_seen_as_targets"] for e in episode_rows)),
        tasks=task_rows,
        source_groups=dict(Counter(e["source_group"] for e in episode_rows)),
        split_record_identity_disjoint=True, split_source_episode_identity_disjoint=True,
        camera_or_layout_split_proven=False, task_semantic_phase_labels_available=False,
        temporal_coverage_meaning="Only current-input indices counted; targets cover many excluded input frames. Last15 excludes final observation itself, which has no next-state label.",
        actual_objective="MSE of 7D pose denoising target; gripper frozen; no gripper objective",
        actual_history="Two Adapter calls with the SAME current visual/language tokens; not two recorded observations",
        state_statistics="128 original training observations; subset of training only, not evaluation leakage",
        author_local_code_comparison=dict(horizon=16, n_obs_steps=2, pad_before=1, pad_after=7,
            sampling="Dense sequence starts including boundary padding; last recorded values repeated",
            warning="Different task/action/observation alignment. Do not copy padding as verified Bridge future measurements or assert all episode-end inputs are covered."),
        root_cause_identified=False, improvement_demonstrated=False,
        recommendation="Preserve207558. Next prepare a CPU-only temporal-coverage/valid-future-mask contract on the same405 train episodes before choosing a new GPU run. No new source/task, architecture, or hyperparameter sweep.",
        candidate_scope="Keep stride4 and all old1836 full windows; consider additional late starts with at least one real future target, explicitly mask absent future supervision. This changes data/objective support, not a proven single-cause fix; incomplete masked denoising design must be resolved before training.",
        nonclaims=["Missing late inputs do not prove why errors on the already-scored full windows occur.",
            "Index positions are not automatically grasp, release or hold semantic phases.",
            "Enabling joint gripper training cannot by itself guarantee better pose generalization.",
            "No claim of closed-loop transfer from Bridge to Panda/LIBERO/CLIPort.",
            "Consumed13-episode assessment must not be reused for model selection."],
    )
    OUT.mkdir(parents=True)
    for filename, values in [("episodes.csv", episode_rows), ("tasks.csv", task_rows)]:
        with (OUT / filename).open("w", encoding="utf-8", newline="") as stream:
            writer=csv.DictWriter(stream, fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(json.dumps(dict(verified=True, counts=report["counts"], tasks=task_rows,
        source_groups=report["source_groups"], output=str(OUT)), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""Sweep DDIM inference steps for the frozen 193195 epsilon checkpoint."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

os.environ.update(USE_TF="0", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--epsilon-run", type=Path, required=True)
    parser.add_argument("--epsilon-report", type=Path, required=True)
    parser.add_argument("--head-report", type=Path, required=True)
    parser.add_argument("--window-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", default="16,32,100")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    step_counts = [int(value) for value in args.steps.split(",")]
    if step_counts != sorted(set(step_counts)) or step_counts[0] < 2:
        raise ValueError("Step counts must be sorted, unique, and at least 2")

    sys.path.insert(0, str(args.pack.resolve()))
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    import torch
    from transformers import CLIPTokenizer
    from dataset import UnifiedRobotDataset
    from models import RobotAdapterModel
    from train import (bridge_plan_selection, bridge_plan_splits, collate_batch,
                       set_seed, trainable_state_dict)
    from evaluate_bridge_one_step_sampler import (
        decode_state_gripper, finish_pose, select_windows, sha256,
    )
    from evaluate_bridge_ddim_steps import ddim_sample, timestep_schedule
    from evaluate_bridge_endpoint_metrics import scalar_metrics, oracle_minimum

    for count in step_counts:
        timestep_schedule(count)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    training_report = json.loads(args.epsilon_report.read_text())
    head_report = json.loads(args.head_report.read_text())
    window_reference = json.loads(args.window_reference.read_text())
    if not (training_report["passed"] and training_report["phase"] == "train"
            and training_report["updates"] == 18360):
        raise ValueError("The epsilon source training did not pass")
    if training_report["protocol"]["source_head_report_sha256"] != sha256(args.head_report):
        raise ValueError("Epsilon training does not bind to the head-control report")

    manifest = args.pack / "manifest.json"
    module_hashes = {
        name: sha256(args.pack / name)
        for name in head_report["protocol"]["module_sha256"]
    }
    if module_hashes != head_report["protocol"]["module_sha256"]:
        raise ValueError("Archived model modules differ")
    selection = bridge_plan_selection(manifest)
    dataset = UnifiedRobotDataset(
        data_dir=str(args.pack / "data"), chunk_size=16, stride=4,
        sources=["tfrecord"], min_trajectory_steps=17,
        exclude_path_parts=[], exclude_schemas=[], tfrecord_splits=["train"],
        bridge_gripper_policy="reverse_scan_valid_steps_v2",
        bridge_current_gripper="continuous", bridge_episode_selection=selection,
    )
    splits = bridge_plan_splits(dataset)
    rows = select_windows(dataset, splits, selection)
    if rows != window_reference["window_selection"]:
        raise ValueError("Window population differs from reference")
    window_sha = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    if window_sha != head_report["protocol"]["window_sha256"]:
        raise ValueError("Window hash differs from source training")
    items = [dataset[row["dataset_index"]] for row in rows]
    targets = np.stack([item[3].numpy() for item in items]).astype(np.float32)

    checkpoint_path = args.epsilon_run / "epsilon-final.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint["split_indices"] != splits:
        raise ValueError("Checkpoint split mismatch")
    model = RobotAdapterModel(checkpoint["config"])
    if set(checkpoint["trainable_state_dict"]) != set(trainable_state_dict(model)):
        raise ValueError("Checkpoint parameter mismatch")
    model.load_state_dict(checkpoint["trainable_state_dict"], strict=False)
    model = model.cuda().eval()
    if not (model.decoder_type == "diffusion" and
            model.diffusion_prediction_type == "epsilon" and
            model.num_diffusion_steps == 100):
        raise ValueError("Unexpected epsilon checkpoint configuration")
    tokenizer = CLIPTokenizer.from_pretrained(
        checkpoint["config"]["model"]["name"], local_files_only=True
    )

    groups = {}
    for partition in ("train", "validation"):
        groups[partition + "/overall"] = [
            i for i, row in enumerate(rows) if row["partition"] == partition
        ]
        for task in sorted({row["task"] for row in rows if row["partition"] == partition}):
            groups[partition + "/task/" + task] = [
                i for i, row in enumerate(rows)
                if row["partition"] == partition and row["task"] == task
            ]
        for shard, record in sorted({
                (row["shard"], row["record_index"])
                for row in rows if row["partition"] == partition}):
            groups[f"{partition}/episode/{shard}::{record}"] = [
                i for i, row in enumerate(rows)
                if row["partition"] == partition
                and (row["shard"], row["record_index"]) == (shard, record)
            ]

    predictions = {
        count: np.empty((3, len(rows), 16, 8), np.float32)
        for count in step_counts
    }
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for start in range(0, len(rows), 2):
            batch_items = items[start:start + 2]
            language, images, current, _, _ = collate_batch(batch_items)
            tokens = tokenizer(language, padding=True, truncation=True, return_tensors="pt")
            context = model.get_context_vector(
                images.cuda(), tokens["input_ids"].cuda(),
                tokens["attention_mask"].cuda(),
            )
            for seed in range(3):
                set_seed(seed * 10000 + start)
                initial = torch.randn(len(batch_items), 16, 7, device="cuda")
                for count in step_counts:
                    pose = ddim_sample(torch, model, context, initial, count)
                    pose = finish_pose(
                        torch, pose, float(model.max_normalized_position or 3.0)
                    )
                    gripper = decode_state_gripper(
                        torch, model, context, pose, current.cuda()
                    )
                    value = torch.cat([pose, gripper.unsqueeze(-1)], dim=-1)
                    if not torch.isfinite(value).all():
                        raise ValueError(f"Non-finite epsilon DDIM result at {count} steps")
                    predictions[count][seed, start:start + len(batch_items)] = value.cpu().numpy()
            if (start + len(batch_items)) % 200 == 0:
                print("EPSILON_SWEEP_PROGRESS", start + len(batch_items), "/", len(rows), flush=True)

    # The 8-step array is the authoritative output produced by the training
    # job itself. Cross-node CUDA kernels are not guaranteed bitwise replay.
    with np.load(args.epsilon_run / "epsilon-ddim8-predictions.npz") as archive:
        saved_eight = np.asarray(archive["predictions"], dtype=np.float32)
        saved_targets = np.asarray(archive["targets"], dtype=np.float32)
    if not np.array_equal(saved_targets, targets):
        raise ValueError("Saved epsilon targets differ")
    predictions[8] = saved_eight
    all_step_counts = [8] + step_counts
    np.savez_compressed(
        args.output / "epsilon-step-predictions.npz",
        **{f"steps_{count}": predictions[count] for count in all_step_counts},
        targets=targets,
    )

    result_groups = {}
    for group, indices in groups.items():
        target = targets[indices]
        static = np.zeros_like(target)
        static[..., 6] = 1
        result_groups[group] = {
            "windows": len(indices),
            "static": scalar_metrics(static, target),
            "epsilon_ddim": {
                str(count): [scalar_metrics(draw[indices], target)
                             for draw in predictions[count]]
                for count in all_step_counts
            },
            "epsilon_oracle": {
                str(count): oracle_minimum(predictions[count][:, indices], target)
                for count in all_step_counts
            },
        }

    report = {
        "stage": "frozen_epsilon_checkpoint_ddim_step_sweep",
        "trained": False,
        "weights_updated": False,
        "reserved_test_targets_read": False,
        "step_counts_are_unet_evaluations": all_step_counts,
        "computed_in_this_job": step_counts,
        "eta": 0.0,
        "source": {
            "epsilon_report_sha256": sha256(args.epsilon_report),
            "head_report_sha256": sha256(args.head_report),
            "window_reference_sha256": sha256(args.window_reference),
            "checkpoint_sha256": sha256(checkpoint_path),
            "saved_eight_predictions_sha256": sha256(
                args.epsilon_run / "epsilon-ddim8-predictions.npz"
            ),
            "window_sha256": window_sha,
        },
        "counts": {partition: sum(row["partition"] == partition for row in rows)
                   for partition in ("train", "validation")},
        "sampling_seeds": [0, 1, 2],
        "eight_step_source": "Saved predictions from the completed 193195 training job; not recomputed across a different GPU node.",
        "timestep_schedules": {str(count): timestep_schedule(count) for count in all_step_counts},
        "groups": result_groups,
        "elapsed_seconds": time.monotonic() - started,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024 ** 3,
        "limits": [
            "Inference-only sweep of the fixed 193195 epsilon checkpoint.",
            "Development data may choose a future sampler but are not held-out test results.",
            "Offline trajectory and window-endpoint errors are not robotic task success.",
            "Oracle metrics inspect ground truth and are not deployable.",
        ],
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("EPSILON STEP SWEEP: PASSED", args.output, flush=True)


if __name__ == "__main__":
    main()

"""Evaluate deterministic DDIM schedules using the frozen 192651 diffusion model."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

os.environ.update(USE_TF="0", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")


def timestep_schedule(evaluations):
    if not 1 <= evaluations <= 100:
        raise ValueError("evaluations must be in [1, 100]")
    if evaluations == 1:
        return [99]
    values = np.rint(np.linspace(99, 0, evaluations)).astype(int).tolist()
    if values[0] != 99 or values[-1] != 0 or len(set(values)) != evaluations:
        raise ValueError("Invalid DDIM timestep schedule")
    return values


def clip_clean(torch, clean, position_limit):
    return torch.cat([
        clean[..., :3].clamp(-position_limit, position_limit),
        clean[..., 3:7].clamp(-1, 1),
    ], dim=-1)


def ddim_sample(torch, model, context, initial_noise, evaluations):
    """Eta=0 generalized reverse process for an x0-prediction network."""
    actions = initial_noise.clone()
    schedule = timestep_schedule(evaluations)
    position_limit = float(model.max_normalized_position or 3.0)
    for index, step in enumerate(schedule):
        timesteps = torch.full(
            (len(actions),), step, device=actions.device, dtype=torch.long
        )
        raw_clean = model.diffusion_decoder(actions, timesteps, context)
        if len(schedule) == 1:
            # Exact replay of 193030: that diagnostic applied only the final
            # pose constraints, not intermediate clip_denoised processing.
            actions = raw_clean
            break
        clean = clip_clean(torch, raw_clean, position_limit)
        if index == len(schedule) - 1:
            actions = clean
            break
        next_step = schedule[index + 1]
        alpha_bar = model.alpha_bars[step]
        next_alpha_bar = model.alpha_bars[next_step]
        predicted_noise = (
            actions - alpha_bar.sqrt() * raw_clean
        ) / (1 - alpha_bar).sqrt()
        actions = (
            next_alpha_bar.sqrt() * clean
            + (1 - next_alpha_bar).sqrt() * predicted_noise
        )
    return actions


def mean_metrics(rows):
    keys = rows[0].keys()
    return {
        key: (float(np.mean([row[key] for row in rows]))
              if all(row[key] is not None for row in rows) else None)
        for key in keys
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--head-run", type=Path, required=True)
    parser.add_argument("--window-reference", type=Path, required=True)
    parser.add_argument("--head-report", type=Path, required=True)
    parser.add_argument("--one-step-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", default="1,2,4,8,16,32,50,100")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    step_counts = [int(value) for value in args.steps.split(",")]
    if len(step_counts) != len(set(step_counts)) or step_counts[0] != 1:
        raise ValueError("Step counts must be unique and begin with 1")
    for count in step_counts:
        timestep_schedule(count)

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
    from evaluate_bridge_endpoint_metrics import scalar_metrics, oracle_minimum

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    source_report = json.loads(args.head_report.read_text())
    one_step_reference = json.loads(args.one_step_report.read_text())
    window_reference = json.loads(args.window_reference.read_text())
    if not (source_report["passed"] and source_report["phase"] == "train"):
        raise ValueError("Source training report did not pass")

    manifest = args.pack / "manifest.json"
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
    if window_sha != source_report["protocol"]["window_sha256"]:
        raise ValueError("Window hash differs from source training")
    items = [dataset[row["dataset_index"]] for row in rows]
    targets = np.stack([item[3].numpy() for item in items]).astype(np.float32)

    checkpoint_path = args.head_run / "diffusion-final.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = RobotAdapterModel(checkpoint["config"])
    if checkpoint["split_indices"] != splits:
        raise ValueError("Checkpoint split mismatch")
    if set(checkpoint["trainable_state_dict"]) != set(trainable_state_dict(model)):
        raise ValueError("Checkpoint parameter mismatch")
    model.load_state_dict(checkpoint["trainable_state_dict"], strict=False)
    model = model.cuda().eval()
    if not (model.decoder_type == "diffusion" and
            model.diffusion_prediction_type == "sample" and
            model.num_diffusion_steps == 100 and model.clip_denoised):
        raise ValueError("Unexpected diffusion configuration")
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
    noise_hashes = [hashlib.sha256() for _ in range(3)]
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
                noise_hashes[seed].update(initial.cpu().contiguous().numpy().tobytes())
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
                        raise ValueError(f"Non-finite DDIM prediction at {count} evaluations")
                    predictions[count][seed, start:start + len(batch_items)] = value.cpu().numpy()
            if (start + len(batch_items)) % 200 == 0:
                print("DDIM_PROGRESS", start + len(batch_items), "/", len(rows), flush=True)

    if [value.hexdigest() for value in noise_hashes] != one_step_reference["initial_noise_sha256"]:
        raise ValueError("Initial noise differs from the 193030 one-step run")
    saved = {}
    for name in ("diffusion", "regression"):
        path = args.head_run / f"{name}-predictions.npz"
        with np.load(path) as archive:
            saved[name] = np.asarray(archive["predictions"], dtype=np.float32)
            saved_target = np.asarray(archive["targets"], dtype=np.float32)
        if not np.array_equal(saved_target, targets):
            raise ValueError(f"Saved {name} targets differ")

    result_groups = {}
    for group, indices in groups.items():
        target = targets[indices]
        static = np.zeros_like(target)
        static[..., 6] = 1
        result_groups[group] = {
            "windows": len(indices),
            "static": scalar_metrics(static, target),
            "regression": scalar_metrics(saved["regression"][0, indices], target),
            "saved_full_ddpm": [scalar_metrics(draw[indices], target) for draw in saved["diffusion"]],
            "ddim": {
                str(count): [scalar_metrics(draw[indices], target)
                             for draw in predictions[count]]
                for count in step_counts
            },
            "ddim_oracle": {
                str(count): oracle_minimum(predictions[count][:, indices], target)
                for count in step_counts
            },
        }

    current_one = result_groups["validation/overall"]["ddim"]["1"]
    prior_one = one_step_reference["groups"]["validation/overall"]["one_step_x0"]
    replay_difference = max(
        abs(current_one[seed][metric] - prior_one[seed][metric])
        for seed in range(3)
        for metric in ("path_position_cm", "path_rotation_deg",
                       "endpoint_position_cm", "endpoint_rotation_deg")
    )
    if replay_difference > 1e-6:
        raise ValueError(f"One-step replay differs by {replay_difference}")

    report = {
        "stage": "frozen_same_checkpoint_deterministic_ddim_step_sweep",
        "trained": False,
        "weights_updated": False,
        "reserved_test_targets_read": False,
        "step_counts_are_unet_evaluations": step_counts,
        "eta": 0.0,
        "timestep_schedules": {str(count): timestep_schedule(count) for count in step_counts},
        "source": {
            "head_report_sha256": sha256(args.head_report),
            "one_step_report_sha256": sha256(args.one_step_report),
            "window_reference_sha256": sha256(args.window_reference),
            "checkpoint_sha256": sha256(checkpoint_path),
            "window_sha256": window_sha,
        },
        "counts": {partition: sum(row["partition"] == partition for row in rows)
                   for partition in ("train", "validation")},
        "sampling_seeds": [0, 1, 2],
        "initial_noise_sha256": [value.hexdigest() for value in noise_hashes],
        "one_step_replay_max_metric_difference": replay_difference,
        "groups": result_groups,
        "elapsed_seconds": time.monotonic() - started,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024 ** 3,
        "limits": [
            "This is an inference-only DDIM eta=0 ablation of a DDPM-trained x0 network.",
            "Development results may choose a future schedule but are not held-out test performance.",
            "The 16th target is a window endpoint, not task completion.",
            "Oracle metrics inspect ground truth and are not deployable.",
            "Offline errors do not establish robotic task success.",
        ],
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    validation = result_groups["validation/overall"]
    print("DDIM VALIDATION", json.dumps({
        count: mean_metrics(validation["ddim"][str(count)])
        for count in step_counts
    }), flush=True)
    print("DDIM STEP SWEEP: PASSED", args.output, flush=True)


if __name__ == "__main__":
    main()

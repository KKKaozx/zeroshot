"""Evaluate the frozen multiscale x0 checkpoint against saved paired controls."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

os.environ.update(USE_TF="0", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def extended_metrics(prediction, target, pose_errors):
    position, rotation = pose_errors(prediction, target)
    true_open = target[..., 7] > 0
    predicted_open = prediction[..., 7] > 0
    classes = np.unique(true_open)
    recalls = [
        float((predicted_open[true_open == value] == value).mean())
        for value in classes
    ]
    result = {
        "windows": len(target),
        "action_targets": int(np.prod(target.shape[:2])),
        "path_position_cm": float(position.mean()),
        "path_rotation_deg": float(rotation.mean()),
        "gripper_accuracy": float((predicted_open == true_open).mean()),
        "gripper_balanced_accuracy": float(np.mean(recalls)),
        "gripper_class_count": int(len(classes)),
        "endpoint_position_cm": float(position[:, -1].mean()),
        "endpoint_rotation_deg": float(rotation[:, -1].mean()),
        "endpoint_gripper_accuracy": float(
            (predicted_open[:, -1] == true_open[:, -1]).mean()
        ),
    }
    for name, before, after in (
        ("grasp", True, False),
        ("release", False, True),
    ):
        event = (true_open[:, :-1] == before) & (true_open[:, 1:] == after)
        predicted_event = (
            (predicted_open[:, :-1] == before)
            & (predicted_open[:, 1:] == after)
        )
        count = int(event.sum())
        result[name + "_events"] = count
        result[name + "_exact_timing_accuracy"] = (
            float(predicted_event[event].mean()) if count else None
        )
        result[name + "_position_cm_at_true_event"] = (
            float(position[:, 1:][event].mean()) if count else None
        )
        result[name + "_rotation_deg_at_true_event"] = (
            float(rotation[:, 1:][event].mean()) if count else None
        )
    return result


def groups_for(rows):
    groups = {}
    for partition in ("train", "validation"):
        groups[partition + "/overall"] = [
            index for index, row in enumerate(rows) if row["partition"] == partition
        ]
        for task in sorted(
            {row["task"] for row in rows if row["partition"] == partition}
        ):
            groups[partition + "/task/" + task] = [
                index
                for index, row in enumerate(rows)
                if row["partition"] == partition and row["task"] == task
            ]
        episodes = sorted(
            {
                (row["shard"], row["record_index"])
                for row in rows
                if row["partition"] == partition
            }
        )
        for shard, record in episodes:
            groups[f"{partition}/episode/{shard}::{record}"] = [
                index
                for index, row in enumerate(rows)
                if row["partition"] == partition
                and (row["shard"], row["record_index"]) == (shard, record)
            ]
    return groups


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--training-run", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--window-reference", type=Path, required=True)
    parser.add_argument("--compact-run", type=Path, required=True)
    parser.add_argument("--compact-report", type=Path, required=True)
    parser.add_argument("--head-run", type=Path, required=True)
    parser.add_argument("--head-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)

    script_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(args.pack.resolve()))
    sys.path.insert(0, str(script_dir))
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    import torch
    from transformers import CLIPTokenizer
    from dataset import UnifiedRobotDataset
    from models import RobotAdapterModel
    from train import (
        bridge_plan_selection,
        bridge_plan_splits,
        collate_batch,
        set_seed,
        trainable_state_dict,
    )
    from evaluate_bridge_endpoint_metrics import pose_errors
    from evaluate_bridge_one_step_sampler import (
        decode_state_gripper,
        finish_pose,
        select_windows,
    )

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    training = json.loads(args.training_report.read_text())
    compact = json.loads(args.compact_report.read_text())
    head = json.loads(args.head_report.read_text())
    reference = json.loads(args.window_reference.read_text())
    if not (
        training.get("passed")
        and training.get("stage") == "multiscale_x0_training"
        and training.get("trained")
        and not training.get("evaluated")
        and not training.get("reserved_test_targets_read")
    ):
        raise ValueError("Multiscale training report is not eligible")
    if not (
        compact.get("stage") == "same_checkpoint_one_step_x0_vs_full_ddpm"
        and not compact.get("weights_updated")
        and compact.get("sampling_seeds") == [0, 1, 2]
        and head.get("passed")
        and head.get("phase") == "train"
    ):
        raise ValueError("Compact control reports are not eligible")
    if training["protocol"]["source_head_report_sha256"] != sha256(args.head_report):
        raise ValueError("Training report does not bind to the head control")
    if training["protocol"]["window_reference_sha256"] != sha256(args.window_reference):
        raise ValueError("Training report does not bind to the frozen windows")
    if compact["source"]["head_report_sha256"] != sha256(args.head_report):
        raise ValueError("Compact report does not bind to the head control")

    module_paths = {
        name: (
            script_dir / name
            if name in {"models.py", "diffusion_decoder.py"}
            else args.pack / name
        )
        for name in training["protocol"]["module_sha256"]
    }
    module_hashes = {name: sha256(path) for name, path in module_paths.items()}
    if module_hashes != training["protocol"]["module_sha256"]:
        raise ValueError("Evaluation modules differ from the training run")

    selection = bridge_plan_selection(args.pack / "manifest.json")
    dataset = UnifiedRobotDataset(
        data_dir=str(args.pack / "data"),
        chunk_size=16,
        stride=4,
        sources=["tfrecord"],
        min_trajectory_steps=17,
        exclude_path_parts=[],
        exclude_schemas=[],
        tfrecord_splits=["train"],
        bridge_gripper_policy="reverse_scan_valid_steps_v2",
        bridge_current_gripper="continuous",
        bridge_episode_selection=selection,
    )
    splits = bridge_plan_splits(dataset)
    rows = select_windows(dataset, splits, selection)
    if rows != reference["window_selection"]:
        raise ValueError("Window population differs from the frozen reference")
    window_sha = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    if window_sha != compact["source"]["window_sha256"]:
        raise ValueError("Window hash differs from compact evaluation")
    items = [dataset[row["dataset_index"]] for row in rows]
    targets = np.stack([item[3].numpy() for item in items]).astype(np.float32)

    checkpoint_path = args.training_run / "final.pt"
    if sha256(checkpoint_path) != training["checkpoint"]["sha256"]:
        raise ValueError("Multiscale checkpoint hash differs from training report")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint["protocol"] != training["protocol"]:
        raise ValueError("Checkpoint protocol differs from training report")
    model = RobotAdapterModel(checkpoint["config"])
    if not (
        model.decoder_type == "diffusion"
        and model.diffusion_prediction_type == "sample"
        and model.diffusion_architecture == "multiscale"
        and model.num_diffusion_steps == 100
        and model.separate_gripper_head
    ):
        raise ValueError("Unexpected multiscale checkpoint configuration")
    expected_keys = set(trainable_state_dict(model))
    if set(checkpoint["model"]) != expected_keys:
        raise ValueError("Checkpoint parameter set differs from the model")
    model.load_state_dict(checkpoint["model"], strict=False)
    model = model.cuda().eval()
    tokenizer = CLIPTokenizer.from_pretrained(
        checkpoint["config"]["model"]["name"], local_files_only=True
    )

    compact_predictions_path = args.compact_run / "one-step-predictions.npz"
    regression_predictions_path = args.head_run / "regression-predictions.npz"
    with np.load(compact_predictions_path) as saved:
        compact_predictions = np.asarray(saved["predictions"], dtype=np.float32)
        compact_targets = np.asarray(saved["targets"], dtype=np.float32)
    with np.load(regression_predictions_path) as saved:
        regression_predictions = np.asarray(saved["predictions"], dtype=np.float32)
        regression_targets = np.asarray(saved["targets"], dtype=np.float32)
    if not (
        compact_predictions.shape == (3, len(rows), 16, 8)
        and regression_predictions.shape == (1, len(rows), 16, 8)
        and np.array_equal(targets, compact_targets)
        and np.array_equal(targets, regression_targets)
    ):
        raise ValueError("Saved paired controls differ from current targets")

    predictions = np.empty((3, len(rows), 16, 8), np.float32)
    noise_hashes = [hashlib.sha256() for _ in range(3)]
    checkpoint_hash_before = sha256(checkpoint_path)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    with torch.inference_mode():
        for start in range(0, len(rows), 2):
            batch_items = items[start:start + 2]
            language, images, current, _, _ = collate_batch(batch_items)
            tokens = tokenizer(
                language, padding=True, truncation=True, return_tensors="pt"
            )
            context = model.get_context_vector(
                images.cuda(),
                tokens["input_ids"].cuda(),
                tokens["attention_mask"].cuda(),
            )
            for seed in range(3):
                set_seed(seed * 10000 + start)
                noise = torch.randn(len(batch_items), 16, 7, device="cuda")
                noise_hashes[seed].update(noise.cpu().contiguous().numpy().tobytes())
                timestep = torch.full(
                    (len(batch_items),), 99, device="cuda", dtype=torch.long
                )
                pose = model.diffusion_decoder(noise, timestep, context)
                pose = finish_pose(
                    torch, pose, float(model.max_normalized_position or 3.0)
                )
                gripper = decode_state_gripper(
                    torch, model, context, pose, current.cuda()
                )
                value = torch.cat((pose, gripper.unsqueeze(-1)), dim=-1)
                if not torch.isfinite(value).all():
                    raise ValueError("Non-finite multiscale prediction")
                predictions[seed, start:start + len(batch_items)] = value.cpu().numpy()
            if (start // 2 + 1) % 200 == 0:
                print("EVALUATION_PROGRESS", start + len(batch_items), "/", len(rows), flush=True)
    noise_values = [value.hexdigest() for value in noise_hashes]
    if noise_values != compact["initial_noise_sha256"]:
        raise ValueError("Initial noise differs from the compact one-step control")
    if sha256(checkpoint_path) != checkpoint_hash_before:
        raise ValueError("Checkpoint changed during evaluation")

    groups = groups_for(rows)
    result_groups = {}
    for name, indices in groups.items():
        target = targets[indices]
        static = np.zeros_like(target)
        static[..., 6] = 1
        result_groups[name] = {
            "windows": len(indices),
            "static": extended_metrics(static, target, pose_errors),
            "regression": extended_metrics(
                regression_predictions[0, indices], target, pose_errors
            ),
            "compact_x0_one_step": [
                extended_metrics(draw[indices], target, pose_errors)
                for draw in compact_predictions
            ],
            "multiscale_x0_one_step": [
                extended_metrics(draw[indices], target, pose_errors)
                for draw in predictions
            ],
        }

    predictions_path = args.output / "multiscale-one-step-predictions.npz"
    np.savez_compressed(predictions_path, predictions=predictions, targets=targets)
    report = {
        "stage": "multiscale_x0_one_step_paired_evaluation",
        "trained": False,
        "weights_updated": False,
        "reserved_test_targets_read": False,
        "sampling_seeds": [0, 1, 2],
        "initial_noise_sha256": noise_values,
        "source": {
            "training_report_sha256": sha256(args.training_report),
            "checkpoint_sha256": checkpoint_hash_before,
            "window_reference_sha256": sha256(args.window_reference),
            "compact_report_sha256": sha256(args.compact_report),
            "compact_predictions_sha256": sha256(compact_predictions_path),
            "regression_predictions_sha256": sha256(regression_predictions_path),
            "module_sha256": module_hashes,
            "window_sha256": window_sha,
        },
        "counts": {
            "train": len(groups["train/overall"]),
            "validation": len(groups["validation/overall"]),
        },
        "groups": result_groups,
        "predictions": {
            "path": str(predictions_path),
            "sha256": sha256(predictions_path),
        },
        "elapsed_seconds": time.monotonic() - started,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "limits": [
            "Development results compare frozen final epoch-20 checkpoints; no checkpoint selection occurs.",
            "One-step x0 is an offline inference ablation, not a full diffusion rollout.",
            "Overlapping windows are not independent demonstrations.",
            "Offline errors and gripper labels are not robotic task success.",
        ],
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    summary = result_groups["validation/overall"]
    print("VALIDATION", json.dumps(summary, ensure_ascii=False), flush=True)
    print("MULTISCALE X0 ONE-STEP EVALUATION: PASSED", args.output, flush=True)


if __name__ == "__main__":
    main()

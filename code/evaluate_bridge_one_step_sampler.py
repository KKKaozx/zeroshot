"""Compare one-step x0 readout with saved full DDPM samples from the same model."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

os.environ.update(USE_TF="0", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def finish_pose(torch, pose, position_limit):
    """Apply the same final pose constraints as RobotAdapterModel.sample."""
    pose = pose.clone()
    pose[..., :3] = pose[..., :3].clamp(-position_limit, position_limit)
    quaternion = pose[..., 3:7]
    identity = torch.zeros_like(quaternion)
    identity[..., 3] = 1.0
    pose[..., 3:7] = torch.where(
        quaternion.norm(dim=-1, keepdim=True) > 1e-6,
        torch.nn.functional.normalize(quaternion, dim=-1),
        identity,
    )
    return pose


def decode_state_gripper(torch, model, context, pose, current):
    if model.gripper_target_mode != "state":
        raise ValueError("This paired evaluation expects the saved state gripper head")
    logits = model.predict_gripper_logits(context, pose, current)
    return torch.where(logits >= 0, torch.ones_like(logits), -torch.ones_like(logits))


def select_windows(dataset, splits, selection):
    """Recreate the frozen source-order population without an external probe module."""
    lookup = {(row["shard"], row["record_index"]): row for row in selection}
    rows = []
    for partition in ("train", "validation"):
        ordered = sorted(
            splits[partition],
            key=lambda index: (
                Path(dataset.samples[index]["file_path"]).name,
                dataset.samples[index]["record_index"],
                dataset.samples[index]["start_index"],
            ),
        )
        for index in ordered:
            sample = dataset.samples[index]
            key = Path(sample["file_path"]).name, sample["record_index"]
            rows.append({
                "dataset_index": index,
                "partition": partition,
                "task": lookup[key]["instruction"],
                "shard": key[0],
                "record_index": key[1],
                "start_index": sample["start_index"],
            })
    expected = set(splits["train"]) | set(splits["validation"])
    if {row["dataset_index"] for row in rows} != expected:
        raise ValueError("Selected windows differ from the complete frozen split")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--head-run", type=Path, required=True)
    parser.add_argument("--window-reference", type=Path, required=True)
    parser.add_argument("--head-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)

    sys.path.insert(0, str(args.pack.resolve()))
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    import torch
    from transformers import CLIPTokenizer
    from dataset import UnifiedRobotDataset
    from models import RobotAdapterModel
    from train import (bridge_plan_selection, bridge_plan_splits, collate_batch,
                       set_seed, trainable_state_dict)
    from evaluate_bridge_endpoint_metrics import scalar_metrics, oracle_minimum

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")

    report_source = json.loads(args.head_report.read_text())
    reference = json.loads(args.window_reference.read_text())
    if not (report_source["passed"] and report_source["phase"] == "train"):
        raise ValueError("The source head-control run did not pass")
    if report_source["groups"]["diffusion"]["sampling_seeds"] != [0, 1, 2]:
        raise ValueError("Unexpected source sampling seeds")

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
    expected_window_sha = hashlib.sha256(
        json.dumps(rows, sort_keys=True).encode()
    ).hexdigest()
    if rows != reference["window_selection"]:
        raise ValueError("Window population differs from the frozen reference")
    if expected_window_sha != report_source["protocol"]["window_sha256"]:
        raise ValueError("Window hash differs from the source training run")

    items = [dataset[row["dataset_index"]] for row in rows]
    targets = np.stack([item[3].numpy() for item in items]).astype(np.float32)
    tokenizer = CLIPTokenizer.from_pretrained(
        report_source["protocol"]["source_config"]["model"]["name"],
        local_files_only=True,
    )
    checkpoint_path = args.head_run / "diffusion-final.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint["split_indices"] != splits:
        raise ValueError("Checkpoint split differs from the frozen split")
    model = RobotAdapterModel(checkpoint["config"])
    if not (model.decoder_type == "diffusion" and
            model.diffusion_prediction_type == "sample" and
            model.num_diffusion_steps == 100 and model.separate_gripper_head):
        raise ValueError("Unexpected diffusion checkpoint configuration")
    if set(checkpoint["trainable_state_dict"]) != set(trainable_state_dict(model)):
        raise ValueError("Checkpoint parameter set differs from the archived model")
    model.load_state_dict(checkpoint["trainable_state_dict"], strict=False)
    model = model.cuda().eval()

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
        episodes = sorted({
            (row["shard"], row["record_index"])
            for row in rows if row["partition"] == partition
        })
        for shard, record in episodes:
            groups[f"{partition}/episode/{shard}::{record}"] = [
                i for i, row in enumerate(rows)
                if row["partition"] == partition
                and (row["shard"], row["record_index"]) == (shard, record)
            ]

    predictions = np.empty((3, len(rows), 16, 8), np.float32)
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
                noise = torch.randn(len(batch_items), 16, 7, device="cuda")
                noise_hashes[seed].update(noise.cpu().contiguous().numpy().tobytes())
                timestep = torch.full(
                    (len(batch_items),), 99, device="cuda", dtype=torch.long
                )
                pose = model.diffusion_decoder(noise, timestep, context)
                pose = finish_pose(torch, pose, float(model.max_normalized_position or 3.0))
                gripper = decode_state_gripper(
                    torch, model, context, pose, current.cuda()
                )
                value = torch.cat([pose, gripper.unsqueeze(-1)], dim=-1)
                if not torch.isfinite(value).all():
                    raise ValueError("Non-finite one-step prediction")
                predictions[seed, start:start + len(batch_items)] = value.cpu().numpy()

    saved = {}
    for name in ("diffusion", "regression"):
        path = args.head_run / f"{name}-predictions.npz"
        with np.load(path) as archive:
            saved[name] = np.asarray(archive["predictions"], dtype=np.float32)
            saved_targets = np.asarray(archive["targets"], dtype=np.float32)
        if not np.array_equal(saved_targets, targets):
            raise ValueError(f"Saved {name} targets differ from current targets")
    if saved["diffusion"].shape != predictions.shape or saved["regression"].shape != (1, len(rows), 16, 8):
        raise ValueError("Unexpected saved prediction shapes")

    result_groups = {}
    for group, indices in groups.items():
        target = targets[indices]
        static = np.zeros_like(target)
        static[..., 6] = 1
        result_groups[group] = {
            "windows": len(indices),
            "static": scalar_metrics(static, target),
            "regression": scalar_metrics(saved["regression"][0, indices], target),
            "full_ddpm": [scalar_metrics(draw[indices], target) for draw in saved["diffusion"]],
            "full_ddpm_oracle": oracle_minimum(saved["diffusion"][:, indices], target),
            "one_step_x0": [scalar_metrics(draw[indices], target) for draw in predictions],
            "one_step_x0_oracle": oracle_minimum(predictions[:, indices], target),
        }

    np.savez_compressed(
        args.output / "one-step-predictions.npz",
        predictions=predictions, targets=targets,
    )
    report = {
        "stage": "same_checkpoint_one_step_x0_vs_full_ddpm",
        "trained": False,
        "weights_updated": False,
        "reserved_test_targets_read": False,
        "semantics": (
            "At t=99, feed pure Gaussian noise to the saved x0-prediction U-Net once; "
            "apply the production final pose and gripper decoding; compare with saved full 100-step samples."
        ),
        "source": {
            "head_report_sha256": sha256(args.head_report),
            "window_reference_sha256": sha256(args.window_reference),
            "checkpoint_sha256": sha256(checkpoint_path),
            "diffusion_predictions_sha256": sha256(args.head_run / "diffusion-predictions.npz"),
            "regression_predictions_sha256": sha256(args.head_run / "regression-predictions.npz"),
            "window_sha256": expected_window_sha,
        },
        "counts": {partition: sum(row["partition"] == partition for row in rows)
                   for partition in ("train", "validation")},
        "sampling_seeds": [0, 1, 2],
        "initial_noise_sha256": [value.hexdigest() for value in noise_hashes],
        "alpha_bar_t99": float(model.alpha_bars[99]),
        "groups": result_groups,
        "elapsed_seconds": time.monotonic() - started,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024 ** 3,
        "limits": [
            "The 16th target is a fixed-window endpoint, not task completion.",
            "One-step x0 is an inference ablation of the trained DDPM, not a separately trained consistency model.",
            "Oracle metrics inspect ground truth independently per scalar and are not deployable.",
            "Offline errors cannot establish collision-free execution or robotic task success.",
        ],
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("ONE STEP", json.dumps(result_groups["validation/overall"], ensure_ascii=False), flush=True)
    print("SAME-CHECKPOINT ONE-STEP EVALUATION: PASSED", args.output, flush=True)


if __name__ == "__main__":
    main()

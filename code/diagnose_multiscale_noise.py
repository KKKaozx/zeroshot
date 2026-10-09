"""Frozen training-window probe of noise level and image conditioning."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

os.environ.update(USE_TF="0", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")


def select_training(rows, per_task=8):
    selected, seen, counts = [], set(), {}
    for row in rows:
        if row["partition"] != "train":
            continue
        episode = (row["shard"], row["record_index"])
        task = row["task"]
        if episode in seen or counts.get(task, 0) >= per_task:
            continue
        selected.append(row)
        seen.add(episode)
        counts[task] = counts.get(task, 0) + 1
    if len(counts) != 5 or any(value != per_task for value in counts.values()):
        raise ValueError("Expected eight independent training demonstrations per task")
    donors = []
    for index, row in enumerate(selected):
        donors.append(next(i for i, other in enumerate(selected)
                           if other["task"] == row["task"] and i != index))
    return selected, donors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pack", "training-run", "training-report", "window-reference", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    script_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(args.pack.resolve()))
    sys.path.insert(0, str(script_dir))
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    import torch
    from transformers import CLIPTokenizer
    from models import RobotAdapterModel
    from dataset import UnifiedRobotDataset
    from train import bridge_plan_selection, bridge_plan_splits, collate_batch, set_seed, trainable_state_dict
    from evaluate_bridge_one_step_sampler import finish_pose, select_windows, sha256
    from evaluate_bridge_endpoint_metrics import pose_errors

    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    source = json.loads(args.training_report.read_text())
    reference = json.loads(args.window_reference.read_text())
    if not source["passed"] or source["stage"] != "multiscale_x0_training":
        raise ValueError("Invalid source training report")
    if source["protocol"]["window_reference_sha256"] != sha256(args.window_reference):
        raise ValueError("Window reference differs")
    module_hashes = {name: sha256((script_dir if name in {"models.py", "diffusion_decoder.py"}
                                   else args.pack) / name)
                     for name in source["protocol"]["module_sha256"]}
    if module_hashes != source["protocol"]["module_sha256"]:
        raise ValueError("Model/data modules differ from training")
    selection = bridge_plan_selection(args.pack / "manifest.json")
    dataset = UnifiedRobotDataset(
        data_dir=str(args.pack / "data"), chunk_size=16, stride=4,
        sources=["tfrecord"], min_trajectory_steps=17, exclude_path_parts=[],
        exclude_schemas=[], tfrecord_splits=["train"],
        bridge_gripper_policy="reverse_scan_valid_steps_v2",
        bridge_current_gripper="continuous", bridge_episode_selection=selection)
    rows = select_windows(dataset, bridge_plan_splits(dataset), selection)
    if rows != reference["window_selection"]:
        raise ValueError("Population differs from frozen reference")
    selected, donors = select_training(rows)
    items = [dataset[row["dataset_index"]] for row in selected]
    language, images, _, targets, masks = collate_batch(items)
    if not (masks[..., :7] > .5).all():
        raise ValueError("Probe requires fully supervised pose targets")
    if any(torch.equal(images[i], images[j]) for i, j in enumerate(donors)):
        raise ValueError("A donor image is identical after production preprocessing")
    checkpoint_path = args.training_run / "final.pt"
    initial_hash = sha256(checkpoint_path)
    if initial_hash != source["checkpoint"]["sha256"]:
        raise ValueError("Checkpoint hash differs")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint["protocol"] != source["protocol"]:
        raise ValueError("Checkpoint protocol differs")
    model = RobotAdapterModel(checkpoint["config"])
    if model.diffusion_architecture != "multiscale" or model.diffusion_prediction_type != "sample":
        raise ValueError("Expected multiscale x0 model")
    if set(checkpoint["model"]) != set(trainable_state_dict(model)):
        raise ValueError("Checkpoint keys differ")
    model.load_state_dict(checkpoint["model"], strict=False)
    model.cuda().eval()
    tokenizer = CLIPTokenizer.from_pretrained(checkpoint["config"]["model"]["name"], local_files_only=True)
    contexts = {"correct": [], "image_swap": []}
    with torch.inference_mode():
        for start in range(0, len(items), 2):
            ids = list(range(start, min(start + 2, len(items))))
            tokens = tokenizer([language[i] for i in ids], padding=True, truncation=True, return_tensors="pt")
            for kind in contexts:
                image_ids = ids if kind == "correct" else [donors[i] for i in ids]
                contexts[kind].append(model.get_context_vector(
                    images[image_ids].cuda(), tokens["input_ids"].cuda(), tokens["attention_mask"].cuda()).cpu())
    contexts = {key: torch.cat(value) for key, value in contexts.items()}
    truth = targets[..., :7].numpy()
    cases = {}
    began = time.monotonic()
    with torch.inference_mode():
        for label, step in [("t0", 0), ("t24", 24), ("t49", 49), ("t74", 74), ("t99", 99), ("pure_noise_t99", 99)]:
            kinds = ("correct", "image_swap") if label == "pure_noise_t99" else ("correct",)
            cases[label] = {}
            outputs = {}
            for kind in kinds:
                seeds = []
                for seed in range(3):
                    raw = np.empty_like(truth)
                    clean = np.empty_like(truth)
                    for start in range(0, len(items), 2):
                        target = targets[start:start + 2, :, :7].cuda()
                        set_seed(seed * 10000 + start)
                        noise = torch.randn_like(target)
                        alpha = model.alpha_bars[step]
                        noisy = noise if label == "pure_noise_t99" else alpha.sqrt() * target + (1-alpha).sqrt() * noise
                        timestep = torch.full((len(target),), step, device="cuda", dtype=torch.long)
                        value = model.diffusion_decoder(noisy, timestep, contexts[kind][start:start + 2].cuda())
                        if not torch.isfinite(value).all():
                            raise ValueError("Nonfinite output")
                        raw[start:start + len(target)] = value.cpu().numpy()
                        clean[start:start + len(target)] = finish_pose(torch, value, float(model.max_normalized_position or 3)).cpu().numpy()
                    seeds.append((raw, clean))
                outputs[kind] = seeds
                groups = {"overall": list(range(len(items)))}
                groups.update({"task/" + task: [i for i,r in enumerate(selected) if r["task"] == task]
                               for task in sorted({r["task"] for r in selected})})
                cases[label][kind] = {}
                for group, ids in groups.items():
                    scores = []
                    for seed, (raw, clean) in enumerate(seeds):
                        pos, rot = pose_errors(clean[ids], truth[ids])
                        value = {"seed": seed, "position_cm": float(pos.mean()), "rotation_deg": float(rot.mean()),
                                 "raw_component_mse": float(((raw[ids]-truth[ids])**2).mean()),
                                 "raw_xyz_mse": float(((raw[ids,:,:3]-truth[ids,:,:3])**2).mean()),
                                 "raw_quaternion_mse": float(((raw[ids,:,3:]-truth[ids,:,3:])**2).mean())}
                        if kind == "image_swap":
                            change = clean[ids] - outputs["correct"][seed][1][ids]
                            value["paired_position_change_cm"] = float(np.linalg.norm(change[..., :3], axis=-1).mean()*10)
                        scores.append(value)
                    cases[label][kind][group] = scores
                print("NOISE_PROBE", label, kind, cases[label][kind]["overall"], flush=True)
    if sha256(checkpoint_path) != initial_hash:
        raise ValueError("Checkpoint changed")
    static = np.zeros_like(truth)
    static[..., 6] = 1
    pos, rot = pose_errors(static, truth)
    report = {"stage": "frozen_multiscale_noise_image_probe", "trained": False, "weights_updated": False,
              "reserved_test_targets_read": False, "validation_payload_read": False,
              "training_report_sha256": sha256(args.training_report), "checkpoint_sha256": initial_hash,
              "module_sha256": module_hashes, "window_selection": selected, "image_donor_indices": donors,
              "training_windows": len(selected), "sampling_seeds": [0,1,2],
              "alpha_bars": {str(t): float(model.alpha_bars[t]) for t in [0,24,49,74,99]},
              "static": {"position_cm": float(pos.mean()), "rotation_deg": float(rot.mean())},
              "cases": cases, "elapsed_seconds": time.monotonic()-began,
              "limits": ["Forty fixed training windows, one per demonstration; not full training evaluation.",
                         "Noisy-target cases contain ground truth and are reconstruction diagnostics only.",
                         "Image swaps test the use of the observed scene, not semantic correctness.",
                         "No automatic threshold can by itself establish a unique root cause."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("FROZEN MULTISCALE NOISE/IMAGE PROBE: PASSED", args.output, flush=True)


if __name__ == "__main__":
    main()

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


def restoration_comparison(noisy_raw, noisy_pose, output_pose, target, pose_errors):
    """Compare the same noisy input before and after the learned restoration."""
    input_position, input_rotation = pose_errors(noisy_pose, target)
    output_position, output_rotation = pose_errors(output_pose, target)
    return {
        "input_position_cm": float(input_position.mean()),
        "input_rotation_deg": float(input_rotation.mean()),
        "position_improvement_cm": float((input_position-output_position).mean()),
        "rotation_improvement_deg": float((input_rotation-output_rotation).mean()),
        "position_improved_target_fraction": float((output_position < input_position).mean()),
        "rotation_improved_target_fraction": float((output_rotation < input_rotation).mean()),
        "input_raw_xyz_mse": float(((noisy_raw[..., :3]-target[..., :3])**2).mean()),
        "input_raw_quaternion_mse": float(((noisy_raw[..., 3:]-target[..., 3:])**2).mean()),
    }


def traced_ddim(torch, model, context, noise, ddim_sample):
    """Observe one unchanged deterministic chain; remove hook on all exits."""
    trace, calls = {}, []
    def observe(module, inputs, output):
        step = int(inputs[1][0])
        calls.append(step)
        if step in (99, 49, 24, 9, 0):
            trace[step] = (inputs[0].detach().cpu().clone(), output.detach().cpu().clone())
    hook = model.diffusion_decoder.register_forward_hook(observe)
    try:
        final = ddim_sample(torch, model, context, noise, 100)
    finally:
        hook.remove()
    if calls != list(range(99, -1, -1)):
        raise ValueError("Expected a full 100-step DDIM chain")
    return final, trace


def run_tail_probe(args, model, contexts, targets, selected, donors, source, module_hashes, initial_hash):
    import torch
    from train import set_seed
    from evaluate_bridge_ddim_steps import ddim_sample
    from evaluate_bridge_one_step_sampler import finish_pose, sha256
    from evaluate_bridge_endpoint_metrics import pose_errors

    truth = targets[..., :7].numpy()
    started = time.monotonic()
    steps = (99, 49, 24, 9, 0)
    groups = {"overall": list(range(len(selected)))}
    groups.update({"task/" + task: [i for i,r in enumerate(selected) if r["task"] == task]
                   for task in sorted({r["task"] for r in selected})})
    results = {group: [] for group in groups}
    limit = float(model.max_normalized_position or 3)
    with torch.inference_mode():
        for seed in range(3):
            inputs = {step: np.empty_like(truth) for step in steps}
            x0s = {step: np.empty_like(truth) for step in steps}
            final = np.empty_like(truth)
            for start in range(0, len(selected), 2):
                context = contexts["correct"][start:start + 2].cuda()
                set_seed(seed * 10000 + start)
                noise = torch.randn(len(context), 16, 7, device="cuda")
                completed, trace = traced_ddim(torch, model, context, noise, ddim_sample)
                final[start:start + len(context)] = finish_pose(torch, completed, limit).cpu().numpy()
                for step in steps:
                    noisy, estimate = trace[step]
                    inputs[step][start:start + len(context)] = finish_pose(torch, noisy, limit).numpy()
                    x0s[step][start:start + len(context)] = finish_pose(torch, estimate, limit).numpy()
            for group, ids in groups.items():
                def score(value):
                    pos, rot = pose_errors(value[ids], truth[ids])
                    return {"position_cm": float(pos.mean()), "rotation_deg": float(rot.mean())}
                final_score = score(final)
                stage_results = {}
                for step in steps:
                    input_score, x0_score = score(inputs[step]), score(x0s[step])
                    input_pos, input_rot = pose_errors(inputs[step][ids], truth[ids])
                    final_pos, final_rot = pose_errors(final[ids], truth[ids])
                    stage_results[str(step)] = {
                        "input_before_step": input_score,
                        "clean_estimate_at_step": x0_score,
                        "final_minus_input_position_cm": final_score["position_cm"]-input_score["position_cm"],
                        "final_minus_input_rotation_deg": final_score["rotation_deg"]-input_score["rotation_deg"],
                        "final_minus_estimate_position_cm": final_score["position_cm"]-x0_score["position_cm"],
                        "final_minus_estimate_rotation_deg": final_score["rotation_deg"]-x0_score["rotation_deg"],
                        "final_worse_than_input_position_fraction": float((final_pos > input_pos).mean()),
                        "final_worse_than_input_rotation_fraction": float((final_rot > input_rot).mean()),
                    }
                results[group].append({"seed": seed, "final": final_score, "steps": stage_results})
            print("TAIL_SEED", seed, json.dumps(results["overall"][-1]), flush=True)
    checkpoint_path = args.training_run / "final.pt"
    if sha256(checkpoint_path) != initial_hash:
        raise ValueError("Checkpoint changed")
    report = {
        "stage": "frozen_multiscale_ddim_tail_probe", "trained": False, "weights_updated": False,
        "reserved_test_targets_read": False, "validation_payload_read": False,
        "training_report_sha256": sha256(args.training_report), "checkpoint_sha256": initial_hash,
        "reference_probe_sha256": sha256(args.reference_probe), "module_sha256": module_hashes,
        "window_selection": selected, "image_donor_indices": donors, "sampling_seeds": [0,1,2],
        "sampler": "100-step DDIM eta=0, unchanged existing ddim_sample implementation",
        "trace_steps": list(steps), "groups": results, "elapsed_seconds": time.monotonic()-started,
        "difference_sign": "Positive final-minus-input/estimate means continuing increases target error.",
        "limits": [
            "Same forty training windows; no development or test payloads.",
            "Scoring uses target data, but generation never receives target actions.",
            "Input snapshot is still a noisy latent with final pose constraints; estimate snapshot is predicted x0.",
            "These snapshots are offline ablations, not executable early-stop policies.",
            "Tail results concern this deterministic DDIM chain, not every sampler or training parameterization.",
            "No timing or checkpoint is chosen as best on these diagnostic results.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("FROZEN MULTISCALE DDIM TAIL PROBE: PASSED", args.output, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pack", "training-run", "training-report", "window-reference", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--reference-probe", type=Path)
    parser.add_argument("--chain-tail", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.chain_tail and args.reference_probe is None:
        raise ValueError("Chain-tail probe requires a prior reconstruction report")
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
    prior_probe = None
    if args.reference_probe is not None:
        prior_probe = json.loads(args.reference_probe.read_text())
        if (prior_probe["window_selection"] != selected
                or prior_probe["image_donor_indices"] != donors
                or prior_probe["checkpoint_sha256"] != source["checkpoint"]["sha256"]):
            raise ValueError("Probe subset or checkpoint differs from 194940")
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
    contexts = {"correct": []} if args.chain_tail else {"correct": [], "image_swap": []}
    with torch.inference_mode():
        for start in range(0, len(items), 2):
            ids = list(range(start, min(start + 2, len(items))))
            tokens = tokenizer([language[i] for i in ids], padding=True, truncation=True, return_tensors="pt")
            for kind in contexts:
                image_ids = ids if kind == "correct" else [donors[i] for i in ids]
                contexts[kind].append(model.get_context_vector(
                    images[image_ids].cuda(), tokens["input_ids"].cuda(), tokens["attention_mask"].cuda()).cpu())
    contexts = {key: torch.cat(value) for key, value in contexts.items()}
    if args.chain_tail:
        run_tail_probe(args, model, contexts, targets, selected, donors, source, module_hashes, initial_hash)
        return
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
                    input_raw = np.empty_like(truth)
                    input_pose = np.empty_like(truth)
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
                        input_raw[start:start + len(target)] = noisy.cpu().numpy()
                        input_pose[start:start + len(target)] = finish_pose(torch, noisy, float(model.max_normalized_position or 3)).cpu().numpy()
                    seeds.append((raw, clean, input_raw, input_pose))
                outputs[kind] = seeds
                groups = {"overall": list(range(len(items)))}
                groups.update({"task/" + task: [i for i,r in enumerate(selected) if r["task"] == task]
                               for task in sorted({r["task"] for r in selected})})
                cases[label][kind] = {}
                for group, ids in groups.items():
                    scores = []
                    for seed, (raw, clean, input_raw, input_pose) in enumerate(seeds):
                        pos, rot = pose_errors(clean[ids], truth[ids])
                        value = {"seed": seed, "position_cm": float(pos.mean()), "rotation_deg": float(rot.mean()),
                                 "raw_component_mse": float(((raw[ids]-truth[ids])**2).mean()),
                                 "raw_xyz_mse": float(((raw[ids,:,:3]-truth[ids,:,:3])**2).mean()),
                                 "raw_quaternion_mse": float(((raw[ids,:,3:]-truth[ids,:,3:])**2).mean())}
                        value.update(restoration_comparison(
                            input_raw[ids], input_pose[ids], clean[ids], truth[ids], pose_errors))
                        norms = np.linalg.norm(raw[ids, :, 3:], axis=-1)
                        value["output_raw_quaternion_norm_mean"] = float(norms.mean())
                        value["output_degenerate_quaternions"] = int((norms <= 1e-6).sum())
                        if kind == "image_swap":
                            change = clean[ids] - outputs["correct"][seed][1][ids]
                            value["paired_position_change_cm"] = float(np.linalg.norm(change[..., :3], axis=-1).mean()*10)
                        scores.append(value)
                    cases[label][kind][group] = scores
                overall = cases[label][kind]["overall"]
                mean = lambda key: float(np.mean([row[key] for row in overall]))
                print("RESTORATION", label, kind,
                      "input=", round(mean("input_position_cm"), 4), round(mean("input_rotation_deg"), 4),
                      "output=", round(mean("position_cm"), 4), round(mean("rotation_deg"), 4),
                      "improvement=", round(mean("position_improvement_cm"), 4), round(mean("rotation_improvement_deg"), 4),
                      flush=True)
    if sha256(checkpoint_path) != initial_hash:
        raise ValueError("Checkpoint changed")
    static = np.zeros_like(truth)
    static[..., 6] = 1
    pos, rot = pose_errors(static, truth)
    report = {"stage": "frozen_multiscale_noise_image_probe", "trained": False, "weights_updated": False,
              "schema_version": 2,
              "input_baseline": "Same noisy input with production position clipping and quaternion normalization; positive improvement means output error is lower.",
              "reference_probe_sha256": sha256(args.reference_probe) if args.reference_probe is not None else None,
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
    if prior_probe is not None:
        report["prior_overall_metric_differences"] = {
            label: {
                kind: [{key: row[key]-prior_probe["cases"][label][kind]["overall"][seed][key]
                        for key in ("position_cm", "rotation_deg")}
                       for seed, row in enumerate(groups["overall"])]
                for kind, groups in variants.items()
            }
            for label, variants in cases.items()
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("FROZEN MULTISCALE NOISE/IMAGE PROBE: PASSED", args.output, flush=True)


if __name__ == "__main__":
    main()

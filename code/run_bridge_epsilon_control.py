"""Train an epsilon-target control with the frozen 192651 data and initialization."""
import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

os.environ.update(USE_TF="0", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
HEADS = ("regression_head.", "diffusion_decoder.")


def orders(indices, epochs=20):
    rng = np.random.default_rng(42)
    return [rng.permutation(indices).tolist() for _ in range(epochs)]


def digest(state):
    value = hashlib.sha256()
    for key, tensor in sorted(state.items()):
        value.update(key.encode())
        value.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


def common_state(state):
    return {key: value.clone() for key, value in state.items()
            if not key.startswith(HEADS)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--head-run", type=Path, required=True)
    parser.add_argument("--head-report", type=Path, required=True)
    parser.add_argument("--window-reference", type=Path, required=True)
    parser.add_argument("--one-step-report", type=Path, required=True)
    parser.add_argument("--ddim-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("preflight", "train"), required=True)
    parser.add_argument("--preflight", type=Path)
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
                       set_seed, trainable_state_dict, policy_loss,
                       build_learning_rate_scheduler)
    from evaluate_bridge_one_step_sampler import (
        decode_state_gripper, finish_pose, select_windows, sha256,
    )
    from evaluate_bridge_ddim_steps import ddim_sample
    from evaluate_bridge_endpoint_metrics import scalar_metrics

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    source_report = json.loads(args.head_report.read_text())
    one_step_report = json.loads(args.one_step_report.read_text())
    ddim_report = json.loads(args.ddim_report.read_text())
    window_reference = json.loads(args.window_reference.read_text())
    if not (source_report["passed"] and source_report["phase"] == "train"):
        raise ValueError("Source head-control report did not pass")
    if one_step_report["source"]["head_report_sha256"] != sha256(args.head_report):
        raise ValueError("One-step report does not bind to the source report")
    if ddim_report["source"]["head_report_sha256"] != sha256(args.head_report):
        raise ValueError("DDIM report does not bind to the source report")

    manifest = args.pack / "manifest.json"
    module_hashes = {
        name: sha256(args.pack / name)
        for name in source_report["protocol"]["module_sha256"]
    }
    if module_hashes != source_report["protocol"]["module_sha256"]:
        raise ValueError("Archived training modules differ from 192651")
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
    schedule = orders([i for i, row in enumerate(rows) if row["partition"] == "train"])
    order_sha = hashlib.sha256(json.dumps(schedule).encode()).hexdigest()
    if order_sha != source_report["protocol"]["order_sha256"]:
        raise ValueError("Training order differs from 192651")
    items = [dataset[row["dataset_index"]] for row in rows]
    targets = np.stack([item[3].numpy() for item in items]).astype(np.float32)

    source_config = copy.deepcopy(source_report["protocol"]["source_config"])
    regression_config = copy.deepcopy(source_config)
    regression_config["model"]["decoder_type"] = "regression"
    set_seed(42)
    regression_model = RobotAdapterModel(regression_config)
    shared = common_state(trainable_state_dict(regression_model))
    del regression_model
    gc.collect()

    epsilon_config = copy.deepcopy(source_config)
    epsilon_config["model"]["decoder_type"] = "diffusion"
    epsilon_config["model"]["diffusion_prediction_type"] = "epsilon"
    set_seed(42)
    model = RobotAdapterModel(epsilon_config)
    model.load_state_dict(shared, strict=False)
    initial_shared = digest(common_state(trainable_state_dict(model)))
    expected_shared = source_report["groups"]["diffusion"]["initial_shared_sha256"]
    if initial_shared != expected_shared:
        raise ValueError("Shared initialization differs from the x0 control")
    initial_trainable = digest(trainable_state_dict(model))
    encoders = lambda: {key: value for key, value in model.state_dict().items()
                        if key.startswith(("vision_encoder.", "text_encoder."))}
    clip_hash = digest(encoders())
    if clip_hash != source_report["groups"]["diffusion"]["clip_sha256"]:
        raise ValueError("Frozen CLIP initialization differs")
    model = model.cuda()
    if not (model.diffusion_prediction_type == "epsilon" and
            model.decoder_type == "diffusion" and model.num_diffusion_steps == 100):
        raise ValueError("Unexpected epsilon model configuration")

    tokenizer = CLIPTokenizer.from_pretrained(
        epsilon_config["model"]["name"], local_files_only=True
    )
    tokens = tokenizer(
        [item[0] for item in items], padding=True, truncation=True,
        return_tensors="pt",
    )

    def batch(indices):
        _, images, current, actions, mask = collate_batch([items[i] for i in indices])
        return (
            images.cuda(), tokens["input_ids"][indices].cuda(),
            tokens["attention_mask"][indices].cuda(), current.cuda(),
            actions.cuda(), mask.cuda(),
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

    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4)
    scheduler = build_learning_rate_scheduler(optimizer, "cosine", 20)
    protocol = {
        "source_head_report_sha256": sha256(args.head_report),
        "source_one_step_report_sha256": sha256(args.one_step_report),
        "source_ddim_report_sha256": sha256(args.ddim_report),
        "window_reference_sha256": sha256(args.window_reference),
        "module_sha256": module_hashes,
        "order_sha256": order_sha,
        "initial_shared_sha256": initial_shared,
        "initial_trainable_sha256": initial_trainable,
        "clip_sha256": clip_hash,
        "only_intended_change": "diffusion_prediction_type sample(x0) -> epsilon",
        "epochs": 20, "batch_size": 2, "updates": 18360,
        "optimizer": "AdamW", "learning_rate": 1e-4,
        "weight_decay": 1e-4, "lr_schedule": "cosine_per_epoch",
        "gradient_clip_norm": 1.0, "evaluation": "8-step DDIM eta=0",
    }
    if args.phase == "train":
        if args.preflight is None:
            raise ValueError("Training requires a passed preflight report")
        preflight = json.loads(args.preflight.read_text())
        if not (preflight["passed"] and preflight["phase"] == "preflight"
                and preflight["protocol"] == protocol):
            raise ValueError("Preflight does not authorize this exact protocol")

    torch.cuda.reset_peak_memory_stats()
    times, history, updates = [], [], 0
    for epoch, order in enumerate(schedule, 1):
        model.train()
        losses = []
        for start in range(0, len(order), 2):
            if args.phase == "preflight" and updates == 6:
                break
            images, text, attention, current, actions, mask = batch(order[start:start + 2])
            set_seed(420000 + updates)
            torch.cuda.synchronize()
            began = time.monotonic()
            output = model(
                images, text, attention_mask=attention,
                current_gripper=current, actions=actions,
            )
            loss, pose, gripper = policy_loss(
                model, output, actions, torch.nn.MSELoss(),
                current_grippers=current, supervision_masks=mask,
            )
            if not torch.isfinite(loss):
                raise ValueError("Non-finite epsilon training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
            optimizer.step()
            torch.cuda.synchronize()
            times.append(time.monotonic() - began)
            losses.append(float(loss))
            updates += 1
            if updates == 1:
                for prefix in ("adapter.", "diffusion_decoder.", "gripper_head."):
                    if not any(parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
                               for name, parameter in model.named_parameters()
                               if name.startswith(prefix)):
                        raise ValueError(f"No first-update gradient for {prefix}")
            if updates == 1 or updates % 200 == 0 or args.phase == "preflight":
                print("EPSILON_UPDATE", updates, "loss", float(loss),
                      "pose", float(pose), "gripper", float(gripper), flush=True)
        scheduler.step()
        history.append({"epoch": epoch, "updates": updates,
                        "mean_loss": float(np.mean(losses))})
        if args.phase == "preflight":
            break
        (args.output / "history.json").write_text(json.dumps(history, indent=2) + "\n")

    def evaluate(indices, seeds):
        model.eval()
        predictions = np.empty((len(seeds), len(indices), 16, 8), np.float32)
        with torch.inference_mode():
            for start in range(0, len(indices), 2):
                ids = indices[start:start + 2]
                images, text, attention, current, _, _ = batch(ids)
                context = model.get_context_vector(images, text, attention)
                for seed_index, seed in enumerate(seeds):
                    set_seed(seed * 10000 + start)
                    initial = torch.randn(len(ids), 16, 7, device="cuda")
                    pose = ddim_sample(torch, model, context, initial, 8)
                    pose = finish_pose(
                        torch, pose, float(model.max_normalized_position or 3.0)
                    )
                    gripper = decode_state_gripper(
                        torch, model, context, pose, current
                    )
                    value = torch.cat([pose, gripper.unsqueeze(-1)], dim=-1)
                    if not torch.isfinite(value).all():
                        raise ValueError("Non-finite epsilon DDIM prediction")
                    predictions[seed_index, start:start + len(ids)] = value.cpu().numpy()
        return predictions

    evaluation_ids = list(range(len(rows))) if args.phase == "train" else [
        i for i, row in enumerate(rows) if row["partition"] == "train"
    ][:8]
    seeds = [0, 1, 2] if args.phase == "train" else [0]
    model.eval()
    torch.cuda.synchronize()
    evaluation_began = time.monotonic()
    predictions = evaluate(evaluation_ids, seeds)
    torch.cuda.synchronize()
    evaluation_seconds = time.monotonic() - evaluation_began
    result = {
        "phase": args.phase,
        "passed": True,
        "trained": True,
        "reserved_test_targets_read": False,
        "protocol": protocol,
        "updates": updates,
        "trainable_parameters": sum(parameter.numel() for parameter in params),
        "warm_update_seconds": float(np.mean(times[1:])),
        "evaluation_seconds": evaluation_seconds,
        "evaluation_windows": len(evaluation_ids),
        "sampling_seeds": seeds,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024 ** 3,
        "clip_unchanged": digest(encoders()) == clip_hash,
        "trainable_state_changed": digest(trainable_state_dict(model)) != initial_trainable,
    }
    if not (result["clip_unchanged"] and result["trainable_state_changed"]):
        raise ValueError("Frozen/trainable parameter contract failed")
    if args.phase == "preflight":
        estimate = (
            result["warm_update_seconds"] * 18360
            + evaluation_seconds / len(evaluation_ids) * len(rows) * 3
        ) * 1.5 / 3600
        result["estimated_train_and_evaluation_hours_with_50_percent_margin"] = estimate
        result["passed"] = estimate < 1.4 and result["peak_reserved_gib"] < 30
    else:
        if updates != 18360:
            raise ValueError("Formal epsilon run did not complete 18360 updates")
        result["groups"] = {
            group: [scalar_metrics(draw[indices], targets[indices])
                    for draw in predictions]
            for group, indices in groups.items()
        }
        result["x0_references"] = {
            "one_step_validation": one_step_report["groups"]["validation/overall"]["one_step_x0"],
            "eight_step_validation": ddim_report["groups"]["validation/overall"]["ddim"]["8"],
        }
        torch.save({
            "config": epsilon_config, "epoch": 20,
            "split_indices": splits,
            "trainable_state_dict": trainable_state_dict(model),
        }, args.output / "epsilon-final.pt")
        np.savez_compressed(
            args.output / "epsilon-ddim8-predictions.npz",
            predictions=predictions, targets=targets,
        )
    (args.output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print("EPSILON CONTROL", args.phase, "PASSED", result["passed"],
          args.output / "report.json", flush=True)
    if not result["passed"]:
        raise RuntimeError("Epsilon resource preflight failed")


if __name__ == "__main__":
    main()

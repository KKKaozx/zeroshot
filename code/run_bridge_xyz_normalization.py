"""Matched xyz normalization control using the frozen multiscale x0 protocol."""

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


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


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
    return {
        key: value.clone()
        for key, value in state.items()
        if not key.startswith(HEADS)
    }


class TrainableEMA:
    """Shadow trainable parameters; match the verified author's EMA warmup."""

    def __init__(self, model):
        import torch
        if any(isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
               for module in model.modules()):
            raise ValueError("This EMA scope assumes the archived model has no BatchNorm")
        self.shadow = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        self.optimization_step = 0
        self.decay = 0.0
        self.recipe = {
            "update_after_step": 0, "inv_gamma": 1.0, "power": 0.75,
            "min_value": 0.0, "max_value": 0.9999,
            "update_timing": "after_each_optimizer_step",
            "scope": "all_trainable_parameters; frozen_CLIP_excluded",
        }

    def step(self, model):
        import torch
        parameters = {
            name: parameter for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        if set(parameters) != set(self.shadow):
            raise ValueError("Trainable parameter set changed during EMA")
        step = max(0, self.optimization_step - 1)
        self.decay = min(0.9999, 1 - (1 + step) ** -0.75) if step else 0.0
        with torch.no_grad():
            # Separate storage and no random operations: optimizer/RNG unaffected.
            values = list(self.shadow.values())
            current = [parameters[name].detach() for name in self.shadow]
            torch._foreach_mul_(values, self.decay)
            torch._foreach_add_(values, current, alpha=1 - self.decay)
        self.optimization_step += 1

    def state_dict(self, raw_state):
        # Fixed buffers keep the current raw value; only parameters are averaged.
        if not set(self.shadow).issubset(raw_state):
            raise ValueError("EMA parameters are not covered by saved trainable state")
        return {
            name: self.shadow[name].detach().cpu().clone()
            if name in self.shadow else value.detach().cpu().clone()
            for name, value in raw_state.items()
        }


def select_windows(dataset, splits, selection):
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
            rows.append(
                {
                    "dataset_index": index,
                    "partition": partition,
                    "task": lookup[key]["instruction"],
                    "shard": key[0],
                    "record_index": key[1],
                    "start_index": sample["start_index"],
                }
            )
    expected = set(splits["train"]) | set(splits["validation"])
    if {row["dataset_index"] for row in rows} != expected:
        raise ValueError("Selected windows differ from the complete frozen split")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--head-report", type=Path, required=True)
    parser.add_argument("--window-reference", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ema", action="store_true", help="Save EMA shadow without changing raw optimization")
    parser.add_argument("--baseline-report", type=Path, help="Bind EMA run to the original 193761 protocol")
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--group", choices=["baseline", "axis_limits"], required=True)
    parser.add_argument("--temporary-preflight", action="store_true")
    args = parser.parse_args()
    if args.ema:
        raise ValueError("EMA is excluded from this control")
    stats = json.loads(args.stats.read_text())
    if not (stats.get("execution_checks_passed") and stats["counts"]["train_windows"] == 1836 and stats["normalizer_fit_partitions"] == ["train"]):
        raise ValueError("Unexpected frozen training statistics")
    normalizer = {"scale": [1.,1.,1.], "offset": [0.,0.,0.]}
    if args.group == "axis_limits":
        normalizer = {"scale": stats["candidate_axis_limits_normalizer"]["scale_from_existing_xyz"], "offset": stats["candidate_axis_limits_normalizer"]["offset"]}
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
    from bridge_xyz_normalization import NormalizedRobotAdapterModel as RobotAdapterModel
    from train import (
        bridge_plan_selection,
        bridge_plan_splits,
        build_learning_rate_scheduler,
        collate_batch,
        policy_loss,
        set_seed,
        trainable_state_dict,
    )

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    source = json.loads(args.head_report.read_text())
    preflight = json.loads(args.preflight.read_text())
    reference = json.loads(args.window_reference.read_text())
    baseline = json.loads(args.baseline_report.read_text()) if args.baseline_report else None
    if baseline is None or stats["training_report_sha256"] != sha256(args.baseline_report):
        raise ValueError("Training statistics must bind the original 193761 training report")
    if args.ema and baseline is None:
        raise ValueError("EMA comparison requires the original training report")
    if baseline is not None:
        if not (baseline.get("passed") and baseline.get("stage") == "multiscale_x0_training"):
            raise ValueError("Unexpected original multiscale baseline report")
        if sha256(Path(baseline["checkpoint"]["path"])) != baseline["checkpoint"]["sha256"]:
            raise ValueError("Original checkpoint is missing or changed")
    if not (source.get("passed") and source.get("phase") == "train"):
        raise ValueError("192651 source report did not pass")
    if not (
        preflight.get("passed")
        and preflight.get("stage") == "official_style_unet_real_bridge_preflight"
        and preflight.get("temporary_updates_discarded")
        and preflight.get("protocol", {}).get("source_head_report_sha256")
        == sha256(args.head_report)
        and preflight.get("protocol", {}).get("window_reference_sha256")
        == sha256(args.window_reference)
    ):
        raise ValueError("193711 preflight does not authorize this source protocol")

    module_paths = {
        name: (
            script_dir / name
            if name in {"models.py", "diffusion_decoder.py"}
            else args.pack / name
        )
        for name in source["protocol"]["module_sha256"]
    }
    module_hashes = {name: sha256(path) for name, path in module_paths.items()}
    for name, expected in source["protocol"]["module_sha256"].items():
        if name not in {"models.py", "diffusion_decoder.py"} and module_hashes[name] != expected:
            raise ValueError(f"Unexpected archived module change: {name}")
    if module_hashes != preflight["protocol"]["module_sha256"]:
        raise ValueError("Core model modules differ from the passed real-batch preflight")

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
    train_indices = [
        index for index, row in enumerate(rows) if row["partition"] == "train"
    ]
    schedule = orders(train_indices)
    order_sha = hashlib.sha256(json.dumps(schedule).encode()).hexdigest()
    if order_sha != source["protocol"]["order_sha256"]:
        raise ValueError("Training order differs from 192651")
    # Materialize training payloads only. Validation rows are identity-indexed
    # for protocol matching but their images/actions are not read in this job.
    items = {
        index: dataset[rows[index]["dataset_index"]] for index in train_indices
    }

    if stats["window_reference_sha256"] != sha256(args.window_reference):
        raise ValueError("Statistics refer to a different window population")
    source_config = copy.deepcopy(source["protocol"]["source_config"])
    regression_config = copy.deepcopy(source_config)
    regression_config["model"]["decoder_type"] = "regression"
    set_seed(42)
    source_model = RobotAdapterModel(regression_config)
    shared = common_state(trainable_state_dict(source_model))
    del source_model
    gc.collect()

    config = copy.deepcopy(source_config)
    config["model"].update(
        decoder_type="diffusion",
        diffusion_prediction_type="sample",
        diffusion_architecture="multiscale",
        diffusion_down_dims=[256, 512, 1024],
        diffusion_step_embed_dim=128,
        diffusion_kernel_size=5,
    )
    config["xyz_normalizer"] = normalizer
    set_seed(42)
    model = RobotAdapterModel(config)
    model.load_state_dict(shared, strict=False)
    initial_shared = digest(common_state(trainable_state_dict(model)))
    if initial_shared != source["groups"]["diffusion"]["initial_shared_sha256"]:
        raise ValueError("Shared initialization differs from the compact x0 control")
    initial_trainable = digest(trainable_state_dict(model))
    encoders = lambda: {
        key: value
        for key, value in model.state_dict().items()
        if key.startswith(("vision_encoder.", "text_encoder."))
    }
    clip_hash = digest(encoders())
    if clip_hash != source["groups"]["diffusion"]["clip_sha256"]:
        raise ValueError("Frozen CLIP initialization differs from 192651")
    initial_groups = {
        "adapter": digest({k: v for k, v in trainable_state_dict(model).items() if k.startswith("adapter.")}),
        "decoder": digest({k: v for k, v in trainable_state_dict(model).items() if k.startswith("diffusion_decoder.")}),
        "gripper": digest({k: v for k, v in trainable_state_dict(model).items() if not k.startswith(("adapter.", "diffusion_decoder."))}),
    }

    tokenizer = CLIPTokenizer.from_pretrained(
        config["model"]["name"], local_files_only=True
    )
    tokens = tokenizer(
        [items[index][0] for index in train_indices],
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    token_position = {index: position for position, index in enumerate(train_indices)}
    model = model.cuda()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=1e-4, weight_decay=1e-4)
    scheduler = build_learning_rate_scheduler(optimizer, "cosine", 20)
    ema = TrainableEMA(model) if args.ema else None

    def batch(indices):
        _, images, current, actions, mask = collate_batch([items[index] for index in indices])
        positions = [token_position[index] for index in indices]
        return (
            images.cuda(),
            tokens["input_ids"][positions].cuda(),
            tokens["attention_mask"][positions].cuda(),
            current.cuda(),
            actions.cuda(),
            mask.cuda(),
        )

    protocol = {
        "source_head_report_sha256": sha256(args.head_report),
        "window_reference_sha256": sha256(args.window_reference),
        "preflight_sha256": sha256(args.preflight),
        "module_sha256": module_hashes,
        "order_sha256": order_sha,
        "initial_shared_sha256": initial_shared,
        "initial_trainable_sha256": initial_trainable,
        "clip_sha256": clip_hash,
        "only_intended_change_from_compact_x0": "compact -> multiscale Conditional U-Net",
        "prediction_type": "sample_x0",
        "architecture": "multiscale_256_512_1024_kernel5_film",
        "epochs": 20,
        "batch_size": 2,
        "updates": 18360,
        "optimizer": "AdamW",
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "lr_schedule": "cosine_per_epoch",
        "gradient_clip_norm": 1.0,
        "evaluation": "separate job after checkpoint is safely saved",
    }
    if ema is not None:
        for key in ("module_sha256", "order_sha256", "initial_shared_sha256",
                    "initial_trainable_sha256", "clip_sha256", "prediction_type",
                    "architecture", "epochs", "batch_size", "updates", "optimizer",
                    "learning_rate", "weight_decay", "lr_schedule", "gradient_clip_norm"):
            if protocol[key] != baseline["protocol"][key]:
                raise ValueError(f"EMA control differs from baseline: {key}")
        protocol["ema"] = ema.recipe
        protocol["baseline_report_sha256"] = sha256(args.baseline_report)
        protocol["only_intended_change_from_compact_x0"] = "193761 raw training unchanged; EMA shadow added"
        protocol["runner_sha256"] = sha256(Path(__file__))
    protocol.update(normalization_group=args.group, xyz_normalizer=normalizer,
                    stats_sha256=sha256(args.stats), runner_sha256=sha256(Path(__file__)),
                    normalization_module_sha256=sha256(script_dir / "bridge_xyz_normalization.py"),
                    loss_xyz_inverse_scale_squared_compensation=True,
                    physical_xyz_clipping_unchanged=True, gripper_condition_physical_units=True,
                    only_intended_change_from_compact_x0="paired xyz normalization with physical-unit MSE preserved",
                    temporary_preflight=args.temporary_preflight)
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")

    torch.cuda.reset_peak_memory_stats()
    began_training = time.monotonic()
    history = []
    durations = []
    updates = 0
    first_gradient = None
    for epoch, order in enumerate(schedule, 1):
        model.train()
        epoch_losses = []
        epoch_pose = []
        epoch_gripper = []
        for start in range(0, len(order), 2):
            images, text, attention, current, actions, mask = batch(order[start:start + 2])
            set_seed(420000 + updates)
            torch.cuda.synchronize()
            began = time.monotonic()
            output = model(
                images,
                text,
                attention_mask=attention,
                current_gripper=current,
                actions=actions,
            )
            loss, pose, gripper = policy_loss(
                model,
                output,
                actions,
                torch.nn.MSELoss(),
                current_grippers=current,
                supervision_masks=mask,
            )
            if not torch.isfinite(loss):
                raise ValueError("Non-finite multiscale training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if updates == 0:
                first_gradient = {
                    prefix: any(
                        parameter.grad is not None
                        and torch.count_nonzero(parameter.grad).item() > 0
                        for name, parameter in model.named_parameters()
                        if name.startswith(prefix + ".")
                    )
                    for prefix in ("adapter", "diffusion_decoder", "gripper_head")
                }
                if not all(first_gradient.values()):
                    raise ValueError(f"Missing first-update gradient: {first_gradient}")
            torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            optimizer.step()
            if ema is not None:
                ema.step(model)
            torch.cuda.synchronize()
            durations.append(time.monotonic() - began)
            epoch_losses.append(float(loss))
            epoch_pose.append(float(pose))
            epoch_gripper.append(float(gripper))
            updates += 1
            if updates == 1 or updates % 250 == 0:
                print(
                    "MULTISCALE_UPDATE",
                    updates,
                    "loss",
                    float(loss),
                    "pose",
                    float(pose),
                    "gripper",
                    float(gripper),
                    flush=True,
                )
            if args.temporary_preflight and updates == 6:
                break
        scheduler.step()
        row = {
            "epoch": epoch,
            "updates": updates,
            "mean_loss": float(np.mean(epoch_losses)),
            "mean_pose_loss": float(np.mean(epoch_pose)),
            "mean_gripper_loss": float(np.mean(epoch_gripper)),
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        (args.output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        print("MULTISCALE_EPOCH", row, flush=True)

        if args.temporary_preflight:
            break

    if args.temporary_preflight:
        check = {"stage": "xyz_normalization_real_preflight", "execution_checks_passed": updates == 6 and all(first_gradient.values()),
                 "group": args.group, "protocol": protocol, "temporary_updates_discarded": True,
                 "updates": updates, "initial_trainable_sha256": initial_trainable,
                 "clip_unchanged": digest(encoders()) == clip_hash,
                 "trainable_parameters_changed": digest(trainable_state_dict(model)) != initial_trainable,
                 "peak_reserved_gib": torch.cuda.max_memory_reserved()/2**30,
                 "mean_warm_update_seconds": float(np.mean(durations[1:])), "losses": epoch_losses}
        (args.output / "report.json").write_text(json.dumps(check,indent=2)+"\n")
        if not (check["execution_checks_passed"] and check["clip_unchanged"] and check["trainable_parameters_changed"]):
            raise RuntimeError("Normalization preflight failed")
        print("XYZ NORMALIZATION PREFLIGHT PASSED", args.group, args.output / "report.json", flush=True)
        return
    if updates != 18360:
        raise ValueError(f"Expected 18360 updates, got {updates}")
    final_state = trainable_state_dict(model)
    checkpoint = args.output / "final.pt"
    torch.save(
        {"config": config, "model": final_state, "protocol": protocol}, checkpoint
    )
    checkpoint_hash = sha256(checkpoint)
    ema_checkpoint = None
    raw_replay_matches = None
    if ema is not None:
        if ema.optimization_step != updates:
            raise ValueError("EMA update count differs from optimizer update count")
        ema_checkpoint = args.output / "ema.pt"
        torch.save({"config": config, "model": ema.state_dict(final_state),
                    "protocol": protocol, "ema_updates": ema.optimization_step}, ema_checkpoint)
        original = torch.load(baseline["checkpoint"]["path"], map_location="cpu", weights_only=False)
        raw_replay_matches = digest(original["model"]) == digest(final_state)
        del original
    final_groups = {
        "adapter": digest({k: v for k, v in final_state.items() if k.startswith("adapter.")}),
        "decoder": digest({k: v for k, v in final_state.items() if k.startswith("diffusion_decoder.")}),
        "gripper": digest({k: v for k, v in final_state.items() if not k.startswith(("adapter.", "diffusion_decoder."))}),
    }
    module_changed = {
        name: initial_groups[name] != final_groups[name] for name in initial_groups
    }
    clip_unchanged = digest(encoders()) == clip_hash
    passed = bool(clip_unchanged and all(module_changed.values()) and checkpoint.exists())
    report = {
        "stage": "xyz_normalization_training",
        "group": args.group,
        "passed": passed,
        "trained": True,
        "evaluated": False,
        "reserved_test_targets_read": False,
        "protocol": protocol,
        "counts": {"train_windows": 1836, "validation_windows_indexed_only": 40, "updates": updates},
        "parameters": {
            "all": sum(parameter.numel() for parameter in model.parameters()),
            "trainable": sum(parameter.numel() for parameter in parameters),
            "decoder": sum(parameter.numel() for parameter in model.diffusion_decoder.parameters()),
        },
        "first_update_nonzero_gradient": first_gradient,
        "module_changed": module_changed,
        "clip_unchanged": clip_unchanged,
        "checkpoint": {"path": str(checkpoint), "sha256": checkpoint_hash, "bytes": checkpoint.stat().st_size},
        "history_path": str(args.output / "history.json"),
        "elapsed_seconds": time.monotonic() - began_training,
        "mean_warm_update_seconds": sum(durations[1:]) / len(durations[1:]),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "limits": [
            "Final epoch-20 weights are saved before any evaluation.",
            "No development or held-out test target was used for checkpoint selection.",
            "Offline evaluation must run as a separate immutable-checkpoint job.",
        ],
    }
    if ema_checkpoint is not None:
        report["ema_checkpoint"] = {"path": str(ema_checkpoint), "sha256": sha256(ema_checkpoint),
                                    "bytes": ema_checkpoint.stat().st_size}
        report["ema_updates"] = ema.optimization_step
        report["ema_final_decay"] = ema.decay
        report["raw_final_tensor_replay_matches_193761"] = raw_replay_matches
        report["limits"].append("Training integrity passing is not an EMA quality improvement result.")
        print("EMA SAVED", ema_checkpoint, "raw replay matches:", raw_replay_matches, flush=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("MULTISCALE X0 TRAINING", "PASSED" if passed else "FAILED")
    print("CHECKPOINT", checkpoint, checkpoint_hash)
    print("REPORT", args.output / "report.json")
    if not passed:
        raise RuntimeError("Multiscale x0 training integrity checks failed")


if __name__ == "__main__":
    main()

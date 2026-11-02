"""Run six real Bridge updates through CLIP, adapter and multiscale U-Net."""

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


def state_digest(state: dict, prefixes: tuple[str, ...]) -> str:
    value = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        if name.startswith(prefixes):
            value.update(name.encode())
            value.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


def select_windows(dataset, splits, selection):
    """Recreate the frozen source-order population used by the controls."""
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
    parser.add_argument("--resource-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    script_dir = Path(__file__).resolve().parent
    # Prefer the two integration modules shipped beside this driver; resolve
    # unchanged dataset/adapter/training helpers from the archived 192651 pack.
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
        policy_loss,
        set_seed,
    )
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    resource = json.loads(args.resource_report.read_text())
    if not resource.get("passed") or resource.get("stage") != "decoder_resource_and_gradient_preflight":
        raise ValueError("Resource preflight did not pass")
    source = json.loads(args.head_report.read_text())
    if not (source.get("passed") and source.get("phase") == "train"):
        raise ValueError("192651 source report did not pass")
    window_reference = json.loads(args.window_reference.read_text())
    module_paths = {
        name: (
            script_dir / name
            if name in {"models.py", "diffusion_decoder.py"}
            else args.pack / name
        )
        for name in source["protocol"]["module_sha256"]
    }
    module_hashes = {name: sha256(path) for name, path in module_paths.items()}
    # models.py and diffusion_decoder.py are the two intentional integration changes.
    for name, expected in source["protocol"]["module_sha256"].items():
        if name not in {"models.py", "diffusion_decoder.py"} and module_hashes[name] != expected:
            raise ValueError(f"Unexpected archived module change: {name}")

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
    if rows != window_reference["window_selection"]:
        raise ValueError("Window population differs from the frozen reference")
    train_indices = [
        index for index, row in enumerate(rows) if row["partition"] == "train"
    ]
    order = np.random.default_rng(42).permutation(train_indices).tolist()
    selected = order[:12]
    items = {index: dataset[rows[index]["dataset_index"]] for index in selected}

    config = json.loads(json.dumps(source["protocol"]["source_config"]))
    config["model"].update(
        decoder_type="diffusion",
        diffusion_prediction_type="sample",
        diffusion_architecture="multiscale",
        diffusion_down_dims=[256, 512, 1024],
        diffusion_step_embed_dim=128,
        diffusion_kernel_size=5,
    )
    set_seed(42)
    model = RobotAdapterModel(config)
    if model.diffusion_architecture != "multiscale":
        raise ValueError("Multiscale decoder was not selected")
    tokenizer = CLIPTokenizer.from_pretrained(
        config["model"]["name"], local_files_only=True
    )
    model = model.cuda()
    model.train()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-4, weight_decay=1e-4)
    prefixes = {
        "clip": ("vision_encoder.", "text_encoder."),
        "adapter": ("adapter.",),
        "decoder": ("diffusion_decoder.",),
        "gripper": (
            "gripper_context_projection.",
            "gripper_pose_projection.",
            "current_gripper_projection.",
            "gripper_time_embedding.",
            "gripper_head.",
        ),
    }
    before = {
        name: state_digest(model.state_dict(), group_prefixes)
        for name, group_prefixes in prefixes.items()
    }
    torch.cuda.reset_peak_memory_stats()
    losses = []
    durations = []
    first_gradient = None
    for update, start in enumerate(range(0, 12, 2), 1):
        indices = selected[start:start + 2]
        batch_items = [items[index] for index in indices]
        text, images, current, actions, mask = collate_batch(batch_items)
        tokens = tokenizer(
            text, padding=True, truncation=True, return_tensors="pt"
        )
        set_seed(420000 + update - 1)
        torch.cuda.synchronize()
        began = time.monotonic()
        output = model(
            images.cuda(),
            tokens["input_ids"].cuda(),
            attention_mask=tokens["attention_mask"].cuda(),
            current_gripper=current.cuda(),
            actions=actions.cuda(),
        )
        loss, pose, gripper = policy_loss(
            model,
            output,
            actions.cuda(),
            torch.nn.MSELoss(),
            current_grippers=current.cuda(),
            supervision_masks=mask.cuda(),
        )
        if not torch.isfinite(loss):
            raise ValueError("Non-finite real-batch loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if update == 1:
            first_gradient = {
                name: any(
                    parameter.grad is not None
                    and torch.count_nonzero(parameter.grad).item() > 0
                    for parameter_name, parameter in model.named_parameters()
                    if parameter_name.startswith(group_prefixes)
                )
                for name, group_prefixes in prefixes.items()
            }
        torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
        optimizer.step()
        torch.cuda.synchronize()
        durations.append(time.monotonic() - began)
        losses.append(
            {"update": update, "loss": float(loss), "pose": float(pose), "gripper": float(gripper)}
        )
        print("REAL_UNET_UPDATE", update, losses[-1], flush=True)

    after = {
        name: state_digest(model.state_dict(), group_prefixes)
        for name, group_prefixes in prefixes.items()
    }
    changed = {name: before[name] != after[name] for name in prefixes}
    passed = bool(
        first_gradient
        and not first_gradient["clip"]
        and all(first_gradient[name] for name in ("adapter", "decoder", "gripper"))
        and not changed["clip"]
        and all(changed[name] for name in ("adapter", "decoder", "gripper"))
        and torch.cuda.max_memory_reserved() / 2**30 < 32
    )
    report = {
        "stage": "official_style_unet_real_bridge_preflight",
        "passed": passed,
        "trained_policy": False,
        "temporary_updates_discarded": True,
        "reserved_test_targets_read": False,
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "protocol": {
            "source_head_report_sha256": sha256(args.head_report),
            "window_reference_sha256": sha256(args.window_reference),
            "resource_report_sha256": sha256(args.resource_report),
            "module_sha256": module_hashes,
            "updates": 6,
            "batch_size": 2,
            "prediction_type": "sample_x0",
            "architecture": "multiscale_256_512_1024_kernel5_film",
            "evaluation": "none",
        },
        "counts": {"all_windows": len(rows), "train_windows": len(train_indices), "updates": 6},
        "parameters": {
            "all": sum(parameter.numel() for parameter in model.parameters()),
            "trainable": sum(parameter.numel() for parameter in trainable),
            "decoder": sum(parameter.numel() for parameter in model.diffusion_decoder.parameters()),
        },
        "first_update_nonzero_gradient": first_gradient,
        "module_changed": changed,
        "losses": losses,
        "mean_warm_update_seconds": sum(durations[1:]) / len(durations[1:]),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "limits": [
            "Six temporary real Bridge updates only; resulting weights are discarded.",
            "No accuracy, generation, development or test evaluation was run.",
            "Passing permits preparation of a paired full-training protocol only.",
        ],
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("REAL BRIDGE OFFICIAL U-NET PREFLIGHT", "PASSED" if passed else "FAILED")
    print("REPORT", args.output)
    if not passed:
        raise RuntimeError("Real Bridge official-style U-Net preflight failed")


if __name__ == "__main__":
    main()

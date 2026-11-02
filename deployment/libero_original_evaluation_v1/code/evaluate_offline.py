"""Evaluate a checkpoint on held-out trajectories without robot simulation.

By default this reconstructs the exact dataset and test split stored by
``train.py``. Passing ``--dataset-dir`` evaluates all compatible trajectories
under another root, for example the held-out LIBERO directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from transformers import CLIPTokenizer

from dataset import ACTION_REPRESENTATION, POSITION_SCALE_METERS, UnifiedRobotDataset
from models import RobotAdapterModel
from train import (
    collate_batch,
    evaluate_action_metrics,
    evaluate_loss,
    set_seed,
    tokenise,
    dataset_split_identity,
)


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def make_loader(
    dataset: UnifiedRobotDataset,
    indices: list[int],
    batch_size: int,
    seed: int,
) -> DataLoader:
    # 文件扫描顺序往往会把同一任务、同一夹爪状态集中在一起。先用固定种子
    # 打乱完整评估索引，再截取 max_batches，既避免“只测开头文件”的偏差，
    # 又保证同一命令可重复得到相同样本。
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(indices), generator=generator).tolist()
    shuffled_indices = [indices[position] for position in order]
    return DataLoader(
        Subset(dataset, shuffled_indices),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=0,
    )


def metric_coverage(dataset, ordered_indices, batch_size, max_batches, samples_per_batch):
    """按实际loader顺序记录被测窗口；多个重叠窗口不等于多条独立演示。"""
    selected = []
    for start in range(0, min(len(ordered_indices), batch_size * max_batches), batch_size):
        batch = ordered_indices[start:start + batch_size]
        selected.extend(batch[:samples_per_batch])
    return {"sampled_windows": len(selected),
            "sampled_episodes": len({dataset.group_key(i) for i in selected}),
            "available_episodes": len({dataset.group_key(i) for i in ordered_indices}),
            "sampled_indices": selected}


def select_episode_windows(dataset, indices, limit, seed):
    """每条演示最多随机保留limit个窗口；只按身份抽样，不按目标状态挑样本。"""
    if limit < 0:
        raise ValueError("每条演示窗口上限不能为负")
    if limit == 0:
        return list(indices)
    groups = defaultdict(list)
    for i in indices:
        groups[dataset.group_key(i)].append(i)
    generator = torch.Generator().manual_seed(seed)
    selected = []
    for values in groups.values():
        order = torch.randperm(len(values), generator=generator).tolist()
        selected.extend(values[i] for i in order[:limit])
    return selected


def print_metrics(label: str, loss: float, metrics: dict[str, float]) -> None:
    print(
        f"[离线评估] {label}: sampled_windows="
        f"{int(metrics.get('metric_trajectories', 0))}, "
        f"sampled_actions={int(metrics.get('metric_action_steps', 0))}, "
        f"loss={loss:.6f}, "
        f"position={metrics.get('position_error_cm', float('nan')):.2f}cm, "
        f"rotation={metrics.get('rotation_error_deg', float('nan')):.2f}°, "
        f"gripper={metrics.get('gripper_accuracy', float('nan')):.1%}, "
        f"teacher_pose_gripper={metrics.get('gripper_teacher_pose_accuracy', float('nan')):.1%}, "
        f"zero_motion_position={metrics.get('zero_motion_position_error_cm', float('nan')):.2f}cm, "
        f"identity_rotation={metrics.get('identity_rotation_error_deg', float('nan')):.2f}°, "
        f"balanced={metrics.get('gripper_balanced_accuracy', float('nan')):.1%}, "
        f"majority={metrics.get('gripper_majority_baseline', float('nan')):.1%}, "
        f"persistence={metrics.get('gripper_persistence_baseline', float('nan')):.1%}, "
        f"change_rate={metrics.get('gripper_change_rate', float('nan')):.1%}, "
        f"change_acc={metrics.get('gripper_change_accuracy', float('nan')):.1%}, "
        f"hold_acc={metrics.get('gripper_hold_accuracy', float('nan')):.1%}, "
        f"true_open={metrics.get('gripper_true_open_rate', float('nan')):.1%}, "
        f"pred_open={metrics.get('gripper_predicted_open_rate', float('nan')):.1%}, "
        f"action_mae={metrics.get('action_mae', float('nan')):.4f}"
    )


@torch.no_grad()
def export_trajectory_comparisons(model, tokenizer, dataset, indices, device, args):
    """保存逐窗口预测、真实轨迹和多次采样；固定窗口顺序保证对照可比。"""
    import numpy as np
    from audit_dataset import plot_sample

    output = Path(args.trajectory_dir)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for order, index in enumerate(indices[:args.trajectory_samples]):
        instruction, image, gripper, target, mask = dataset[index]
        text = tokenise(tokenizer, [instruction], device)
        context = model.get_context_vector(image.unsqueeze(0).to(device), text["input_ids"], text.get("attention_mask"))
        draws = args.trajectory_draws if model.decoder_type == "diffusion" else 1
        with torch.random.fork_rng(devices=[device.index or 0] if device.type == "cuda" else []):
            set_seed(args.seed + 30_000 + order)
            predictions = model.sample(context.expand(draws, -1), gripper.reshape(1, 1).to(device).expand(draws, -1)).cpu().numpy()
        targets = target.numpy()
        valid = mask.numpy() > .5
        xyz_errors = np.linalg.norm((predictions[..., :3] - targets[None, :, :3]) * valid[None, :, :3], axis=-1) * POSITION_SCALE_METERS * 100
        dot = np.abs((predictions[..., 3:7] * targets[None, :, 3:7]).sum(axis=-1)).clip(0, 1)
        rotation_errors = 2 * np.degrees(np.arccos(dot))
        valid_xyz = valid[:, :3].any(axis=1)
        valid_rotation = valid[:, 3:7].all(axis=1)
        valid_gripper = valid[:, 7]
        # 每个时间步相对于同一个输入时刻，不能将这些位置再累计相加。
        position_mean = float(xyz_errors[:, valid_xyz].mean()) if valid_xyz.any() else None
        rotation_mean = float(rotation_errors[:, valid_rotation].mean()) if valid_rotation.any() else None
        grip_accuracy = float(((predictions[..., 7] >= 0) == (targets[None, :, 7] >= 0))[:, valid_gripper].mean()) if valid_gripper.any() else None
        plot = f"window_{order + 1:02d}.png"
        if not valid.all():
            print(f"[轨迹对比] 窗口 {order + 1} 有缺失监督，仅保存数值，不画占位值")
            plot = None
        else:
            plot_sample(output / plot, f"{model.decoder_type} window={index} draws={draws}", instruction,
                        image, float(gripper.item()), targets, predictions)
        records.append({"dataset_index": index, "sample": dataset.samples[index], "instruction": instruction,
                        "plot": plot, "draws": draws, "target": targets.tolist(), "predictions": predictions.tolist(),
                        "supervision_mask": valid.tolist(), "position_mean_cm": position_mean,
                        "rotation_mean_deg": rotation_mean, "gripper_accuracy": grip_accuracy,
                        "position_by_draw_cm": xyz_errors[:, valid_xyz].mean(axis=1).tolist() if valid_xyz.any() else [],
                        "rotation_by_draw_deg": rotation_errors[:, valid_rotation].mean(axis=1).tolist() if valid_rotation.any() else []})
        print(f"[轨迹对比] 窗口 {order + 1}：位置={position_mean}cm，旋转={rotation_mean}°，抽样={draws}", flush=True)
    (output / "trajectories.json").write_text(json.dumps({"scope": args.split, "decoder_type": model.decoder_type,
        "seed": args.seed, "records": records}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[轨迹对比] 图像与原始预测已保存：{output.resolve()}")


@torch.no_grad()
def diagnose_conditions(model, tokenizer, dataset, split_indices, device, args, fixed_partitions=None):
    """冻结权重；同窗口、同采样噪声，仅替换一个输入条件。"""
    report = {"purpose": "input_condition_diagnostic_not_robot_success",
              "adapter_pooling": getattr(model, "adapter_pooling", None),
              "bcz_target": dataset.bcz_target, "chunk_size": dataset.chunk_size,
              "bcz_current_gripper": getattr(dataset, "bcz_current_gripper", "binary"),
              "checkpoint": str(Path(args.checkpoint).resolve()), "seed": args.seed, "partitions": {}}
    # 可显式复用已保存验证探针；默认的旧拟合诊断行为不变，测试分区禁止干预。
    if fixed_partitions is not None:
        if not fixed_partitions or set(fixed_partitions) - {"train", "validation"}:
            raise ValueError("固定条件诊断仅允许训练或验证分区，不能使用test")
        for name, values in fixed_partitions.items():
            if len(values) != len(set(values)) or not set(values).issubset(split_indices[name]):
                raise ValueError("固定条件窗口必须唯一且属于对应checkpoint分区")
    partition_names = tuple(fixed_partitions) if fixed_partitions is not None else (
        ("train",) if dataset.bcz_target == "native_commands" else ("train", "validation"))
    for partition_number, name in enumerate(partition_names):
        indices = split_indices[name]
        if fixed_partitions is not None:
            chosen = list(fixed_partitions[name])
        else:
            order = torch.randperm(len(indices), generator=torch.Generator().manual_seed(args.seed)).tolist()
            chosen = [indices[i] for i in order[:args.diagnostic_samples]]
        if len(chosen) < 2:
            raise ValueError("条件诊断每个分区至少需要两个窗口")
        items = [dataset[i] for i in chosen]
        results = {key: [] for key in ("original", "swap_image", "swap_language", "flip_current_gripper")}
        contexts = {key: [] for key in results}
        boundary_logits, teacher_boundary_logits = [], []
        for start in range(0, len(items), args.batch_size):
            base = items[start:start + args.batch_size]
            altered = [items[(i + 1) % len(items)] for i in range(start, start + len(base))]
            texts, images, grippers, targets, masks = collate_batch(base)
            other_texts, other_images, _, _, _ = collate_batch(altered)
            for variant in results:
                tokens = tokenise(tokenizer, other_texts if variant == "swap_language" else texts, device)
                pixels = other_images if variant == "swap_image" else images
                context = model.get_context_vector(pixels.to(device), tokens["input_ids"], tokens.get("attention_mask"))
                current = (-grippers if variant == "flip_current_gripper" else grippers).to(device)
                # 所有输入干预复用同一扩散噪声，不能将随机采样差异算作条件影响。
                with torch.random.fork_rng(devices=[device.index or 0] if device.type == "cuda" else []):
                    set_seed(args.seed + 40_000 + partition_number * 1000 + start)
                    prediction = model.sample(context, current).cpu()
                results[variant].append(prediction)
                contexts[variant].append(context.cpu())
                if variant == "original" and dataset.bcz_target == "native_commands":
                    if model.gripper_target_mode != "state":
                        raise ValueError("原生序列边界诊断要求绝对state夹爪监督")
                    boundary_logits.append(model.predict_gripper_logits(context, prediction[..., :7].to(device), current).cpu())
                    teacher_boundary_logits.append(model.predict_gripper_logits(context, targets[..., :7].to(device), current).cpu())
            print(f"[条件诊断] {name}：已处理 {start + len(base)}/{len(items)} 个窗口", flush=True)
        targets = torch.stack([item[3] for item in items])
        masks = torch.stack([item[4] for item in items]) > .5
        current_open = torch.stack([item[2] for item in items]).reshape(-1, 1) >= 0
        truth_open = targets[..., 7] >= 0
        change = truth_open != current_open
        original = torch.cat(results["original"])
        original_context = torch.cat(contexts["original"])
        boundary_diagnostic = None
        if boundary_logits:
            logits = torch.cat(boundary_logits)
            teacher_logits = torch.cat(teacher_boundary_logits)
            predicted_state = logits >= 0
            teacher_state = teacher_logits >= 0
            if not torch.equal(predicted_state, original[..., 7] >= 0):
                raise RuntimeError("导出的夹爪logit与原始推理不一致，拒绝诊断")
            erroneous_rows = ((predicted_state != truth_open) & masks[..., 7]).any(-1).nonzero().flatten().tolist()
            probabilities = logits.sigmoid()
            valid_open = masks[..., 7] & truth_open
            valid_closed = masks[..., 7] & ~truth_open
            min_open = float(probabilities[valid_open].min()) if valid_open.any() else None
            max_closed = float(probabilities[valid_closed].max()) if valid_closed.any() else None
            element_bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, truth_open.float(), reduction="none") * masks[..., 7]
            total_bce = float(element_bce.sum())
            boundary_diagnostic = {
                "scope": f"same_{name}_windows_frozen_weights_no_threshold_tuning",
                "logit_semantics": "positive=open, zero threshold; probability is sigmoid(logit), not calibrated physical probability",
                "open_logits": logits.tolist(), "open_probabilities": logits.sigmoid().tolist(),
                "teacher_pose_open_logits": teacher_logits.tolist(),
                "teacher_pose_open_probabilities": teacher_logits.sigmoid().tolist(),
                "teacher_pose_error_count": int(((teacher_state != truth_open) & masks[..., 7]).sum()),
                "global_threshold_feasibility": {
                    "minimum_true_open_probability": min_open, "maximum_true_closed_probability": max_closed,
                    "perfect_single_threshold_exists": max_closed < min_open if min_open is not None and max_closed is not None else None,
                    "note": "打开要求threshold<=最小打开概率；关闭要求threshold>最大关闭概率。仅检验可行性，不选阈值、不改推理。"},
                "wrong_windows_fraction_of_total_gripper_bce": float(element_bce[erroneous_rows].sum()) / total_bce if total_bce else 0.,
                "wrong_windows": [{"index": chosen[i], "sample": dataset.samples[chosen[i]],
                    "instruction": items[i][0], "sensed_close": (1. - float(items[i][2])) / 2.,
                    "target_open": truth_open[i].tolist(), "predicted_open": predicted_state[i].tolist(),
                    "teacher_pose_predicted_open": teacher_state[i].tolist(),
                    "open_probabilities": logits[i].sigmoid().tolist(),
                    "teacher_pose_open_probabilities": teacher_logits[i].sigmoid().tolist()} for i in erroneous_rows],
                "exact_input_conflicts": [],
            }
            from audit_bcz import inspect_command_neighborhood
            for record in boundary_diagnostic["wrong_windows"] if name == "train" else []:
                sample = record["sample"]
                source_example = dataset._load_tfrecord_example(sample["file_path"], sample["record_index"])
                record["raw_source_neighborhood"] = inspect_command_neighborhood(source_example.features.feature, int(sample["start_index"]))
            # 完整10目标也须检查精确输入冲突，不能只沿用旧第一目标检查。
            import hashlib
            signatures = {}
            for i, item in enumerate(items):
                signature = hashlib.sha256(item[0].encode("utf-8") + item[1].numpy().tobytes() + item[2].numpy().tobytes()).hexdigest()
                if signature in signatures:
                    other = signatures[signature]
                    left, right = items[other][3], item[3]
                    equivalent = (torch.allclose(left[:, :3], right[:, :3], atol=1e-5, rtol=0)
                        and bool(((left[:, 3:7] * right[:, 3:7]).sum(-1).abs() >= 1 - 1e-5).all())
                        and torch.equal(left[:, 7], right[:, 7]))
                    if not equivalent:
                        boundary_diagnostic["exact_input_conflicts"].append([chosen[other], chosen[i]])
                else:
                    signatures[signature] = i
        records = {}
        for variant, batches in results.items():
            prediction = torch.cat(batches)
            valid_xyz = masks[..., :3].all(-1)
            valid_q = masks[..., 3:7].all(-1)
            valid_grip = masks[..., 7]
            predicted_open = prediction[..., 7] >= 0
            def mean_or_none(values, mask):
                return float(values[mask].float().mean()) if mask.any() else None
            def xyz_length(values):
                return values[..., :3].norm(dim=-1) * POSITION_SCALE_METERS * 100
            records[variant] = {
                "position_error_cm": mean_or_none(xyz_length(prediction - targets), valid_xyz),
                "predicted_motion_cm": mean_or_none(xyz_length(prediction), valid_xyz),
                "target_motion_cm": mean_or_none(xyz_length(targets), valid_xyz),
                "rotation_error_deg": mean_or_none(2 * torch.rad2deg(torch.acos((prediction[..., 3:7] * targets[..., 3:7]).sum(-1).abs().clamp(0, 1))), valid_q),
                "gripper_accuracy": mean_or_none(predicted_open == truth_open, valid_grip),
                "change_accuracy": mean_or_none(predicted_open == truth_open, valid_grip & change),
                "persistence_accuracy": mean_or_none(current_open == truth_open, valid_grip),
                "position_delta_from_original_cm": mean_or_none(xyz_length(prediction - original), valid_xyz),
                "gripper_delta_from_original_rate": mean_or_none(predicted_open != (original[..., 7] >= 0), valid_grip),
                "context_delta_relative": float(((torch.cat(contexts[variant]) - original_context).norm(dim=-1) / original_context.norm(dim=-1).clamp_min(1e-8)).mean()),
            }
        report["partitions"][name] = {"indices": chosen, "samples": [dataset.samples[i] for i in chosen],
             "instructions": [item[0] for item in items],
             "donor_indices": chosen[1:] + chosen[:1],
             "sampled_windows": len(chosen),
             "sampled_episodes": len({dataset.group_key(i) for i in chosen}),
             "changed_instruction_fraction": sum(items[i][0] != items[(i + 1) % len(items)][0] for i in range(len(items))) / len(items),
             "variants": records, "targets": targets.tolist(),
             "gripper_boundary_diagnostic": boundary_diagnostic,
             "gripper_errors": [{"index": chosen[i], "waypoint_zero_based": k,
                 "target_open": bool(truth_open[i, k]), "predicted_open": bool(original[i, k, 7] >= 0)}
                 for i, k in ((original[..., 7] >= 0) != truth_open).logical_and(masks[..., 7]).nonzero().tolist()],
             "predictions": {key: torch.cat(value).tolist() for key, value in results.items()}}
        print(f"[条件诊断] {name}：{records}", flush=True)
    path = Path(args.condition_diagnostic)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[条件诊断] 已保存：{path.resolve()}")


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"找不到 checkpoint：{checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "data_config" not in checkpoint:
        raise ValueError(
            "checkpoint 没有 data_config；请使用独立夹爪头版本的 train.py 重新训练。"
        )
    data_config = dict(checkpoint["data_config"])
    if checkpoint.get("experiment_kind") == "training_fit_diagnostic":
        print("[诊断模型] 本权重仅用于少量训练窗口拟合检查，不能当作正式泛化或操控结果")
    if checkpoint.get("experiment_kind") == "offline_command_pilot":
        print("[离线试验] 有独立演示划分，但目标时序与执行接口未确认，不代表闭环操控成绩")
    if checkpoint.get("experiment_kind") == "bridge_single_task_offline_diagnostic":
        print("[Bridge诊断] 单任务独立演示回归；连续测量输入，不代表闭环；persistence仅为测量阈值代理，不是保持先前命令")
    if args.split in {"train", "seen-train"}:
        print("[训练窗口评估] 此范围包含已见样本，不是独立验证")
    print("[辅助指标] teacher_pose_gripper 使用真实位姿，仅用于排查夹爪头；实际推理请看 gripper")
    checkpoint_representation = checkpoint.get("config", {}).get("action", {}).get(
        "representation"
    )
    if checkpoint_representation != ACTION_REPRESENTATION:
        raise ValueError(
            "checkpoint 的动作语义与当前加载器不兼容："
            f"checkpoint={checkpoint_representation!r}，"
            f"current={ACTION_REPRESENTATION!r}。"
            "旧模型使用了未统一的夹爪正负号，只能保留作诊断记录，不能继续比较。"
        )

    if args.dataset_dir:
        dataset = UnifiedRobotDataset(
            data_dir=args.dataset_dir,
            chunk_size=int(data_config["chunk_size"]),
            stride=int(data_config["stride"]),
            sources=("auto",),
            min_trajectory_steps=int(data_config.get("min_trajectory_steps", 10)),
            bcz_target=data_config.get("bcz_target", "reached"),
            bcz_current_gripper=data_config.get("bcz_current_gripper", "binary"),
            bridge_gripper_policy=data_config.get("bridge_gripper_policy", "threshold_v1"),
            bridge_current_gripper=data_config.get("bridge_current_gripper", "binary"),
            rt1_gripper_policy=data_config.get("rt1_gripper_policy", "legacy_threshold_v1"),
            bcz_reached_gripper_policy=data_config.get("bcz_reached_gripper_policy", "future_measured_v1"),
        )
        evaluation_indices = list(range(len(dataset)))
        evaluation_name = f"external:{args.dataset_dir}"
    else:
        dataset = UnifiedRobotDataset(
            data_dir=data_config["dataset_dir"],
            chunk_size=int(data_config["chunk_size"]),
            stride=int(data_config["stride"]),
            sources=tuple(data_config["sources"]),
            max_samples=data_config.get("max_samples"),
            max_samples_per_schema=data_config.get("max_samples_per_schema"),
            max_tfrecord_episodes=data_config.get("max_tfrecord_episodes"),
            max_tfrecord_episodes_per_schema=data_config.get(
                "max_tfrecord_episodes_per_schema"
            ),
            # 老 checkpoint 创建于严格10步过滤之前；用最小合法值2重建原索引。
            min_trajectory_steps=int(data_config.get("min_trajectory_steps", 2)),
            exclude_path_parts=tuple(data_config.get("exclude_path_parts", ())),
            exclude_schemas=tuple(data_config.get("exclude_schemas", ())),
            tfrecord_splits=tuple(data_config.get("tfrecord_splits", ())),
            bcz_target=data_config.get("bcz_target", "reached"),
            bcz_current_gripper=data_config.get("bcz_current_gripper", "binary"),
            bridge_gripper_policy=data_config.get("bridge_gripper_policy", "threshold_v1"),
            bridge_current_gripper=data_config.get("bridge_current_gripper", "binary"),
            rt1_gripper_policy=data_config.get("rt1_gripper_policy", "legacy_threshold_v1"),
            bcz_reached_gripper_policy=data_config.get("bcz_reached_gripper_policy", "future_measured_v1"),
            bridge_episode_selection=data_config.get("bridge_episode_selection"),
        )
        if (checkpoint.get("dataset_identity") is not None
                and checkpoint["dataset_identity"] != dataset_split_identity(dataset)):
            raise ValueError("checkpoint的数据身份改变，不能复用旧分区评估")
        expected_size = int(checkpoint.get("dataset_size", len(dataset)))
        if len(dataset) != expected_size:
            raise ValueError(
                f"数据规模改变：checkpoint={expected_size}，current={len(dataset)}"
            )
        if args.split == "seen-train":
            evaluation_indices = list(dict.fromkeys(checkpoint.get("training_access_order", [])))
            if not evaluation_indices or not set(evaluation_indices).issubset(checkpoint["split_indices"]["train"]):
                raise ValueError("权重没有有效实际训练访问记录，不能把随机训练池当已见拟合评估")
        else:
            evaluation_indices = list(checkpoint["split_indices"][args.split])
        evaluation_name = f"checkpoint-{args.split}"

    if args.condition_diagnostic:
        if dataset.bcz_target == "native_commands":
            evaluation_name = "condition-diagnostic:train (原生序列仅已见拟合)"
            evaluation_indices = checkpoint["split_indices"]["train"]
        else:
            evaluation_name = "condition-diagnostic:train+validation (不评估test)"
            evaluation_indices = checkpoint["split_indices"]["train"] + checkpoint["split_indices"]["validation"]
    original_window_count = len(evaluation_indices)
    evaluation_indices = select_episode_windows(dataset, evaluation_indices, args.max_windows_per_episode, args.seed)
    if args.max_windows_per_episode:
        print(f"[分层评估] 从{original_window_count}个窗口中按演示身份抽取{len(evaluation_indices)}个；"
              f"每条最多{args.max_windows_per_episode}个，不按标签筛选")
    print("=" * 68)
    print(f"[离线评估] 设备：{device}")
    print(f"[离线评估] 范围：{evaluation_name}，样本={len(evaluation_indices)}")
    print(f"[监督目标] BC-Z={dataset.bcz_target}；窗口长度={dataset.chunk_size}；不可混同实际状态与控制目标")
    print(f"[监督版本] Bridge夹爪={dataset.bridge_gripper_policy}（从checkpoint恢复，旧权重缺省threshold_v1）")
    print(
        f"[离线评估] 动作指标最多抽样 {args.diagnostic_samples * (1 if dataset.bcz_target == 'native_commands' else 2) if args.condition_diagnostic else args.max_batches * min(args.samples_per_batch, args.batch_size)} "
        "个窗口；窗口不等于独立演示；括号中的 n 是数据集窗口规模。"
    )
    print(f"[离线评估] 抽样方式：全评估范围固定随机抽样（seed={args.seed}）")
    model = RobotAdapterModel(checkpoint["config"], cache_dir=args.cache_dir).to(device)
    if args.transition_decode is not None:
        model.gripper_transition_decode = args.transition_decode
    if args.switch_threshold is not None:
        model.gripper_switch_threshold = args.switch_threshold
    print(
        f"[离线评估] 解码器：{model.decoder_type}；图文融合：{model.fusion_type}；"
        f"视觉读出：{getattr(model, 'adapter_pooling', '不适用')}；"
        f"夹爪监督：{model.gripper_target_mode}；"
        f"切换解码：{model.gripper_transition_decode}；"
        f"阈值={model.gripper_switch_threshold:.2f}"
    )
    incompatible = model.load_state_dict(checkpoint["trainable_state_dict"], strict=False)
    if incompatible.unexpected_keys:
        raise ValueError(f"checkpoint 含未知参数：{incompatible.unexpected_keys}")
    model.eval()
    tokenizer = CLIPTokenizer.from_pretrained(
        str(checkpoint["config"]["model"]["name"]), cache_dir=args.cache_dir
    )
    criterion = nn.MSELoss()
    if args.condition_diagnostic:
        if args.dataset_dir or args.diagnostic_samples < 2:
            raise ValueError("条件诊断使用 checkpoint 原始划分，样本数至少为2")
        diagnose_conditions(model, tokenizer, dataset, checkpoint["split_indices"], device, args)
        return
    if args.trajectory_dir:
        export_trajectory_comparisons(model, tokenizer, dataset, evaluation_indices, device, args)

    groups: dict[str, list[int]] = defaultdict(list)
    for index in evaluation_indices:
        groups[str(dataset.samples[index]["source"])].append(index)
    groups = {"overall": evaluation_indices, **groups}
    report = {"checkpoint": str(checkpoint_path.resolve()), "scope": evaluation_name,
              "bcz_target": dataset.bcz_target, "chunk_size": dataset.chunk_size,
              "experiment_kind": checkpoint.get("experiment_kind", "held_out_training"),
              "original_evaluation_windows": original_window_count,
              "max_windows_per_episode": args.max_windows_per_episode,
              "seed": args.seed, "groups": {}}
    for group_number, (label, indices) in enumerate(groups.items()):
        loader = make_loader(
            dataset,
            indices,
            args.batch_size,
            seed=args.seed + group_number,
        )
        coverage = metric_coverage(dataset, loader.dataset.indices, args.batch_size,
                                   args.max_batches, args.samples_per_batch)
        print(f"[评估覆盖] {label}：实际抽样{coverage['sampled_windows']}个窗口，"
              f"来自{coverage['sampled_episodes']}/{coverage['available_episodes']}条独立演示")
        loss = evaluate_loss(
            model,
            loader,
            tokenizer,
            device,
            criterion,
            max_batches=args.max_batches,
        )
        metrics = evaluate_action_metrics(
            model,
            loader,
            tokenizer,
            device,
            max_batches=args.max_batches,
            samples_per_batch=args.samples_per_batch,
        )
        if int(metrics.get("metric_trajectories", 0)) != coverage["sampled_windows"]:
            raise RuntimeError("评估覆盖记录与实际指标窗口数不一致")
        print_metrics(f"{label} (n={len(indices)})", loss, metrics)
        report["groups"][label] = {"dataset_windows": len(indices), "loss": loss, **metrics, **coverage}
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[离线评估] 已保存报告：{output.resolve()}")


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=str(project_root / "results" / "relative_v5" / "best.pt"),
    )
    parser.add_argument(
        "--dataset-dir",
        help="可选外部数据根目录；不填时使用 checkpoint 保存的测试划分。",
    )
    parser.add_argument(
        "--split",
        choices=("train", "seen-train", "validation", "test"),
        default="test",
        help="未指定外部目录时评估哪个划分；seen-train仅评估有访问记录的实际已见窗口；默认test。",
    )
    parser.add_argument("--cache-dir", default=r"D:\ntu_related\dissertation\hf_cache")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--max-batches",
        type=int,
        default=5,
        help="每个总体/schema 最多评估多少个 batch；CPU 下完整扩散较慢。",
    )
    parser.add_argument(
        "--samples-per-batch",
        type=int,
        default=4,
        help="每个 batch 实际进行完整扩散抽样的轨迹数。",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", help="可选：保存指标与评估范围，避免只留下终端日志。")
    parser.add_argument("--max-windows-per-episode", type=int, default=0, help="每条演示最多固定随机抽取多少窗口；0保持原窗口分布，1用于等演示权重低成本评估。")
    parser.add_argument("--condition-diagnostic", help="仅运行训练/验证分区的匹配输入条件诊断，并保存JSON；不训练。")
    parser.add_argument("--diagnostic-samples", type=int, default=16, help="条件诊断每个分区抽取的窗口数。")
    parser.add_argument("--trajectory-dir", help="可选：保存逐窗口真实/预测轨迹图及预测数值。")
    parser.add_argument("--trajectory-samples", type=int, default=8)
    parser.add_argument("--trajectory-draws", type=int, default=5, help="扩散每窗口独立采样次数；回归自动为1。")
    parser.add_argument(
        "--transition-decode",
        choices=("cumulative", "single_switch"),
        help="可覆盖 checkpoint 的 transition 解码方式，用于不重训的受控比较。",
    )
    parser.add_argument(
        "--switch-threshold",
        type=float,
        help="可覆盖 checkpoint 的切换 logit 阈值。",
    )
    args = parser.parse_args()
    if min(args.batch_size, args.max_batches, args.samples_per_batch, args.diagnostic_samples) <= 0:
        parser.error("batch-size、max-batches、samples-per-batch、diagnostic-samples必须大于0")
    if args.max_windows_per_episode < 0 or (args.max_windows_per_episode and args.condition_diagnostic):
        parser.error("max-windows-per-episode必须非负，且不能与condition-diagnostic同时使用")
    if args.split == "seen-train" and (args.dataset_dir or args.condition_diagnostic):
        parser.error("seen-train仅用于权重记录的实际已见窗口，不能与外部目录或条件诊断混用")
    return args


if __name__ == "__main__":
    evaluate(parse_args())

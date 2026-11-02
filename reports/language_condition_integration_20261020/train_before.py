
"""Train the 8-D CLIP-adapter diffusion policy with grouped data splits."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from transformers import CLIPTokenizer

from dataset import (
    ACTION_DIM,
    ACTION_REPRESENTATION,
    POSITION_SCALE_METERS,
    UnifiedRobotDataset,
)
from models import RobotAdapterModel


# Windows 重定向日志时可能仍使用系统编码；统一成 UTF-8，避免中文提示乱码。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_by_trajectory(
    dataset: UnifiedRobotDataset,
    seed: int,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
) -> Tuple[List[int], List[int], List[int]]:
    """按完整轨迹划分，避免同一轨迹的重叠片段泄漏到不同集合。"""
    groups: Dict[str, List[int]] = defaultdict(list)
    for index in range(len(dataset)):
        groups[dataset.group_key(index)].append(index)
    group_keys = list(groups)
    if len(group_keys) < 3:
        raise RuntimeError("At least three trajectories are required for train/val/test splits")
    random.Random(seed).shuffle(group_keys)
    train_groups = max(1, int(len(group_keys) * train_fraction))
    validation_groups = max(1, int(len(group_keys) * validation_fraction))
    if train_groups + validation_groups >= len(group_keys):
        train_groups = len(group_keys) - 2
        validation_groups = 1
    partitions = (
        group_keys[:train_groups],
        group_keys[train_groups : train_groups + validation_groups],
        group_keys[train_groups + validation_groups :],
    )
    return tuple(
        [index for key in keys for index in groups[key]] for keys in partitions
    )  # type: ignore[return-value]


def dataset_split_identity(dataset, *, target_override=None, chunk_override=None, gripper_override=None):
    """标识扫描窗口及源文件元信息；同样本数不等于同一份训练数据。"""
    paths = sorted({str(Path(s["file_path"]).resolve()) for s in dataset.samples})
    files = []
    for path in paths:
        stat = Path(path).stat()
        files.append([path, stat.st_size, stat.st_mtime_ns])
    payload = {"samples": dataset.samples, "files": files,
               "chunk_size": dataset.chunk_size if chunk_override is None else chunk_override, "stride": dataset.stride,
               "action_representation": ACTION_REPRESENTATION}
    # 默认 reached 的指纹保持与已有划分清单兼容；控制命令是另一监督契约。
    target = dataset.bcz_target if target_override is None else target_override
    gripper = getattr(dataset, "bcz_current_gripper", "binary") if gripper_override is None else gripper_override
    if target != "reached":
        payload["bcz_target"] = target
    if gripper != "binary":
        payload["bcz_current_gripper"] = gripper
    bridge_policy = getattr(dataset, "bridge_gripper_policy", "threshold_v1")
    if bridge_policy != "threshold_v1" and any(s["source"] == "tfrecord_bridge_state_action" for s in dataset.samples):
        payload["bridge_gripper_policy"] = bridge_policy
    if getattr(dataset, "bridge_current_gripper", "binary") != "binary":
        payload["bridge_current_gripper"] = dataset.bridge_current_gripper
        payload["bridge_measurement_contract"] = "measured_affine_unbounded_v1"
    if getattr(dataset, "bridge_episode_selection", None):
        payload["bridge_episode_selection"] = dataset.bridge_episode_selection
    if getattr(dataset, "bridge_window_horizon", None) is not None:
        payload["bridge_window_horizon"] = dataset.bridge_window_horizon
    if (getattr(dataset, "rt1_gripper_policy", "legacy_threshold_v1") != "legacy_threshold_v1"
            and any(s["source"] == "tfrecord_rt1_pose" for s in dataset.samples)):
        payload["rt1_gripper_policy"] = dataset.rt1_gripper_policy
    if (getattr(dataset, "bcz_reached_gripper_policy", "future_measured_v1") != "future_measured_v1"
            and any(s["source"] == "tfrecord_bc_z_pose" for s in dataset.samples)):
        payload["bcz_reached_gripper_policy"] = dataset.bcz_reached_gripper_policy
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def bridge_plan_selection(path, dataset_dir=None):
    plan = json.loads(Path(path).read_text(encoding="utf-8"))
    if plan.get("purpose") == "bridge_multitask_offline_pilot_manifest_v1":
        if (plan.get("status") != "verified_train_development_only"
                or plan.get("reserved_test_targets_read") is not False
                or plan.get("partitions", {}).get("test") != []):
            raise ValueError("多任务pilot清单必须只包含已核对的训练/开发数据")
        root = Path(path).resolve().parent / plan["data_directory"]
        if dataset_dir is not None and Path(dataset_dir).resolve() != root.resolve():
            raise ValueError("数据目录与多任务清单不同")
        for name, expected in plan["shard_sha256"].items():
            digest = hashlib.sha256()
            with (root / name).open("rb") as stream:
                for block in iter(lambda: stream.read(4 * 1024**2), b""):
                    digest.update(block)
            if digest.hexdigest() != expected:
                raise ValueError("多任务pilot数据指纹不一致")
        selected = []
        for part in ("train", "validation"):
            if not plan["partitions"][part]:
                raise ValueError("训练/开发分区不能为空")
            selected.extend({**r, "partition": part} for r in plan["partitions"][part])
        if any(r["shard"] not in plan["shard_sha256"] for r in selected):
            raise ValueError("清单缺少数据指纹")
        keys = [(r["shard"], r["record_index"]) for r in selected]
        reserved = {(r["shard"], r["record_index"]) for r in plan["reserved_test_identity_only"]}
        if len(keys) != len(set(keys)) or set(keys) & reserved:
            raise ValueError("多任务演示身份重复或测试泄漏")
        return selected
    if (plan.get("purpose") != "bridge_single_task_metadata_plan_not_training_manifest"
            or plan.get("version") != "0.0.1" or not plan.get("selected_instruction")):
        raise ValueError("不是已准备的Bridge单任务计划，或没有足够演示")
    selection = []
    for partition in ("train", "validation", "test"):
        rows = plan["partitions"][partition]
        if not rows:
            raise ValueError("Bridge计划分区不能为空")
        selection.extend({**row, "partition": partition, "instruction": plan["selected_instruction"]} for row in rows)
    return selection


def bridge_plan_splits(dataset):
    mapping = {(r["shard"], r["record_index"]): r["partition"] for r in dataset.bridge_episode_selection}
    splits = {name: [] for name in ("train", "validation", "test")}
    for i, sample in enumerate(dataset.samples):
        splits[mapping[Path(sample["file_path"]).name, sample["record_index"]]].append(i)
    validate_split_indices(dataset, splits)
    return splits


def select_bridge_fit_windows(dataset, train_indices, count):
    """按元数据均匀选择训练演示，每演示取中间窗口；不看目标或误差。"""
    groups = defaultdict(list)
    for index in train_indices:
        groups[dataset.group_key(index)].append(index)
    values = list(groups.values())
    if count < 2 or count > len(values):
        raise ValueError("Bridge拟合窗口数须在2与训练演示数量之间，每演示最多1窗口")
    chosen = np.linspace(0, len(values) - 1, count, dtype=int)
    return [values[g][len(values[g]) // 2] for g in chosen]


def select_bridge_gripper_candidates(dataset, train_indices, report_path):
    """只接受已核对的训练候选；类别选样是拟合诊断，绝不当自然分布评估。"""
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    checks = report.get("candidate_input_checks", {})
    rows = checks.get("checks", [])
    if (report.get("purpose") != "bridge_training_gripper_coverage_candidates_not_training_manifest"
            or report.get("dataset_identity") != dataset_split_identity(dataset)
            or report.get("validation_test_targets_inspected") is not False
            or checks.get("exact_input_conflicts") != 0
            or checks.get("checked_candidates") != len(rows) or not rows):
        raise ValueError("Bridge夹爪候选报告未经核对或数据身份不同")
    indices = [row["index"] for row in rows]
    if (any(type(i) is not int or i not in set(train_indices) for i in indices)
            or len(indices) != len(set(indices))):
        raise ValueError("夹爪候选重复或不在固定训练分区")
    required = {"hold_open", "hold_closed", "open_to_closed", "closed_to_open"}
    if {row["category"] for row in rows} != required:
        raise ValueError("夹爪候选须覆盖保持开/关及双向单次切换")
    for row in rows:
        if (dataset.samples[row["index"]] != row["sample"]
                or row.get("loader_image_matches_raw_preprocessing") is not True
                or row.get("loader_gripper_matches_command_labels") is not True):
            raise ValueError("夹爪候选原始窗口或核对状态不一致")
    return indices, rows


def validate_split_indices(dataset, splits):
    """对照必须覆盖全部窗口，且同一完整演示不能跨分区。"""
    seen_indices, seen_groups = set(), set()
    for name in ("train", "validation", "test"):
        indices = splits[name]
        pilot_without_test = (name == "test" and not indices
            and bool(getattr(dataset, "bridge_episode_selection", None))
            and {r["partition"] for r in dataset.bridge_episode_selection} == {"train", "validation"})
        if (not indices and not pilot_without_test) or any(type(i) is not int or not 0 <= i < len(dataset) for i in indices):
            raise ValueError(f"数据划分 {name} 为空或包含非法索引")
        index_set = set(indices)
        groups = {dataset.group_key(i) for i in indices}
        if len(index_set) != len(indices) or seen_indices & index_set or seen_groups & groups:
            raise ValueError("数据划分有重复窗口或完整轨迹泄漏")
        seen_indices.update(index_set)
        seen_groups.update(groups)
    if seen_indices != set(range(len(dataset))):
        raise ValueError("数据划分没有覆盖全部扫描窗口")


def select_overfit_samples(dataset, indices, count, trajectory_count):
    """选取有完整监督的真实训练窗口，兼顾运动和夹爪变化；不用于泛化评分。"""
    groups = defaultdict(list)
    for index in indices:
        sample = dataset.samples[index]
        length = sample.get("trajectory_steps", 0)
        if int(sample.get("start_index", 0)) + dataset.chunk_size >= length:
            continue  # 排除尾部补齐窗口，防止只记住静止占位动作。
        groups[dataset.group_key(index)].append(index)
    scored_groups = []
    decoded = {}
    # 诊断只需少量轨迹，不为挑选窗口解码整个训练集。
    for key, candidates in list(groups.items())[:max(trajectory_count * 4, 8)]:
        scored = []
        # 每条轨迹最多解码 12 个分散窗口，控制检查耗时。
        positions = np.linspace(0, len(candidates) - 1, min(12, len(candidates)), dtype=int)
        for position in positions:
            index = candidates[int(position)]
            item = dataset[index]
            if not bool((item[4] > .5).all()):
                continue
            actions = item[3]
            changing = bool(((actions[:, 7] >= 0) != (item[2].item() >= 0)).any())
            motion = float(actions[:, :3].norm(dim=-1).max())
            decoded[index] = item
            scored.append((int(changing), motion, index))
        if scored:
            scored.sort(reverse=True)
            scored_groups.append((scored[0][:2], key, scored))
    scored_groups.sort(reverse=True)
    chosen_groups = scored_groups[:trajectory_count]
    selected = []
    # 轮流从每条轨迹选窗口，避免全部来自一条演示。
    for rank in range(12):
        for _, _, candidates in chosen_groups:
            if rank < len(candidates) and len(selected) < count:
                selected.append(candidates[rank][2])
    if len(selected) < count:
        raise ValueError(f"完整监督窗口不足：需要{count}，找到{len(selected)}；增加扫描范围或轨迹数")
    print(f"[拟合诊断] 使用 {len(chosen_groups)} 条训练轨迹中的 {count} 个完整窗口")
    return selected, [decoded[index] for index in selected]


def select_balanced_overfit_samples(dataset, indices, count, trajectory_count, output_dir):
    """仅诊断：训练分区定量抽样，检查精确相同输入是否带有冲突目标。"""
    if dataset.bcz_target != "first_command" or count % 2:
        raise ValueError("平衡拟合仅支持BC-Z第一原生目标，样本数必须为偶数")
    groups = defaultdict(list)
    for i in indices:
        groups[dataset.group_key(i)].append(i)
    decoded, records, identities = {}, [], defaultdict(list)
    for candidates in list(groups.values())[:max(trajectory_count * 4, 8)]:
        positions = np.linspace(0, len(candidates) - 1, min(12, len(candidates)), dtype=int)
        for position in positions:
            i = candidates[int(position)]
            item = dataset[i]
            if not bool((item[4] > .5).all()) or not torch.isfinite(item[3]).all():
                continue
            instruction, image, current, target, _ = item
            signature = hashlib.sha256(instruction.encode("utf-8") + image.numpy().tobytes() + current.numpy().tobytes()).hexdigest()
            identities[signature].append(i)
            decoded[i] = item
            records.append({"index": i, "episode": dataset.group_key(i),
                            "target_open": bool(target[0, 7] >= 0),
                            "current_open": bool(current.item() >= 0),
                            "motion_cm_project_convention": float(target[0, :3].norm() * POSITION_SCALE_METERS * 100)})
    conflicting = []
    for values in identities.values():
        def equivalent_target(i):
            left, right = decoded[values[0]][3], decoded[i][3]
            # q与-q等价；不能把四元数符号差别当作控制目标冲突。
            return (torch.allclose(left[:, :3], right[:, :3], atol=1e-5, rtol=0)
                    and bool(((left[:, 3:7] * right[:, 3:7]).sum(-1).abs() >= 1 - 1e-5).all())
                    and torch.equal(left[:, 7], right[:, 7]))
        if len(values) > 1 and any(not equivalent_target(i) for i in values[1:]):
            conflicting.append(values)
    audit = {"purpose": "training_only_balanced_fit_not_population_statistics",
             "candidate_windows": len(records), "candidate_episodes": len({r['episode'] for r in records}),
             "target_counts": dict(Counter('open' if r['target_open'] else 'closed' for r in records)),
             "exact_input_conflicting_indices": conflicting,
             "limitations": ["精确相同输入检查不检测视觉相近但状态不同的情况。", "当前夹爪0.5阈值与位置米单位仍是项目约定。"],
             "candidate_records": records}
    (output_dir / "balanced_overfit_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    if conflicting:
        raise ValueError("检查范围存在精确相同输入但不同目标，先查看balanced_overfit_audit.json，不静默删除")
    selected = []
    for state in (True, False):
        eligible = [r['index'] for r in records if r['target_open'] == state]
        if len(eligible) < count // 2:
            raise ValueError(f"平衡诊断的{'打开' if state else '关闭'}样本不足：{len(eligible)}；见审计报告")
        # 候选按演示再按时间排序；循环跨演示取样，避免全部取同一条轨迹。
        per_episode = defaultdict(list)
        for i in eligible:
            per_episode[dataset.group_key(i)].append(i)
        rank = 0
        chosen = []
        while len(chosen) < count // 2:
            for values in per_episode.values():
                if rank < len(values) and len(chosen) < count // 2:
                    chosen.append(values[rank])
            rank += 1
        selected.extend(chosen)
    audit["selected_indices"] = selected
    audit["selected_records"] = [r for r in records if r['index'] in set(selected)]
    audit["selected_target_counts"] = {"open": count // 2, "closed": count // 2}
    (output_dir / "balanced_overfit_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[输入目标检查] 候选={len(records)}，精确输入冲突=0；已选打开/关闭各{count // 2}个，只用于拟合诊断")
    return selected, [decoded[i] for i in selected]


def collate_batch(
    batch: Sequence[
        Tuple[str, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ],
) -> Tuple[List[str], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    instructions, images, current_grippers, action_chunks, supervision_masks = zip(
        *batch
    )
    return (
        list(instructions),
        torch.stack(images),
        torch.stack(current_grippers),
        torch.stack(action_chunks),
        torch.stack(supervision_masks),
    )


class IndexedTrainingSubset(Dataset):
    """只为离线对照附带原始索引，记录真正执行优化的窗口而非预取窗口。"""
    def __init__(self, dataset, indices):
        self.dataset, self.indices = dataset, list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        original = self.indices[index]
        return self.dataset[original], original


def collate_indexed_training(batch):
    return (*collate_batch([item for item, _ in batch]), [index for _, index in batch])


def build_balanced_sampler(
    dataset: UnifiedRobotDataset,
    train_indices: Sequence[int],
    generator: torch.Generator,
) -> WeightedRandomSampler:
    """让每种 schema 在期望上获得相同的抽样概率。"""
    schemas = [str(dataset.samples[index]["source"]) for index in train_indices]
    counts = Counter(schemas)
    weights = torch.as_tensor(
        [1.0 / counts[schema] for schema in schemas], dtype=torch.double
    )
    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(train_indices),
        replacement=True,
        generator=generator,
    )


def tokenise(
    tokenizer: CLIPTokenizer, instructions: List[str], device: torch.device
) -> Dict[str, torch.Tensor]:
    encoded = tokenizer(
        instructions,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    )
    return {key: value.to(device) for key, value in encoded.items()}


def policy_loss(
    model: RobotAdapterModel,
    model_output: Tuple[torch.Tensor, ...],
    actions: torch.Tensor,
    criterion: nn.Module,
    current_grippers: torch.Tensor | None = None,
    supervision_masks: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Combine pose diffusion loss with binary gripper classification loss."""
    predicted_noise, true_noise = model_output[:2]
    if model.decoder_type == "regression":
        # 位移用相同的0.1m尺度；旋转用符号不变的四元数夹角代理，q/-q等价。
        mask = supervision_masks if supervision_masks is not None else torch.ones_like(actions)
        xyz_mask = mask[..., :3]
        xyz_loss = ((predicted_noise[..., :3] - true_noise[..., :3]).square() * xyz_mask).sum() / xyz_mask.sum().clamp_min(1)
        predicted_q = nn.functional.normalize(predicted_noise[..., 3:7], dim=-1)
        target_q = nn.functional.normalize(true_noise[..., 3:7], dim=-1)
        rotation_mask = (mask[..., 3:7] > .5).all(dim=-1)
        rotation_error = 1 - (predicted_q * target_q).sum(dim=-1).abs().clamp(max=1)
        rotation_loss = rotation_error[rotation_mask].mean() if rotation_mask.any() else xyz_loss * 0
        pose_loss = xyz_loss + getattr(model, "regression_rotation_weight", 1.0) * rotation_loss
        valid_gripper = mask[..., 7] > .5
    elif supervision_masks is None:
        pose_loss = criterion(predicted_noise, true_noise)
        valid_gripper = None
    else:
        pose_mask = supervision_masks[..., :7].to(predicted_noise.dtype)
        squared_error = (predicted_noise - true_noise).square()
        pose_loss = (squared_error * pose_mask).sum() / pose_mask.sum().clamp_min(1.0)
        valid_gripper = supervision_masks[..., 7] > 0.5
    gripper_loss = torch.zeros((), device=pose_loss.device)
    if len(model_output) == 3:
        gripper_logits = model_output[2]
        state_targets = actions[..., 7] > 0.0
        if model.gripper_target_mode == "transition":
            if current_grippers is None:
                raise ValueError("Transition targets require current gripper state")
            current_open = current_grippers.reshape(-1, 1) >= 0.0
            previous_states = torch.cat(
                [current_open, state_targets[:, :-1]], dim=1
            )
            gripper_targets = (state_targets != previous_states).float()
        else:
            gripper_targets = state_targets.float()
        per_step_bce = nn.functional.binary_cross_entropy_with_logits(
            gripper_logits, gripper_targets, reduction="none"
        )
        step_weights = torch.ones_like(per_step_bce)
        if model.gripper_target_mode == "transition":
            if model.gripper_change_weight > 1.0:
                step_weights = torch.where(
                    gripper_targets.bool(),
                    torch.full_like(step_weights, model.gripper_change_weight),
                    step_weights,
                )
        elif current_grippers is not None and model.gripper_change_weight > 1.0:
            current_open = (current_grippers >= 0.0).expand_as(gripper_targets)
            change_mask = gripper_targets.bool() != current_open
            step_weights = torch.where(
                change_mask,
                torch.full_like(step_weights, model.gripper_change_weight),
                step_weights,
            )
        if model.balanced_gripper_loss:
            # 消融选项：让切换/不切换对损失各贡献 50%。它有利于召回稀有
            # 切换，但概率不再反映真实事件频率，跨机器人时容易产生误切换。
            class_losses = []
            for class_value in (0.0, 1.0):
                class_mask = gripper_targets == class_value
                if valid_gripper is not None:
                    class_mask = class_mask & valid_gripper
                if class_mask.any():
                    selected_weights = step_weights[class_mask]
                    class_losses.append(
                        (per_step_bce[class_mask] * selected_weights).sum()
                        / selected_weights.sum()
                    )
            if class_losses:
                gripper_loss = torch.stack(class_losses).mean()
        else:
            # 默认使用真实时间步频率，使 sigmoid(logit) 保持概率含义。
            # 这对 LIBERO 这类大部分窗口无需切换的域尤其重要。
            valid = (
                valid_gripper
                if valid_gripper is not None
                else torch.ones_like(gripper_targets, dtype=torch.bool)
            )
            if valid.any():
                selected_weights = step_weights[valid]
                gripper_loss = (
                    per_step_bce[valid] * selected_weights
                ).sum() / selected_weights.sum().clamp_min(1.0)
    total_loss = pose_loss + model.gripper_loss_weight * gripper_loss
    return total_loss, pose_loss, gripper_loss


@torch.no_grad()
def evaluate_loss(
    model: RobotAdapterModel,
    data_loader: DataLoader,
    tokenizer: CLIPTokenizer,
    device: torch.device,
    criterion: nn.Module,
    max_batches: int | None = None,
) -> float:
    model.eval()
    losses: List[float] = []
    for batch_index, (
        instructions,
        images,
        current_grippers,
        actions,
        supervision_masks,
    ) in enumerate(data_loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        text = tokenise(tokenizer, instructions, device)
        device_actions = actions.to(device)
        model_output = model(
            images.to(device),
            text["input_ids"],
            attention_mask=text.get("attention_mask"),
            current_gripper=current_grippers.to(device),
            actions=device_actions,
        )
        total_loss, _, _ = policy_loss(
            model,
            model_output,
            device_actions,
            criterion,
            current_grippers=current_grippers.to(device),
            supervision_masks=supervision_masks.to(device),
        )
        losses.append(float(total_loss.item()))
    return float(np.mean(losses)) if losses else float("inf")


@torch.no_grad()
def evaluate_action_metrics(
    model: RobotAdapterModel,
    data_loader: DataLoader,
    tokenizer: CLIPTokenizer,
    device: torch.device,
    max_batches: int,
    samples_per_batch: int,
) -> Dict[str, float]:
    """用完整反向扩散抽样，计算比 noise loss 更直观的动作误差。"""
    model.eval()
    position_errors: List[torch.Tensor] = []
    rotation_errors: List[torch.Tensor] = []
    gripper_correct: List[torch.Tensor] = []
    gripper_persistence_correct: List[torch.Tensor] = []
    persistence_open_values: List[torch.Tensor] = []
    predicted_open_values: List[torch.Tensor] = []
    target_open_values: List[torch.Tensor] = []
    absolute_errors: List[torch.Tensor] = []
    persistence_position_errors: List[torch.Tensor] = []
    identity_rotation_errors: List[torch.Tensor] = []
    teacher_pose_gripper_correct: List[torch.Tensor] = []
    native_pair_counts = Counter()
    metric_trajectories = 0

    for batch_index, (
        instructions,
        images,
        current_grippers,
        targets,
        supervision_masks,
    ) in enumerate(data_loader):
        if batch_index >= max_batches:
            break
        sample_count = min(samples_per_batch, len(instructions))
        instructions = instructions[:sample_count]
        images = images[:sample_count].to(device)
        current_grippers = current_grippers[:sample_count].to(device)
        targets = targets[:sample_count].to(device)
        supervision_masks = supervision_masks[:sample_count].to(device)
        text = tokenise(tokenizer, instructions, device)
        predictions = model(
            images,
            text["input_ids"],
            attention_mask=text.get("attention_mask"),
            current_gripper=current_grippers,
        )
        metric_trajectories += sample_count

        position_mask = supervision_masks[..., :3] > 0.5
        valid_position = position_mask.any(dim=-1)
        position_error = (
            (predictions[..., :3] - targets[..., :3]).square()
            * position_mask
        ).sum(dim=-1).sqrt()
        if valid_position.any():
            position_errors.append(position_error[valid_position])
            persistence_position_errors.append(
                (targets[..., :3].square() * position_mask).sum(dim=-1).sqrt()[valid_position]
            )
        predicted_quaternion = torch.nn.functional.normalize(
            predictions[..., 3:7], dim=-1
        )
        target_quaternion = torch.nn.functional.normalize(targets[..., 3:7], dim=-1)
        quaternion_similarity = (
            predicted_quaternion * target_quaternion
        ).sum(dim=-1).abs().clamp(0.0, 1.0)
        valid_rotation = (supervision_masks[..., 3:7] > 0.5).all(dim=-1)
        if valid_rotation.any():
            rotation_errors.append(
                (2.0 * torch.acos(quaternion_similarity) * (180.0 / np.pi))[
                    valid_rotation
                ]
            )
            identity_rotation_errors.append(
                (2 * torch.acos(target_quaternion[..., 3].abs().clamp(0, 1)) * (180 / np.pi))[valid_rotation]
            )
        predicted_open = predictions[..., 7] >= 0.0
        target_open = targets[..., 7] >= 0.0
        valid_gripper = supervision_masks[..., 7] > 0.5
        if targets.shape[1] > 1:
            valid_pairs = valid_gripper[:, :-1] & valid_gripper[:, 1:]
            pair_correct = (predicted_open[:, :-1] == target_open[:, :-1]) & (predicted_open[:, 1:] == target_open[:, 1:])
            for name, pair_mask in (("open_to_closed", target_open[:, :-1] & ~target_open[:, 1:]),
                                    ("closed_to_open", ~target_open[:, :-1] & target_open[:, 1:])):
                pair_mask = pair_mask & valid_pairs
                native_pair_counts[name] += int(pair_mask.sum())
                native_pair_counts[name + "_correct"] += int((pair_correct & pair_mask).sum())
        if valid_gripper.any():
            gripper_correct.append(
                (predicted_open[valid_gripper] == target_open[valid_gripper]).float()
            )
            if model.separate_gripper_head and model.gripper_target_mode == "state":
                # 辅助诊断：真实位姿替代预测位姿，区分夹爪头错误与位姿误差传播。
                # 不将此数值当作实际推理准确率。
                context = model.get_context_vector(images, text["input_ids"], text.get("attention_mask"))
                teacher_open = model.predict_gripper_logits(context, targets[..., :7], current_grippers) >= 0
                teacher_pose_gripper_correct.append((teacher_open[valid_gripper] == target_open[valid_gripper]).float())
        persistence_open = (current_grippers >= 0.0).expand_as(target_open)
        if valid_gripper.any():
            gripper_persistence_correct.append(
                (persistence_open[valid_gripper] == target_open[valid_gripper]).float()
            )
            persistence_open_values.append(persistence_open[valid_gripper].cpu())
            predicted_open_values.append(predicted_open[valid_gripper].cpu())
            target_open_values.append(target_open[valid_gripper].cpu())
        # q 与 -q 表示同一旋转；计算 MAE 前先把预测四元数翻到目标同一半球。
        aligned_predictions = predictions.clone()
        quaternion_dot = (predicted_quaternion * target_quaternion).sum(
            dim=-1, keepdim=True
        )
        aligned_predictions[..., 3:7] = predicted_quaternion * torch.where(
            quaternion_dot < 0.0, -1.0, 1.0
        )
        absolute_error = (aligned_predictions - targets).abs()
        absolute_errors.append(
            absolute_error[supervision_masks > 0.5].reshape(-1).cpu()
        )

    if not position_errors:
        return {}
    if target_open_values:
        predicted_open = torch.cat(predicted_open_values).reshape(-1)
        target_open = torch.cat(target_open_values).reshape(-1)
        persistence_open = torch.cat(persistence_open_values).reshape(-1)
        true_open_rate = float(target_open.float().mean().item())
        predicted_open_rate = float(predicted_open.float().mean().item())
        majority_baseline = max(true_open_rate, 1.0 - true_open_rate)
        open_mask = target_open
        closed_mask = ~target_open
        open_recall = (
            float(predicted_open[open_mask].float().mean().item())
            if open_mask.any()
            else float("nan")
        )
        closed_recall = (
            float((~predicted_open[closed_mask]).float().mean().item())
            if closed_mask.any()
            else float("nan")
        )
        balanced_accuracy = (
            0.5 * (open_recall + closed_recall)
            if open_mask.any() and closed_mask.any()
            else float("nan")
        )
        change_mask = target_open != persistence_open
        hold_mask = ~change_mask
        change_accuracy = (
            float((predicted_open[change_mask] == target_open[change_mask]).float().mean())
            if change_mask.any()
            else float("nan")
        )
        hold_accuracy = (
            float((predicted_open[hold_mask] == target_open[hold_mask]).float().mean())
            if hold_mask.any()
            else float("nan")
        )
        gripper_accuracy = float(torch.cat(gripper_correct).mean().item())
        persistence_accuracy = float(
            torch.cat(gripper_persistence_correct).mean().item()
        )
        change_rate = float(change_mask.float().mean().item())
        gripper_step_count = target_open.numel()
    else:
        true_open_rate = predicted_open_rate = majority_baseline = float("nan")
        balanced_accuracy = change_accuracy = hold_accuracy = float("nan")
        gripper_accuracy = persistence_accuracy = change_rate = float("nan")
        gripper_step_count = 0
    return {
        "position_error_cm": float(
            torch.cat(position_errors).mean().item() * POSITION_SCALE_METERS * 100.0
        ),
        "zero_motion_position_error_cm": float(torch.cat(persistence_position_errors).mean() * POSITION_SCALE_METERS * 100),
        "identity_rotation_error_deg": float(torch.cat(identity_rotation_errors).mean()) if identity_rotation_errors else float("nan"),
        "rotation_error_deg": (
            float(torch.cat(rotation_errors).mean().item())
            if rotation_errors
            else float("nan")
        ),
        "gripper_accuracy": gripper_accuracy,
        "gripper_teacher_pose_accuracy": float(torch.cat(teacher_pose_gripper_correct).mean()) if teacher_pose_gripper_correct else float("nan"),
        "gripper_balanced_accuracy": balanced_accuracy,
        "gripper_majority_baseline": majority_baseline,
        "gripper_persistence_baseline": persistence_accuracy,
        "gripper_change_rate": change_rate,
        "gripper_change_accuracy": change_accuracy,
        "gripper_hold_accuracy": hold_accuracy,
        "gripper_true_open_rate": true_open_rate,
        "gripper_predicted_open_rate": predicted_open_rate,
        "metric_trajectories": float(metric_trajectories),
        "metric_action_steps": float(gripper_step_count),
        "action_mae": float(torch.cat(absolute_errors).mean().item()),
        **{f"gripper_target_pair_{name}_count": float(native_pair_counts[name]) for name in ("open_to_closed", "closed_to_open")},
        **{f"gripper_target_pair_{name}_accuracy": (native_pair_counts[name + "_correct"] / native_pair_counts[name]
               if native_pair_counts[name] else float("nan")) for name in ("open_to_closed", "closed_to_open")},
    }


def build_config(args: argparse.Namespace) -> Dict[str, Dict[str, object]]:
    return {
        "action": {
            "representation": ACTION_REPRESENTATION,
            "position_scale_meters": POSITION_SCALE_METERS,
            "max_normalized_position": 3.0,
            "dimensions": "local_dx_dy_dz+dqx_dqy_dqz_dqw+gripper",
        },
        "model": {
            "name": args.model_name,
            "fusion_type": args.fusion_type,
            "adapter_pooling": args.adapter_pooling,
            "decoder_type": args.decoder_type,
            "diffusion_prediction_type": args.diffusion_prediction_type,
            "num_adapter_layers": args.adapter_layers,
            "attention_dim": args.attention_dim,
            "num_attention_heads": args.attention_heads,
            "decoder_hidden_dim": args.decoder_hidden_dim,
            "chunk_size": args.chunk_size,
            "action_dim": ACTION_DIM,
            "num_diffusion_steps": args.diffusion_steps,
            "beta_schedule": args.beta_schedule,
            "clip_denoised": args.beta_schedule == "squaredcos_cap_v2",
            "dropout": args.dropout,
            "separate_gripper_head": True,
            "trajectory_conditioned_gripper": True,
            "gripper_head_type": args.gripper_head_type,
            "condition_on_current_gripper": True,
            "current_gripper_encoding": ("bridge_measured_affine_unbounded_v1"
                if getattr(args, "bridge_current_gripper", "binary") == "continuous" else args.bcz_current_gripper),
            "gripper_loss_weight": args.gripper_loss_weight,
            "gripper_change_weight": args.gripper_change_weight,
            "gripper_target_mode": args.gripper_target_mode,
            "gripper_transition_decode": args.gripper_transition_decode,
            "gripper_switch_threshold": args.gripper_switch_threshold,
            "balanced_gripper_loss": args.balanced_gripper_loss,
            **({"regression_rotation_weight": args.regression_rotation_weight}
               if args.regression_rotation_weight != 1.0 else {}),
        }
    }


GRIPPER_PARAMETER_PREFIXES = ("gripper_head.", "gripper_context_projection.",
                            "gripper_pose_projection.", "gripper_time_embedding.", "current_gripper_projection.")


def frozen_policy_digest(model):
    """仅夹爪对照：冻结的融合/位姿参数逐字节校验；CLIP始终无梯度。"""
    digest = hashlib.sha256()
    for name, value in trainable_state_dict(model).items():
        if not name.startswith(GRIPPER_PARAMETER_PREFIXES):
            digest.update(name.encode())
            digest.update(value.contiguous().numpy().tobytes())
    return digest.hexdigest()


def trainable_state_dict(model: RobotAdapterModel) -> Dict[str, torch.Tensor]:
    """只保存可训练模块；冻结的 CLIP 下次按模型名称重新加载。"""
    prefixes = (
        "adapter.",
        "simple_fusion.",
        "diffusion_decoder.",
        "regression_head.",
        "gripper_head.",
        "gripper_context_projection.",
        "gripper_pose_projection.",
        "gripper_time_embedding.",
        "current_gripper_projection.",
    )
    return {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if name.startswith(prefixes)
    }


def load_gripper_fit_weights(model, source_state, *, reset_readout=False, seed=42):
    """结构对照只允许遗漏读出头；其余模块必须键和形状完全一致。"""
    expected = trainable_state_dict(model)
    keep = lambda key: not (reset_readout and key.startswith("gripper_head."))
    source = {key: value for key, value in source_state.items() if keep(key)}
    required = {key: value for key, value in expected.items() if keep(key)}
    if set(source) != set(required) or any(source[key].shape != required[key].shape for key in required):
        raise ValueError("源权重除显式重置的夹爪读出头外，必须完整且形状相同")
    model.load_state_dict(source, strict=False)
    if reset_readout:
        parameter = next(model.gripper_head.parameters())
        devices = [parameter.device.index] if parameter.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed + 30000)
            for module in model.gripper_head.modules():
                if hasattr(module, "reset_parameters"):
                    module.reset_parameters()


def build_learning_rate_scheduler(optimizer, schedule, epochs):
    """默认保持余弦衰减；constant用于固定样本的单因素拟合对照。"""
    if schedule == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    if schedule == "constant":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)
    raise ValueError(f"未知学习率策略：{schedule}")


def validate_native_experiment(args):
    """区分已见窗口诊断与完整训练池离线基线，二者都不是闭环模型。"""
    if not np.isfinite(args.regression_rotation_weight) or args.regression_rotation_weight <= 0:
        raise ValueError("旋转损失权重须为有限正数")
    if args.regression_rotation_weight != 1.0 and (not args.pool_fit_from or args.decoder_type != "regression"):
        raise ValueError("非默认旋转权重仅用于显式源完整池已见探针回归拟合对照")
    if args.pool_fit_from and (args.bcz_target != "native_commands" or not args.overfit_samples
            or args.offline_command_experiment or args.overfit_manifest or args.resume or args.init_from
            or args.gripper_only_fit_from or args.reset_gripper_readout or args.balanced_overfit_targets
            or args.balanced_sampling or args.prepare_only):
        raise ValueError("完整池探针继续拟合只允许原生固定窗口诊断，禁止混入迁移、重选或完整训练")
    if args.native_adapter_comparison and (args.bcz_target != "native_commands" or not args.offline_command_experiment):
        raise ValueError("Adapter对照仅允许显式原生10目标完整池离线实验")
    if args.bcz_target != "native_commands":
        return
    if (args.chunk_size != 10 or args.decoder_type != "regression"
            or args.bcz_current_gripper != "continuous" or args.gripper_target_mode != "state"
            or not args.split_manifest or args.gripper_change_weight != 1.0 or args.balanced_gripper_loss):
        raise ValueError("原生序列必须使用固定分区、10目标、连续观测、回归/state及原频率损失")
    if args.offline_command_experiment:
        if (args.overfit_samples or args.overfit_manifest or args.balanced_overfit_targets
                or args.gripper_only_fit_from or args.reset_gripper_readout or args.init_from or args.resume
                or args.balanced_sampling):
            raise ValueError("完整原生序列实验须从头训练，不能混入拟合清单、迁移或重采样")
        if args.native_adapter_comparison:
            if args.fusion_type != "cross_attention" or args.adapter_pooling != "cls_patch_mean" or args.adapter_layers < 1:
                raise ValueError("Adapter对照必须显式使用cross_attention及cls_patch_mean视觉读出")
        elif args.fusion_type != "simple_concat":
            raise ValueError("原生完整池默认只允许simple_concat；Adapter须显式开启对照标志")
    elif not args.overfit_samples or not (args.overfit_manifest or args.pool_fit_from) or args.prepare_only or args.balanced_overfit_targets:
        raise ValueError("native_commands须显式选择固定窗口拟合或offline-command-experiment完整池基线")


def train(args: argparse.Namespace) -> None:
    bridge_selection = bridge_plan_selection(args.bridge_task_plan, args.dataset_dir) if args.bridge_task_plan else None
    multitask_pilot = bool(args.bridge_task_plan and json.loads(Path(args.bridge_task_plan).read_text(encoding="utf-8")).get("purpose") == "bridge_multitask_offline_pilot_manifest_v1")
    if multitask_pilot:
        import tensorflow as tf
        tf.config.set_visible_devices([], "GPU")  # TensorFlow reads records; PyTorch owns the GPU.
    if args.bridge_validation_scope == "all_windows" and (not bridge_selection or args.overfit_samples):
        raise ValueError("全窗口验证只用于固定Bridge完整训练分区，不能混用小样本拟合")
    if args.bridge_current_gripper == "continuous" and args.gripper_target_mode != "state":
        raise ValueError("连续Bridge测量不是先前命令，必须使用绝对命令state监督，不能构造transition")
    if args.bridge_current_gripper == "continuous" and (args.gripper_change_weight != 1.0 or args.balanced_gripper_loss):
        raise ValueError("Bridge连续测量基线不按测量阈值构造切换权重，也不启用类别平衡")
    if bridge_selection and (args.bcz_target != "reached" or args.bridge_current_gripper != "continuous"
            or args.bridge_gripper_policy != "reverse_scan_valid_steps_v2" or args.decoder_type != ("diffusion" if multitask_pilot else "regression")
            or args.resume or args.init_from or args.pool_fit_from
            or args.gripper_only_fit_from or args.offline_command_experiment):
        raise ValueError("Bridge单任务计划仅用于独立从头回归基线：连续测量、官方扫描、state命令；不混用迁移或旧诊断")
    if multitask_pilot and (args.overfit_samples or args.chunk_size != 16
            or args.adapter_layers != 8 or args.attention_dim != 512
            or args.adapter_pooling != "cls_patch_mean" or args.balanced_sampling
            or args.fusion_type != "cross_attention" or args.diffusion_prediction_type != "sample"):
        raise ValueError("多任务pilot须使用已预检结构和完整固定分区，不挑选或重采样")
    if bridge_selection and args.overfit_samples and (
            (not args.bridge_gripper_fit_report and args.overfit_trajectories != args.overfit_samples) or not args.split_manifest
            or args.balanced_overfit_targets or args.balanced_sampling):
        raise ValueError("Bridge小样本诊断必须复用固定分区，每训练演示1窗口，不重采样/按标签挑选")
    if args.bridge_gripper_fit_report and (not bridge_selection or not args.overfit_samples
            or args.gripper_only_fit_from or args.pool_fit_from):
        raise ValueError("夹爪候选入口仅支持固定Bridge训练分区小样本联合拟合，不混用旧权重迁移")
    native_fit = args.bcz_target == "native_commands"
    native_pilot = native_fit and args.offline_command_experiment
    validate_native_experiment(args)
    if args.audit_first_update and (not args.overfit_samples or args.dropout != 0 or args.decoder_type != "regression"):
        raise ValueError("首步更新审计仅用于无dropout的回归固定窗口拟合，不用于普通训练")
    if args.reset_gripper_readout and not args.gripper_only_fit_from:
        raise ValueError("重置读出头只允许显式仅夹爪固定窗口结构对照")
    if args.gripper_only_fit_from and (not native_fit or args.resume or args.init_from):
        raise ValueError("仅夹爪对照必须是显式native_commands固定窗口诊断，不能混用resume/init-from")
    if args.bcz_current_gripper != "binary" and not native_fit:
        raise ValueError("连续夹爪观测仅用于native_commands诊断")
    if args.balanced_overfit_targets and (not args.overfit_samples or args.overfit_manifest):
        raise ValueError("balanced-overfit-targets仅用于重新选择过拟合诊断样本，不能混用旧overfit-manifest")
    if args.bcz_target == "first_command" and (args.chunk_size != 1 or args.decoder_type != "regression"):
        raise ValueError("第一BC-Z控制目标诊断目前仅支持 --chunk-size 1 --decoder-type regression")
    if args.offline_command_experiment:
        if args.bcz_target not in {"first_command", "native_commands"} or args.overfit_samples or args.init_from:
            raise ValueError("离线原生目标实验只允许first_command/native_commands，不能混用过拟合或迁移权重")
        if not args.split_manifest:
            raise ValueError("离线原生目标实验必须显式指定已冻结的 --split-manifest")
    if args.bcz_target == "first_command" and not (args.overfit_samples or args.prepare_only or args.offline_command_experiment):
        raise ValueError("first_command需显式选择拟合诊断、prepare-only或offline-command-experiment，不是默认主训练")
    if args.overfit_samples:
        if args.resume or args.init_from:
            raise ValueError("小样本诊断从头训练，暂不支持 resume/init-from")
        if args.overfit_samples < 2 or args.overfit_trajectories < 1:
            raise ValueError("--overfit-samples 至少2；--overfit-trajectories 至少1")
        if args.workers:
            raise ValueError("小样本缓存诊断请使用 --workers 0")
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    if (native_fit or bridge_selection) and output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("原生序列诊断需使用全新输出目录，不能覆盖旧结果")
    output_dir.mkdir(parents=True, exist_ok=True)
    sources = tuple(part.strip() for part in args.sources.split(",") if part.strip())
    exclude_path_parts = tuple(
        part.strip() for part in args.exclude_path_parts.split(",") if part.strip()
    )
    exclude_schemas = tuple(
        part.strip() for part in args.exclude_schemas.split(",") if part.strip()
    )
    tfrecord_splits = tuple(
        part.strip() for part in args.tfrecord_splits.split(",") if part.strip()
    )
    print("=" * 68)
    print("[训练 1/6] 初始化实验")
    print(f"[训练] 计算设备：{device}")
    print(f"[训练] 数据根目录：{args.dataset_dir}")
    print(f"[训练] 输出目录：{output_dir}")
    print(f"[训练] 随机种子：{args.seed}")
    print(f"[监督目标] BC-Z={args.bcz_target}；每个目标8维；窗口长度={args.chunk_size}")
    print(f"[监督版本] Bridge夹爪={args.bridge_gripper_policy}；不会无提示更换旧权重的标签策略")
    if bridge_selection:
        print("[Bridge单任务] 输入为2*原始测量-1（不裁剪、不是概率），输出为绝对夹爪命令；仅离线到达位姿诊断")
        print("[基线限制] persistence仅是当前测量的0.5阈值代理，不是保持先前命令；不要据此宣称切换事件正确")
    if native_fit:
        print(f"[实验性质] 原生10×8目标{'完整训练池离线基线' if native_pilot else '已见窗口拟合'}；连续观测=1-2*sensed_close；无物理时间标定，禁止闭环使用")
        print("[指标限制] persistence/change仍使用0.5候选阈值，只是描述性基线，不是物理开闭事件")
    if args.offline_command_experiment:
        print("[实验性质] 独立演示离线原生控制目标试验；物理时序/执行接口未确认，不用于闭环成功率")
    print(
        f"[训练] 动作表示：{ACTION_REPRESENTATION}，"
        f"位移缩放={POSITION_SCALE_METERS:.3f}m"
    )
    print(
        f"[训练] 图文融合：{args.fusion_type}；"
        f"夹爪监督：{args.gripper_target_mode}；"
        f"类别平衡：{'50/50' if args.balanced_gripper_loss else '真实频率'}"
    )
    print(f"[训练] 留出路径：{exclude_path_parts or '无'}")
    print(f"[训练] TFRecord split：{tfrecord_splits or '全部'}")
    print(f"[训练] 排除的不完整监督 schema：{exclude_schemas or '无'}")

    print("[训练 2/6] 自动发现数据并建立样本索引")
    dataset = UnifiedRobotDataset(
        data_dir=args.dataset_dir,
        chunk_size=args.chunk_size,
        stride=args.stride,
        sources=sources,
        max_samples=args.max_samples,
        max_samples_per_schema=args.max_samples_per_schema,
        max_tfrecord_episodes=args.max_tfrecord_episodes,
        max_tfrecord_episodes_per_schema=args.max_tfrecord_episodes_per_schema,
        min_trajectory_steps=args.min_trajectory_steps,
        exclude_path_parts=exclude_path_parts,
        exclude_schemas=exclude_schemas,
        tfrecord_splits=tfrecord_splits,
        bcz_target=args.bcz_target,
        bcz_current_gripper=args.bcz_current_gripper,
        bridge_gripper_policy=args.bridge_gripper_policy,
        bridge_current_gripper=args.bridge_current_gripper,
        bridge_episode_selection=bridge_selection,
        bridge_window_horizon=args.bridge_window_horizon,
        rt1_gripper_policy=args.rt1_gripper_policy,
        bcz_reached_gripper_policy=args.bcz_reached_gripper_policy,
    )
    if (any(s["source"] == "tfrecord_rt1_pose" for s in dataset.samples)
            and args.rt1_gripper_policy == "legacy_threshold_v1"
            and not args.prepare_only and not args.resume):
        raise ValueError("新Fractal训练必须选择 --rt1-gripper-policy relative_scan_v2，不能沿用旧阈值标签")
    pool_fit_checkpoint = None
    if args.pool_fit_from:
        pool_fit_checkpoint = torch.load(args.pool_fit_from, map_location="cpu", weights_only=False)
        source_manifest = json.loads((Path(args.pool_fit_from).parent / "split_manifest.json").read_text(encoding="utf-8"))
        comparable_source = json.loads(json.dumps(pool_fit_checkpoint["config"]))
        comparable_target = build_config(args)
        comparable_source["model"].pop("regression_rotation_weight", None)
        comparable_target["model"].pop("regression_rotation_weight", None)
        if (pool_fit_checkpoint.get("experiment_kind") != "offline_command_pilot"
                or comparable_source != comparable_target
                or pool_fit_checkpoint["dataset_size"] != len(dataset)
                or source_manifest["dataset_identity"] != dataset_split_identity(dataset)):
            raise ValueError("源完整池结构、目标或数据身份不同，禁止继续拟合")
    gripper_fit_checkpoint = None
    if args.gripper_only_fit_from:
        gripper_fit_checkpoint = torch.load(args.gripper_only_fit_from, map_location="cpu", weights_only=False)
        source_config = json.loads(json.dumps(gripper_fit_checkpoint.get("config", {})))
        target_config = build_config(args)
        source_config.setdefault("model", {}).setdefault("gripper_head_type", "legacy")
        if args.reset_gripper_readout:
            # 唯一允许的结构差异是读出头，绝不放宽输入/位姿/损失配置检查。
            source_config["model"]["gripper_head_type"] = args.gripper_head_type
        if (gripper_fit_checkpoint.get("experiment_kind") != "training_fit_diagnostic"
                or gripper_fit_checkpoint.get("data_config", {}).get("bcz_target") != "native_commands"
                or gripper_fit_checkpoint.get("data_config", {}).get("bcz_current_gripper") != "continuous"
                or source_config != target_config):
            raise ValueError("夹爪对照源模型必须是完全相同结构/输入语义的原生序列拟合权重")
    if args.resume and args.init_from:
        raise ValueError("--resume 与 --init-from 不能同时使用")
    resume_checkpoint = None
    init_checkpoint = None
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"找不到要恢复的 checkpoint：{resume_path}")
        print(f"[恢复] 正在读取：{resume_path}")
        resume_checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        required_keys = {"epoch", "trainable_state_dict", "optimizer_state_dict", "config"}
        missing_keys = required_keys - set(resume_checkpoint)
        if missing_keys:
            raise ValueError(f"checkpoint 缺少字段：{sorted(missing_keys)}")
    if args.init_from:
        init_path = Path(args.init_from)
        if not init_path.exists():
            raise FileNotFoundError(f"找不到初始化 checkpoint：{init_path}")
        print(f"[迁移] 读取已有模型权重并重置优化器：{init_path}")
        init_checkpoint = torch.load(init_path, map_location="cpu", weights_only=False)
        required_keys = {"trainable_state_dict", "config"}
        missing_keys = required_keys - set(init_checkpoint)
        if missing_keys:
            raise ValueError(f"初始化 checkpoint 缺少字段：{sorted(missing_keys)}")

    split_checkpoint = resume_checkpoint or init_checkpoint
    if split_checkpoint and split_checkpoint.get("data_config", {}).get("bridge_gripper_policy", "threshold_v1") != args.bridge_gripper_policy:
        raise ValueError("Bridge夹爪标签版本不同，不能作为同一实验恢复或初始化")
    if split_checkpoint and split_checkpoint.get("data_config", {}).get("bridge_current_gripper", "binary") != args.bridge_current_gripper:
        raise ValueError("Bridge输入测量编码不同，不能恢复或初始化为同一实验")
    if split_checkpoint and split_checkpoint.get("data_config", {}).get("bcz_target", "reached") != args.bcz_target:
        raise ValueError("BC-Z实际状态与控制目标监督不同，不能恢复或迁移为同一实验")
    use_saved_splits = split_checkpoint is not None and "split_indices" in split_checkpoint
    if use_saved_splits:
        saved_splits = split_checkpoint["split_indices"]
        saved_indices = (
            list(saved_splits["train"])
            + list(saved_splits["validation"])
            + list(saved_splits["test"])
        )
        saved_dataset_size = int(
            split_checkpoint.get("dataset_size", len(set(saved_indices)))
        )
        if saved_dataset_size != len(dataset):
            if resume_checkpoint is not None:
                raise ValueError(
                    "checkpoint 对应的数据规模与当前扫描结果不同："
                    f"checkpoint={saved_dataset_size}，current={len(dataset)}。"
                    "--resume 必须使用完全相同的数据。"
                )
            print(
                "[数据划分] --init-from 检测到数据规模变化："
                f"checkpoint={saved_dataset_size}，current={len(dataset)}；"
                "仅迁移权重并重新按轨迹划分数据"
            )
            use_saved_splits = False
        else:
            train_indices = list(saved_splits["train"])
            validation_indices = list(saved_splits["validation"])
            test_indices = list(saved_splits["test"])
            print("[数据划分] 沿用 checkpoint 中原有的训练/验证/测试划分")
    if not use_saved_splits:
        train_indices, validation_indices, test_indices = split_by_trajectory(
            dataset, args.seed
        )
    if bridge_selection:
        planned = bridge_plan_splits(dataset)
        if multitask_pilot:
            expected = json.loads(Path(args.bridge_task_plan).read_text(encoding="utf-8"))["expected_windows"]
            if {name: len(indices) for name, indices in planned.items()} != expected:
                raise ValueError("多任务pilot窗口覆盖与已验收清单不同")
        train_indices, validation_indices, test_indices = (planned[name] for name in ("train", "validation", "test"))
    if args.split_manifest:
        if args.init_from or (args.overfit_samples and not args.balanced_overfit_targets and not native_fit and not bridge_selection):
            raise ValueError("--split-manifest 用于完整数据对照，不能与迁移/拟合诊断混用")
        saved_manifest = json.loads(Path(args.split_manifest).read_text(encoding="utf-8"))
        identity = dataset_split_identity(dataset)
        if native_fit:
            if saved_manifest.get("bcz_target") != "first_command" or saved_manifest.get("chunk_size") != 1:
                raise ValueError("原生序列诊断须复用已冻结的第一目标训练池划分")
            identity = dataset_split_identity(dataset, target_override="first_command", chunk_override=1, gripper_override="binary")
        if saved_manifest["dataset_identity"] != identity:
            raise ValueError("数据窗口、源文件元信息或动作窗口设置改变，不能复用对照划分")
        saved_splits = saved_manifest["split_indices"]
        validate_split_indices(dataset, saved_splits)
        if bridge_selection and saved_splits != bridge_plan_splits(dataset):
            raise ValueError("对照清单与固定Bridge计划分区不一致")
        if args.resume and saved_splits != resume_checkpoint.get("split_indices"):
            raise ValueError("恢复权重的数据划分与对照清单不同，不能继续")
        train_indices, validation_indices, test_indices = (
            saved_splits[name] for name in ("train", "validation", "test")
        )
        print(f"[数据划分] 精确复用对照清单：{args.split_manifest}")
        if native_fit:
            print("[隔离] 只复用源文件/观测窗口/演示分区；标签长度和输入编码已变，不作为同任务公平对照")
    if not args.overfit_samples:
        splits = {"train": train_indices, "validation": validation_indices, "test": test_indices}
        validate_split_indices(dataset, splits)
        split_manifest = {"purpose": "grouped_training_comparison", "seed": args.seed,
                          "bcz_target": dataset.bcz_target, "chunk_size": dataset.chunk_size,
                          "dataset_size": len(dataset), "dataset_identity": dataset_split_identity(dataset),
                          "split_indices": splits,
                          **({"bridge_current_gripper": args.bridge_current_gripper,
                              "bridge_gripper_policy": args.bridge_gripper_policy,
                              "bridge_episode_selection": bridge_selection,
                              "bridge_task_plan": str(Path(args.bridge_task_plan).resolve())} if bridge_selection else {}),
                          "trajectory_counts": {name: len({dataset.group_key(i) for i in values})
                                                for name, values in splits.items()}}
        (output_dir / "split_manifest.json").write_text(
            json.dumps(split_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[数据划分] 独立演示数量：{split_manifest['trajectory_counts']}；清单已保存")
    if args.prepare_only and not (bridge_selection and args.overfit_samples):
        if args.overfit_samples:
            raise ValueError("--prepare-only 只用于完整数据划分，不用于小样本拟合诊断")
        print(f"[准备完成] 仅扫描与验证数据划分，未加载 CLIP 或训练。清单：{output_dir / 'split_manifest.json'}")
        return
    overfit_items = None
    if pool_fit_checkpoint is not None:
        current_splits = {"train": train_indices, "validation": validation_indices, "test": test_indices}
        if current_splits != pool_fit_checkpoint["split_indices"]:
            raise ValueError("源模型训练/验证/测试分区不同")
    if args.overfit_samples:
        if bridge_selection:
            candidate_rows = None
            if args.bridge_gripper_fit_report:
                selected, candidate_rows = select_bridge_gripper_candidates(dataset, train_indices, args.bridge_gripper_fit_report)
                if (len(selected) != args.overfit_samples
                        or len({dataset.group_key(i) for i in selected}) != args.overfit_trajectories):
                    raise ValueError("夹爪候选窗口/独立训练演示数量与显式参数不同")
            else:
                selected = select_bridge_fit_windows(dataset, train_indices, args.overfit_samples)
            bridge_fit_contract = {"purpose": "bridge_fixed_train_fit_only",
                "dataset_identity": dataset_split_identity(dataset), "selected_indices": selected,
                "selected_windows": [dataset.samples[i] for i in selected],
                "selection_rule": "uniform_training_episode_order_then_middle_window",
                "held_out_validation_indices": validation_indices, "held_out_test_indices": test_indices,
                "source_split_manifest": str(Path(args.split_manifest).resolve()),
                "input_encoding": "bridge_measured_affine_unbounded_v1",
                "gripper_policy": args.bridge_gripper_policy,
                "independent_training_episodes": len({dataset.group_key(i) for i in selected})}
            if candidate_rows is not None:
                bridge_fit_contract.update({
                    "selection_rule": "audited_training_gripper_category_coverage_v1",
                    "candidate_report": str(Path(args.bridge_gripper_fit_report).resolve()),
                    "candidate_report_sha256": hashlib.sha256(Path(args.bridge_gripper_fit_report).read_bytes()).hexdigest(),
                    "selected_categories": [r["category"] for r in candidate_rows],
                    "limitations": "按训练标签选择的联合位姿/夹爪拟合；不是仅夹爪冻结实验或泛化评估"})
            if args.overfit_manifest:
                saved_fit = json.loads(Path(args.overfit_manifest).read_text(encoding="utf-8"))
                if saved_fit != bridge_fit_contract:
                    raise ValueError("Bridge拟合清单与固定选择/数据身份/分区不同，禁止换样")
            overfit_items = [dataset[i] for i in selected]
            if candidate_rows is not None:
                from audit_bcz import bridge_gripper_window_category
                for row, item in zip(candidate_rows, overfit_items):
                    commands = (item[3][:, 7].numpy() + 1) / 2
                    signature = hashlib.sha256(item[0].encode("utf-8") + item[1].numpy().tobytes()
                                               + item[2].numpy().tobytes()).hexdigest()
                    if (commands.tolist() != row["commands_open_positive"]
                            or bridge_gripper_window_category(commands) != row["category"]
                            or signature != row["input_signature"]):
                        raise ValueError("夹爪候选图文/测量/目标与已核对报告不同")
                print("[夹爪覆盖拟合] 按训练标签选择，联合优化位姿与夹爪；不是仅夹爪冻结或泛化实验")
            if (any(not bool(torch.isfinite(t).all()) for item in overfit_items for t in item[1:])
                    or any(not bool((item[4] > .5).all()) for item in overfit_items)):
                raise ValueError("Bridge拟合输入/目标非有限或监督不完整")
            seen_pose_inputs = {}
            for text, image, current, targets, mask in overfit_items:
                signature = hashlib.sha256(text.encode("utf-8") + image.numpy().tobytes()).hexdigest()
                if signature in seen_pose_inputs and not torch.equal(seen_pose_inputs[signature], targets[:, :7]):
                    raise ValueError("相同图文输入对应不同位姿目标，小样本不能以精确拟合验收；不自动删除冲突")
                seen_pose_inputs[signature] = targets[:, :7]
            train_indices = selected
            (output_dir / "overfit_manifest.json").write_text(
                json.dumps(bridge_fit_contract, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[Bridge拟合] 固定{len(selected)}窗口/{bridge_fit_contract['independent_training_episodes']}训练演示，已缓存图像/标签；验证与测试目标未读取")
            if args.prepare_only:
                print(f"[准备完成] 仅准备Bridge拟合清单，未加载CLIP或训练：{output_dir / 'overfit_manifest.json'}")
                return
        elif args.overfit_manifest or pool_fit_checkpoint is not None:
            if pool_fit_checkpoint is not None:
                selected = list(pool_fit_checkpoint["fit_probe_indices"])
                if len(set(selected)) != len(selected) or not set(selected).issubset(pool_fit_checkpoint["training_access_order"]):
                    raise ValueError("源固定探针重复或并非实际已训练窗口")
                manifest = {"selected_indices": selected, "selected_windows": [dataset.samples[i] for i in selected],
                            "purpose": "saved_training_probe_continuation_not_generalization", "source_checkpoint": args.pool_fit_from}
            else:
                manifest = json.loads(Path(args.overfit_manifest).read_text(encoding="utf-8"))
            selected = manifest["selected_indices"]
            if len(selected) != args.overfit_samples or len(selected) != len(manifest["selected_windows"]):
                raise ValueError("对照窗口数量与 --overfit-samples 不一致")
            if not set(selected).issubset(train_indices):
                raise ValueError("对照窗口不在当前训练分区，不能复用")
            if any(dataset.samples[i] != sample for i, sample in zip(selected, manifest["selected_windows"])):
                raise ValueError("扫描索引与对照清单不同，不能保证同一输入")
            train_indices = selected
            overfit_items = [dataset[i] for i in selected]
            if native_fit:
                targets = torch.stack([item[3] for item in overfit_items])
                closed = targets[..., 7] < 0
                diagnostic = {"purpose": "native_sequence_training_fit_only",
                    "shape": list(targets.shape), "dataset_identity": dataset_split_identity(dataset),
                    "source_split_manifest": args.split_manifest, "source_overfit_manifest": args.overfit_manifest,
                    "input_encoding": "continuous_1_minus_2_sensed_close", "timing": "unknown_not_executable",
                    "target_open_count": int((~closed).sum()), "target_closed_count": int(closed.sum()),
                    "within_chunk_open_to_closed": int((~closed[:, :-1] & closed[:, 1:]).sum()),
                    "within_chunk_closed_to_open": int((closed[:, :-1] & ~closed[:, 1:]).sum()),
                    "independent_episodes": len({dataset.group_key(i) for i in selected})}
                if not torch.isfinite(targets).all() or any(not bool((item[4] > .5).all()) for item in overfit_items):
                    raise ValueError("原生序列监督非有限或不完整，拒绝训练")
                if not diagnostic["within_chunk_open_to_closed"] or not diagnostic["within_chunk_closed_to_open"]:
                    raise ValueError("固定窗口未覆盖两个原生目标切换方向，拒绝声称双向验收；不能擅自换样")
                (output_dir / "native_fit_contract.json").write_text(json.dumps(diagnostic, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"[原生序列验收范围] {diagnostic}")
            print(f"[拟合诊断] 精确复用对照清单的 {len(selected)} 个窗口")
        else:
            if args.balanced_overfit_targets:
                train_indices, overfit_items = select_balanced_overfit_samples(
                    dataset, train_indices, args.overfit_samples, args.overfit_trajectories, output_dir)
            else:
                train_indices, overfit_items = select_overfit_samples(
                    dataset, train_indices, args.overfit_samples, args.overfit_trajectories)
        print("[拟合诊断] 后续指标在已见训练窗口上计算，不是独立验证或机器人成功率")
    if gripper_fit_checkpoint is not None:
        current_splits = {"train": train_indices, "validation": validation_indices, "test": test_indices}
        if gripper_fit_checkpoint.get("split_indices") != current_splits:
            raise ValueError("仅夹爪对照必须与源模型精确使用相同32窗口及留出分区")
        if gripper_fit_checkpoint["data_config"].get("chunk_size") != dataset.chunk_size:
            raise ValueError("仅夹爪对照动作长度与源模型不同")
    print(
        f"[训练 3/6] 按完整轨迹划分：训练={len(train_indices)}，"
        f"验证={len(validation_indices)}，测试={len(test_indices)}"
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_schema_counts = Counter(
        str(dataset.samples[index]["source"]) for index in train_indices
    )
    print(f"[训练] 训练集各 schema 原始片段数：{dict(train_schema_counts)}")
    train_sampler = (
        build_balanced_sampler(dataset, train_indices, generator)
        if args.balanced_sampling
        else None
    )
    print(
        "[训练] 采样策略："
        + ("按 schema 均衡抽样" if train_sampler is not None else "按全部片段普通随机抽样")
    )
    train_subset = Subset(dataset, train_indices)
    if args.offline_command_experiment:
        train_subset = IndexedTrainingSubset(dataset, train_indices)
    if overfit_items is not None:
        train_subset = overfit_items  # 一次解码后缓存；每轮不再重读 TFRecord。
        if args.gripper_only_fit_from:
            train_subset = IndexedTrainingSubset(overfit_items, list(range(len(overfit_items))))
    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        generator=generator if train_sampler is None else None,
        collate_fn=collate_indexed_training if (args.offline_command_experiment or args.gripper_only_fit_from) else collate_batch,
        drop_last=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )
    validation_loader = DataLoader(
        Subset(dataset, validation_indices),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )
    metric_generator = torch.Generator().manual_seed(args.seed + 10_000)
    metric_order = torch.randperm(
        len(validation_indices), generator=metric_generator
    ).tolist()
    metric_indices = [validation_indices[position] for position in metric_order]
    validation_metric_loader = DataLoader(
        Subset(dataset, metric_indices),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )
    validation_probe_indices = []
    if native_pilot or (bridge_selection and not args.overfit_samples and args.bridge_validation_scope == "one_per_episode"):
        # 每条验证演示固定随机选一个窗口，不让长演示的重叠窗口主导验证。
        groups = defaultdict(list)
        for index in validation_indices:
            groups[dataset.group_key(index)].append(index)
        probe_generator = torch.Generator().manual_seed(args.seed)
        validation_probe_indices = [values[torch.randperm(len(values), generator=probe_generator)[0].item()]
                                    for values in groups.values()]
        validation_loader = DataLoader(Subset(dataset, validation_probe_indices), batch_size=args.batch_size,
                                       shuffle=False, collate_fn=collate_batch, num_workers=args.workers)
        validation_metric_loader = validation_loader
        print(f"[验证协议] 每条独立验证演示固定1窗口，共{len(validation_probe_indices)}窗口；不按标签筛选")
    if bridge_selection and args.bridge_validation_scope == "all_windows":
        if (args.max_validation_batches != 0 or args.metric_interval != 1
                or args.metric_samples < args.batch_size
                or args.metric_batches * args.batch_size < len(validation_indices)):
            raise ValueError("全窗口验证须每轮完整计算损失及所有动作指标，不得仅抽样部分窗口")
        print(f"[验证协议] 全部{len(validation_indices)}个留出窗口；来自"
              f"{len({dataset.group_key(i) for i in validation_indices})}条独立演示，重叠窗口不是独立样本")
    if overfit_items is not None:
        validation_loader = DataLoader(overfit_items, batch_size=args.batch_size, collate_fn=collate_batch)
        validation_metric_loader = validation_loader
        manifest = {
            "purpose": "training_fit_only_not_generalization",
            "selected_indices": train_indices,
            "selected_windows": [dataset.samples[i] for i in train_indices],
            "held_out_validation_indices": validation_indices,
            "held_out_test_indices": test_indices,
        }
        if bridge_selection:
            manifest = bridge_fit_contract
        (output_dir / "overfit_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        "[训练] 验证范围："
        + (
            f"每轮最多 {args.max_validation_batches} 个 batch"
            if args.max_validation_batches > 0
            else "每轮使用完整既定验证协议（原生完整池为每演示1窗口）"
        )
    )

    print(f"[训练 4/6] 加载冻结的 CLIP 主干：{args.model_name}")
    tokenizer = CLIPTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)
    config = (
        resume_checkpoint["config"]
        if resume_checkpoint is not None
        else build_config(args)
    )
    checkpoint_representation = config.get("action", {}).get("representation")
    if checkpoint_representation != ACTION_REPRESENTATION:
        raise ValueError(
            "checkpoint 使用旧的绝对坐标动作，不能与新的相对动作数据继续训练。"
            f"期望={ACTION_REPRESENTATION!r}，实际={checkpoint_representation!r}。"
            "请不要使用 --resume，重新训练到新的输出目录。"
        )
    if not bool(config.get("model", {}).get("separate_gripper_head", False)):
        raise ValueError(
            "该 checkpoint 仍把二值夹爪作为连续扩散量，不能恢复到新版训练。"
            "请不要使用 --resume，重新训练独立夹爪头模型。"
        )
    if not bool(config.get("model", {}).get("trajectory_conditioned_gripper", False)):
        raise ValueError(
            "该 checkpoint 的夹爪头没有使用位姿轨迹条件，不能作为新版断点恢复。"
            "请使用 --init-from 迁移旧位姿权重。"
        )
    if not bool(config.get("model", {}).get("condition_on_current_gripper", False)):
        raise ValueError(
            "该 checkpoint 未使用输入时刻夹爪状态，不能作为新版断点恢复。"
            "请使用 --init-from 迁移旧位姿权重。"
        )
    checkpoint_chunk_size = int(config["model"].get("chunk_size", args.chunk_size))
    if checkpoint_chunk_size != args.chunk_size:
        raise ValueError(
            "恢复训练时动作窗口必须与 checkpoint 一致："
            f"checkpoint={checkpoint_chunk_size}，当前参数={args.chunk_size}"
        )
    model = RobotAdapterModel(config=config, cache_dir=args.cache_dir).to(device)
    if pool_fit_checkpoint is not None:
        expected_keys = set(trainable_state_dict(model))
        if set(pool_fit_checkpoint["trainable_state_dict"]) != expected_keys:
            raise ValueError("源模型可训练权重不完整或结构不一致")
        model.load_state_dict(pool_fit_checkpoint["trainable_state_dict"], strict=False)
        print("[完整池拟合诊断] 沿用源模型固定已见探针及全部可训练权重；优化器/学习率周期重新开始，不是完整训练恢复")
    frozen_module_digest_before = None
    if gripper_fit_checkpoint is not None:
        source_state = gripper_fit_checkpoint["trainable_state_dict"]
        load_gripper_fit_weights(model, source_state, reset_readout=args.reset_gripper_readout, seed=args.seed)
        print(f"[夹爪结构对照] 读出={args.gripper_head_type}，重置读出={args.reset_gripper_readout}；输入投影沿用同一源权重")
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith(GRIPPER_PARAMETER_PREFIXES))
        frozen_module_digest_before = frozen_policy_digest(model)
        print("[仅夹爪对照] 图文/位姿及夹爪输入投影已载入同源权重；冻结CLIP/图文融合/位姿头；"
              + ("仅读出头重新初始化" if args.reset_gripper_readout else "夹爪读出头沿用原权重"))
    print(f"[融合] 实际视觉读出：{getattr(model, 'adapter_pooling', '不适用')}；解码器：{model.decoder_type}")
    if init_checkpoint is not None:
        source_representation = (
            init_checkpoint["config"].get("action", {}).get("representation")
        )
        if source_representation != ACTION_REPRESENTATION:
            raise ValueError("--init-from 的动作表示与当前相对动作不兼容")
        current_state = model.state_dict()
        compatible_state = {}
        skipped_keys = []
        source_gripper_mode = str(
            init_checkpoint["config"].get("model", {}).get(
                "gripper_target_mode", "state"
            )
        )
        target_gripper_mode = str(config["model"].get("gripper_target_mode", "state"))
        gripper_prefixes = (
            "gripper_head.",
            "gripper_context_projection.",
            "gripper_pose_projection.",
            "gripper_time_embedding.",
            "current_gripper_projection.",
        )
        for key, value in init_checkpoint["trainable_state_dict"].items():
            changed_gripper_semantics = (
                source_gripper_mode != target_gripper_mode
                and key.startswith(gripper_prefixes)
            )
            if (
                not changed_gripper_semantics
                and key in current_state
                and current_state[key].shape == value.shape
            ):
                compatible_state[key] = value
            else:
                skipped_keys.append(key)
        incompatible = model.load_state_dict(compatible_state, strict=False)
        if incompatible.unexpected_keys:
            raise ValueError(
                f"初始化 checkpoint 含未知参数：{incompatible.unexpected_keys}"
            )
        print(
            "[迁移] 已载入语义兼容的融合模块与位姿扩散权重；"
            f"跳过 {len(skipped_keys)} 个不兼容旧参数，"
            "新增或缺失模块保持随机初始化"
        )
        print("[迁移] 优化器与学习率重新开始")
        del init_checkpoint
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    total_parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(parameter.numel() for parameter in trainable_parameters)
    print(
        f"[训练] 参数量：总计={total_parameter_count / 1e6:.2f}M，"
        f"可训练={trainable_parameter_count / 1e6:.2f}M，"
        f"占比={100.0 * trainable_parameter_count / total_parameter_count:.2f}%"
    )
    gripper_parameter_names = (
        "gripper_head.",
        "gripper_context_projection.",
        "gripper_pose_projection.",
        "gripper_time_embedding.",
        "current_gripper_projection.",
    )
    named_trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    pose_parameters = [
        parameter
        for name, parameter in named_trainable
        if not name.startswith(gripper_parameter_names)
    ]
    gripper_parameters = [
        parameter
        for name, parameter in named_trainable
        if name.startswith(gripper_parameter_names)
    ]
    gripper_learning_rate = (
        args.gripper_learning_rate
        if args.gripper_learning_rate is not None
        else args.learning_rate
    )
    optimizer = torch.optim.AdamW(
        [
            {"params": pose_parameters, "lr": args.learning_rate},
            {"params": gripper_parameters, "lr": gripper_learning_rate},
        ],
        weight_decay=args.weight_decay,
    )
    print(
        f"[训练] 学习率：Adapter/位姿={args.learning_rate:.2e}，"
        f"夹爪={gripper_learning_rate:.2e}"
    )
    scheduler = build_learning_rate_scheduler(optimizer, args.lr_schedule, args.epochs)
    print(f"[训练] 学习率策略：{args.lr_schedule}；每轮结束后更新，constant保持两组初始学习率")
    criterion = nn.MSELoss()
    if model.decoder_type == "diffusion":
        print(f"[扩散] 噪声表={model.beta_schedule}；终点信号保留 alpha_bar={model.alpha_bars[-1].item():.8f}")
    if args.overfit_samples:
        with torch.random.fork_rng(devices=[device.index or 0] if device.type == "cuda" else []):
            set_seed(args.seed + 20_000)
            before = evaluate_action_metrics(model, validation_metric_loader, tokenizer, device, args.metric_batches, args.metric_samples)
        (output_dir / "before_training.json").write_text(json.dumps(before, indent=2), encoding="utf-8")
        print(f"[拟合诊断] 训练前动作指标：{before}")
    best_validation_loss = float("inf")
    history: List[Dict[str, float | int]] = []
    training_access_order = []
    fit_probe_indices = []
    start_epoch = 1

    if resume_checkpoint is not None:
        training_access_order = list(resume_checkpoint.get("training_access_order", []))
        fit_probe_indices = list(resume_checkpoint.get("fit_probe_indices", []))
        incompatible = model.load_state_dict(
            resume_checkpoint["trainable_state_dict"], strict=False
        )
        if incompatible.unexpected_keys:
            raise ValueError(f"checkpoint 含有无法识别的参数：{incompatible.unexpected_keys}")
        optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        source_schedule = resume_checkpoint.get("run_arguments", {}).get("lr_schedule", "cosine")
        if source_schedule != args.lr_schedule:
            raise ValueError("续训不能无提示切换学习率策略；对照实验请从头训练到新目录")
        if "scheduler_state_dict" in resume_checkpoint:
            scheduler.load_state_dict(resume_checkpoint["scheduler_state_dict"])
            if args.lr_schedule == "cosine":
                scheduler.T_max = max(1, args.epochs)
        start_epoch = int(resume_checkpoint["epoch"]) + 1
        best_validation_loss = float(
            resume_checkpoint.get(
                "best_validation_loss", resume_checkpoint.get("validation_loss", float("inf"))
            )
        )
        history = list(resume_checkpoint.get("history", []))
        if not history and (output_dir / "history.json").exists():
            history = json.loads((output_dir / "history.json").read_text(encoding="utf-8"))
        print(
            f"[恢复] checkpoint 已完成 epoch {start_epoch - 1}；"
            f"将从 epoch {start_epoch} 继续，历史最佳验证损失={best_validation_loss:.6f}"
        )
        # optimizer/model 已经接管所需张量，释放原始大字典以降低内存占用。
        del resume_checkpoint

    if start_epoch > args.epochs:
        print(
            f"[训练] checkpoint 已完成 {start_epoch - 1} 个 epoch，"
            f"不小于目标 --epochs={args.epochs}，无需继续训练。"
        )
        return

    if args.gripper_only_fit_from:
        print("[训练 5/6] 仅优化夹爪投影、时间编码和读出头；图文融合与位姿模块冻结")
    else:
        print(f"[训练 5/6] 开始优化{'图文融合基线' if model.fusion_type == 'simple_concat' else 'Adapter'}与{'扩散' if model.decoder_type == 'diffusion' else '回归'}位姿头、夹爪头；CLIP保持冻结")
    for epoch in range(start_epoch, args.epochs + 1):
        print(
            f"\n[轮次] Epoch {epoch}/{args.epochs} 开始；"
            f"本轮最多训练 {args.max_steps_per_epoch or '全部'} 个 batch"
        )
        model.train()
        running_loss = 0.0
        step_count = 0
        epoch_loader = train_loader
        if overfit_items is not None:
            # 一个 epoch 反复访问同样少量窗口；普通训练保持原有一遍采样。
            def repeated_batches():
                while True:
                    yield from train_loader
            epoch_loader = repeated_batches()
        for batch in epoch_loader:
            instructions, images, current_grippers, actions, supervision_masks = batch[:5]
            if (args.max_steps_per_epoch or (20 if args.overfit_samples else 0)) and step_count >= (args.max_steps_per_epoch or 20):
                break
            text = tokenise(tokenizer, instructions, device)
            device_actions = actions.to(device, non_blocking=True)
            model_output = model(
                images.to(device, non_blocking=True),
                text["input_ids"],
                attention_mask=text.get("attention_mask"),
                current_gripper=current_grippers.to(device, non_blocking=True),
                actions=device_actions,
            )
            loss, pose_loss, gripper_loss = policy_loss(
                model,
                model_output,
                device_actions,
                criterion,
                current_grippers=current_grippers.to(device, non_blocking=True),
                supervision_masks=supervision_masks.to(device, non_blocking=True),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # 仅诊断首步：记录裁剪前梯度与更新量，不改变损失、优化器或随机采样。
            audit_before = None
            if args.audit_first_update and epoch == start_epoch and step_count == 0:
                audit_before = {name: p.detach().cpu().clone() for name, p in model.named_parameters() if p.requires_grad}
                update_audit = {"groups": {}, "scope": "first_training_batch_only"}
                for group, prefixes in (("adapter", ("adapter.",)), ("pose", ("regression_head.",)),
                                        ("gripper", GRIPPER_PARAMETER_PREFIXES)):
                    parameters = [p for name, p in model.named_parameters() if name.startswith(prefixes) and p.requires_grad]
                    update_audit["groups"][group] = {
                        "parameter_tensors": len(parameters),
                        "gradient_tensors": sum(p.grad is not None for p in parameters),
                        "gradient_l2_before_clip": sum(float(p.grad.detach().float().square().sum()) for p in parameters if p.grad is not None) ** .5,
                        "finite_gradients": all(bool(torch.isfinite(p.grad).all()) for p in parameters if p.grad is not None),
                    }
                encoder_parameters = list(model.vision_encoder.parameters()) + list(model.text_encoder.parameters())
                update_audit["clip_frozen_no_grad"] = all(not p.requires_grad and p.grad is None for p in encoder_parameters)
                before_pose = model_output[0].detach().clone()
            torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=1.0)
            optimizer.step()
            if audit_before is not None:
                for group, prefixes in (("adapter", ("adapter.",)), ("pose", ("regression_head.",)),
                                        ("gripper", GRIPPER_PARAMETER_PREFIXES)):
                    deltas = [(p.detach().cpu() - audit_before[name]).float() for name, p in model.named_parameters()
                              if name in audit_before and name.startswith(prefixes)]
                    update_audit["groups"][group]["update_l2"] = sum(float(d.square().sum()) for d in deltas) ** .5
                with torch.no_grad():
                    after_output = model(images.to(device), text["input_ids"], attention_mask=text.get("attention_mask"),
                                         current_gripper=current_grippers.to(device), actions=device_actions)
                update_audit["pose_output_change_mae"] = float((after_output[0] - before_pose).abs().mean())
                (output_dir / "first_update_audit.json").write_text(json.dumps(update_audit, indent=2), encoding="utf-8")
                print("[首步审计] 梯度、参数更新和同批输出变化已保存：first_update_audit.json；不代表拟合通过")
                del audit_before, deltas, before_pose, after_output
            if args.offline_command_experiment:
                training_access_order.extend(batch[5])
            elif args.gripper_only_fit_from:
                training_access_order.extend(train_indices[index] for index in batch[5])
            running_loss += float(loss.item())
            step_count += 1
            if step_count % args.log_interval == 0:
                print(
                    f"[训练] epoch={epoch:03d} step={step_count:04d} "
                    f"total_loss={loss.item():.6f} "
                    f"{'pose_regression' if model.decoder_type == 'regression' else 'pose_x0' if model.diffusion_prediction_type == 'sample' else 'pose_noise'}={pose_loss.item():.6f} "
                    f"gripper_bce={gripper_loss.item():.6f}"
                )

        if step_count == 0:
            raise RuntimeError("Training loader produced no batches")
        train_loss = running_loss / step_count
        scope = "已见训练窗口拟合检查" if args.overfit_samples else "独立验证"
        print(f"[{scope}] Epoch {epoch} 训练结束，正在计算损失……")
        with (torch.random.fork_rng(devices=[device.index or 0] if device.type == "cuda" else []) if multitask_pilot else nullcontext()):
            if multitask_pilot:
                set_seed(args.seed + 10_000)
            validation_loss = evaluate_loss(
                model,
                validation_loader,
                tokenizer,
                device,
                criterion,
                max_batches=(args.max_validation_batches if args.max_validation_batches > 0 else None),
            )
        action_metrics: Dict[str, float] = {}
        should_measure_actions = (
            args.metric_batches > 0
            and (epoch == 1 or epoch % args.metric_interval == 0 or epoch == args.epochs)
        )
        if should_measure_actions:
            print(
                f"[指标] 正在运行少量{'完整反向扩散' if model.decoder_type == 'diffusion' else '直接回归预测'}，"
                "计算位置、旋转和夹爪误差；这一步会比普通验证慢……"
            )
            with torch.random.fork_rng(devices=[device.index or 0] if device.type == "cuda" else []):
                if args.overfit_samples or multitask_pilot:
                    set_seed(args.seed + 20_000)  # 轮次间保持相同采样噪声，便于比较。
                action_metrics = evaluate_action_metrics(
                    model, validation_metric_loader, tokenizer, device,
                    max_batches=args.metric_batches, samples_per_batch=args.metric_samples,
                )
        scheduler.step()
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "optimized_batches": step_count,
        }
        record.update(action_metrics)
        if args.offline_command_experiment:
            # 第一轮选出固定已见探针，后续轮次不换窗口来制造曲线改善。
            if not fit_probe_indices:
                fit_probe_indices = list(dict.fromkeys(training_access_order))[:args.metric_batches * min(args.metric_samples, args.batch_size)]
            if should_measure_actions and fit_probe_indices:
                fit_loader = DataLoader(Subset(dataset, fit_probe_indices), batch_size=args.batch_size,
                                        shuffle=False, collate_fn=collate_batch, num_workers=0)
                fit_metrics = evaluate_action_metrics(model, fit_loader, tokenizer, device,
                                                     args.metric_batches, args.metric_samples)
                record.update({f"train_fit_{key}": value for key, value in fit_metrics.items()})
                print(f"[固定已见拟合] 窗口={len(fit_probe_indices)}，位置={fit_metrics['position_error_cm']:.2f}cm，"
                      f"不动={fit_metrics['zero_motion_position_error_cm']:.2f}cm，夹爪={fit_metrics['gripper_accuracy']:.1%}")
            record["visited_train_windows"] = len(set(training_access_order))
            record["visited_train_episodes"] = len({dataset.group_key(i) for i in training_access_order})
        if args.overfit_samples:
            record["diagnostic_training_fit"] = True
        if args.gripper_only_fit_from:
            record["diagnostic_gripper_only_fit"] = True
            if frozen_policy_digest(model) != frozen_module_digest_before:
                raise RuntimeError("冻结模块发生变化，拒绝保存该夹爪对照")
        history.append(record)
        print(
            f"[轮次] epoch={epoch:03d} train_loss={train_loss:.6f} "
            f"validation_loss={validation_loss:.6f}"
        )
        if action_metrics:
            if native_fit:
                print(f"[原生目标对] 打开→关闭={action_metrics['gripper_target_pair_open_to_closed_accuracy']:.1%} "
                      f"(n={action_metrics['gripper_target_pair_open_to_closed_count']:.0f})；关闭→打开="
                      f"{action_metrics['gripper_target_pair_closed_to_open_accuracy']:.1%} "
                      f"(n={action_metrics['gripper_target_pair_closed_to_open_count']:.0f})；不是物理事件成功率")
            print(
                f"[指标] position_error={action_metrics['position_error_cm']:.2f} cm，"
                f"rotation_error={action_metrics['rotation_error_deg']:.2f}°，"
                f"gripper_accuracy={action_metrics['gripper_accuracy']:.1%}，"
                f"gripper_balanced={action_metrics['gripper_balanced_accuracy']:.1%}，"
                f"majority_baseline={action_metrics['gripper_majority_baseline']:.1%}，"
                f"persistence_baseline={action_metrics['gripper_persistence_baseline']:.1%}，"
                f"change_rate={action_metrics['gripper_change_rate']:.1%}，"
                f"change_accuracy={action_metrics['gripper_change_accuracy']:.1%}，"
                f"hold_accuracy={action_metrics['gripper_hold_accuracy']:.1%}，"
                f"action_mae={action_metrics['action_mae']:.4f}"
            )
            print(f"[基线] 保持位置误差={action_metrics['zero_motion_position_error_cm']:.2f}cm；保持姿态误差={action_metrics['identity_rotation_error_deg']:.2f}°")

        checkpoint = {
            "epoch": epoch,
            "trainable_state_dict": trainable_state_dict(model),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": config,
            "validation_loss": validation_loss,
            "best_validation_loss": min(best_validation_loss, validation_loss),
            "history": history,
            "dataset_size": len(dataset),
            "split_indices": {
                "train": train_indices,
                "validation": validation_indices,
                "test": test_indices,
            },
            "data_config": {
                "dataset_dir": str(args.dataset_dir),
                "chunk_size": args.chunk_size,
                "bridge_window_horizon": args.bridge_window_horizon,
                "stride": args.stride,
                "sources": list(sources),
                "max_samples": args.max_samples,
                "max_samples_per_schema": args.max_samples_per_schema,
                "max_tfrecord_episodes": args.max_tfrecord_episodes,
                "max_tfrecord_episodes_per_schema": (
                    args.max_tfrecord_episodes_per_schema
                ),
                "min_trajectory_steps": args.min_trajectory_steps,
                "exclude_path_parts": list(exclude_path_parts),
                "exclude_schemas": list(exclude_schemas),
                "tfrecord_splits": list(tfrecord_splits),
                "bcz_target": args.bcz_target,
                "bcz_current_gripper": args.bcz_current_gripper,
                "bridge_gripper_policy": args.bridge_gripper_policy,
                "bridge_current_gripper": args.bridge_current_gripper,
                "rt1_gripper_policy": args.rt1_gripper_policy,
                "bcz_reached_gripper_policy": args.bcz_reached_gripper_policy,
                "bridge_episode_selection": bridge_selection,
            },
            "experiment_kind": ("training_fit_diagnostic" if bridge_selection and args.overfit_samples else
                                "bridge_single_task_offline_diagnostic" if bridge_selection else
                                "training_fit_diagnostic" if args.overfit_samples else
                                "offline_command_pilot" if args.offline_command_experiment else "held_out_training"),
            **({"dataset_identity": dataset_split_identity(dataset)} if bridge_selection else {}),
            "run_arguments": vars(args),
            "gripper_only_fit_from": args.gripper_only_fit_from,
            "pool_fit_from": args.pool_fit_from,
            "frozen_module_digest": frozen_module_digest_before,
            "training_access_order": training_access_order,
            "fit_probe_indices": fit_probe_indices,
            "validation_probe_indices": validation_probe_indices,
            "validation_protocol": ("seen_training_windows_only" if args.overfit_samples else
                                    "all_held_out_bridge_windows" if bridge_selection and args.bridge_validation_scope == "all_windows" else
                                    "one_fixed_window_per_episode" if native_pilot or bridge_selection else "existing_protocol"),
        }
        torch.save(checkpoint, output_dir / "latest.pt")
        print(f"[保存] 已更新断点：{output_dir / 'latest.pt'}")
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            torch.save(checkpoint, output_dir / "best.pt")
            print(f"[保存] 验证损失刷新，已保存最佳模型：{output_dir / 'best.pt'}")
        (output_dir / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )

    print("\n[训练 6/6] 全部训练完成")
    print(f"[训练] 最佳验证损失：{best_validation_loss:.6f}")
    print(f"[训练] 权重和历史记录：{output_dir}")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="训练冻结 CLIP + Adapter + 动作解码器", fromfile_prefix_chars="@")
    parser.add_argument(
        "--dataset-dir",
        default=str(project_root / "training_cache" / "oxe_core"),
    )
    parser.add_argument("--cache-dir", default=r"D:\ntu_related\dissertation\hf_cache")
    parser.add_argument("--output-dir", default=str(project_root / "results" / "relative_v5"))
    parser.add_argument(
        "--sources",
        default="auto",
        help="使用 auto 时递归发现数据，并按文件内部 schema 自动选择读取器。",
    )
    parser.add_argument(
        "--exclude-path-parts",
        default=(
            "LIBERO,bridge,bridge_data_msr,old1.0.1,"
            "language_table_sim*,language_table*oracle*,"
            "VIOLA-dataset,fanuc_manipulation,cliport"
        ),
        help=(
            "逗号分隔的路径段或通配模式。默认遵循导师第四封邮件："
            "LIBERO 留作测试，并排除旧 Bridge、MSR Bridge、BC-Z 旧版以及"
            "Language Table 仿真/oracle 变体；VIOLA、Fanuc、CLIPort 留作"
            "非核心附加实验。"
        ),
    )
    parser.add_argument(
        "--tfrecord-splits",
        default="train",
        help="逗号分隔的 TFRecord 文件 split；正式训练默认只读 train。",
    )
    parser.add_argument(
        "--exclude-schemas",
        default="hdf5_position3_action4",
        help=(
            "逗号分隔的 schema；默认排除只有位置、没有旋转真值的数据，"
            "避免把补入的单位四元数当成真实8维监督。"
        ),
    )
    parser.add_argument("--model-name", default="openai/clip-vit-large-patch14")
    parser.add_argument(
        "--fusion-type",
        choices=("cross_attention", "simple_concat"),
        default="cross_attention",
        help="cross_attention 为本文方法；simple_concat 为无交叉注意力对照组。",
    )
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--bridge-window-horizon", type=int,
                        help="Fixed Bridge plan only: retain window starts from this horizon while predicting a shorter chunk.")
    parser.add_argument("--bcz-target", choices=("reached", "first_command", "native_commands"), default="reached", help="默认未来实际状态；native_commands仅用于固定窗口诊断或显式完整池离线基线。")
    parser.add_argument("--bcz-current-gripper", choices=("binary", "continuous"), default="binary", help="默认不变；continuous仅用于native_commands诊断/离线基线，从头训练。")
    parser.add_argument("--bridge-gripper-policy", choices=("threshold_v1", "reverse_scan_v1", "reverse_scan_valid_steps_v2"), default="threshold_v1", help="旧默认不变；v2先排除无效末步再扫描，新策略须显式启用并从头建立实验。")
    parser.add_argument("--offline-command-experiment", action="store_true", help="显式开放固定演示分区的BC-Z一步或原生10步回归离线基线；不作为闭环模型。")
    parser.add_argument("--native-adapter-comparison", action="store_true", help="仅原生10步离线完整池：显式启用CLS+patch交叉注意力对照；不改变动作语义或闭环限制。")
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--max-steps-per-epoch", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lr-schedule", choices=("cosine", "constant"), default="cosine",
                        help="默认余弦衰减；constant为不衰减的学习率对照，不改变模型或标签。")
    parser.add_argument(
        "--gripper-learning-rate",
        type=float,
        help="新夹爪头可使用更高学习率；不填则与主学习率相同。",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--adapter-layers", type=int, default=8)
    parser.add_argument("--adapter-pooling", choices=("cls", "cls_patch_mean"), default="cls", help="默认保留CLS读出；cls_patch_mean为已接通局部视觉通路、但尚未通过泛化验收的实验选项。")
    parser.add_argument("--attention-dim", type=int, default=512)
    parser.add_argument("--attention-heads", type=int, default=8)
    parser.add_argument("--decoder-hidden-dim", type=int, default=256)
    parser.add_argument("--diffusion-steps", type=int, default=100)
    parser.add_argument("--decoder-type", choices=("diffusion", "regression"), default="diffusion", help="regression 是定位扩散问题的直接位姿回归对照，默认仍为扩散。")
    parser.add_argument("--diffusion-prediction-type", choices=("epsilon", "sample"), default="epsilon", help="扩散预测噪声或干净动作；sample 是本轮诊断消融，旧模型默认为epsilon。")
    parser.add_argument("--beta-schedule", choices=("linear", "squaredcos_cap_v2"), default="squaredcos_cap_v2", help="新训练默认余弦；旧 checkpoint 未记录此字段时保持线性表。")
    parser.add_argument("--overfit-samples", type=int, default=0, help="非零时反复训练少量已见窗口并在这些窗口上诊断拟合；不能当泛化结果。")
    parser.add_argument("--balanced-overfit-targets", action="store_true", help="仅诊断：BC-Z第一目标选取一半打开一半关闭，不用于代表真实分布或评估泛化。")
    parser.add_argument("--overfit-trajectories", type=int, default=2, help="小样本诊断最多选取几条训练轨迹。")
    parser.add_argument("--overfit-manifest", help="精确复用已有诊断的窗口清单；数据身份不匹配时拒绝比较。")
    parser.add_argument("--bridge-gripper-fit-report", help="显式使用已核对Bridge训练夹爪候选报告；按类别选样仅用于联合拟合诊断。")
    parser.add_argument("--bridge-validation-scope", choices=("one_per_episode", "all_windows"), default="one_per_episode",
                        help="Bridge默认每演示1固定窗口；all_windows用于每轮全部留出窗口验证，非拟合检查。")
    parser.add_argument("--gripper-only-fit-from", help="仅native_commands诊断：载入同32窗口的完整权重，冻结图文与位姿，只优化夹爪；不是普通resume/init-from。")
    parser.add_argument("--gripper-head-type", choices=["legacy", "mlp2"], default="legacy", help="夹爪读出结构；旧checkpoint缺省legacy。")
    parser.add_argument("--audit-first-update", action="store_true", help="无dropout回归拟合诊断：记录首步梯度、参数更新与同批输出变化。")
    parser.add_argument("--pool-fit-from", help="仅固定窗口诊断：载入完整池模型与其保存的已见探针，重建优化器；不能用于普通续训或闭环。")
    parser.add_argument("--regression-rotation-weight", type=float, default=1.0, help="默认1不变；非默认值仅用于源完整池已见探针回归诊断的旋转损失对照。")
    parser.add_argument("--reset-gripper-readout", action="store_true", help="仅夹爪对照：重置读出头，保留同源输入投影；允许仅读出头结构不同。")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--gripper-loss-weight",
        type=float,
        default=0.25,
        help="二值夹爪 BCE 在总损失中的权重。",
    )
    parser.add_argument(
        "--gripper-change-weight",
        type=float,
        default=1.0,
        help="夹爪切换事件的额外 BCE 权重；默认1，不额外放大切换事件。",
    )
    parser.add_argument(
        "--balanced-gripper-loss",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "是否强制切换/不切换各占一半损失。默认关闭并按真实时间步频率训练，"
            "以减少跨数据集的虚假夹爪切换；开启可作为类别平衡消融实验。"
        ),
    )
    parser.add_argument(
        "--gripper-target-mode",
        choices=("state", "transition"),
        default="transition",
        help="state 逐步预测绝对开合；transition 预测切换事件并累计还原状态。",
    )
    parser.add_argument(
        "--gripper-transition-decode",
        choices=("cumulative", "single_switch"),
        default="cumulative",
        help="transition 推理方式；默认逐步累计，single_switch 仅保留作消融。",
    )
    parser.add_argument(
        "--gripper-switch-threshold",
        type=float,
        default=0.0,
        help="切换 logit 阈值；0 等价于概率0.5。",
    )
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-manifest", help="从头训练时精确复用另一组训练的 split_manifest.json；校验数据身份与轨迹隔离。")
    parser.add_argument("--prepare-only", action="store_true", help="只扫描数据、校验并保存划分清单，不加载模型或启动训练。")
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument(
        "--max-validation-batches",
        type=int,
        default=20,
        help="每轮最多验证多少个 batch；设为 0 表示完整验证集。",
    )
    parser.add_argument(
        "--metric-interval",
        type=int,
        default=5,
        help="每隔多少个 epoch 运行一次较慢的动作抽样指标；epoch 1 和最后一轮也会运行。",
    )
    parser.add_argument(
        "--metric-batches",
        type=int,
        default=1,
        help="每次动作指标最多使用多少个验证 batch；设为 0 可关闭。",
    )
    parser.add_argument(
        "--metric-samples",
        type=int,
        default=2,
        help="每个指标 batch 最多抽样多少条轨迹，数值越小越省计算。",
    )
    parser.add_argument(
        "--balanced-sampling",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="默认按 schema 均衡抽样；使用 --no-balanced-sampling 可关闭。",
    )
    parser.add_argument(
        "--resume",
        help="从 train.py 生成的 latest.pt 继续训练；epochs 表示最终总轮数。",
    )
    parser.add_argument(
        "--init-from",
        help="只载入已有模型权重并重置优化器和数据划分；适合扩充数据集后增量训练。",
    )
    parser.add_argument("--max-samples", type=int)
    parser.add_argument(
        "--max-samples-per-schema",
        type=int,
        help="每种兼容 schema 的样本上限；适合先做低成本冒烟训练。",
    )
    parser.add_argument("--max-tfrecord-episodes", type=int)
    parser.add_argument(
        "--max-tfrecord-episodes-per-schema",
        type=int,
        help=(
            "每种 TFRecord schema 最多扫描多少个 episode；适合在大规模 OXE "
            "数据上做低成本训练，同时仍覆盖每一种已识别 schema。"
        ),
    )
    parser.add_argument(
        "--min-trajectory-steps",
        type=int,
        default=10,
        help="只保留至少包含这么多时间步的完整演示；导师要求默认10。",
    )
    parser.add_argument("--bridge-current-gripper", choices=("binary", "continuous"), default="binary",
                        help="Bridge输入测量编码；连续观测2*opening-1，不等价于上一控制命令")
    parser.add_argument("--bridge-task-plan", help="固定Bridge单任务演示计划；不可与旧模型/拟合诊断混用")
    parser.add_argument("--rt1-gripper-policy", choices=("legacy_threshold_v1", "relative_scan_v2"),
                        default="legacy_threshold_v1", help="新Fractal训练须显式选择relative_scan_v2；旧权重保留原标签合同。")
    parser.add_argument("--bcz-reached-gripper-policy", choices=("future_measured_v1", "preceding_command_v2"),
                        default="future_measured_v1", help="混源到达位姿监督采用preceding_command_v2；旧观测状态标签不暗中更换。")
    return parser.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())

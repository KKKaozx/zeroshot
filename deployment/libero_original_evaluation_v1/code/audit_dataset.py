"""生成第一阶段数据验收报告和代表性轨迹图。

该脚本不训练模型。它从统一加载器中抽样，检查每种 schema 的图像、语言、
8维动作范围以及夹爪切换比例，并把可人工核对的 PNG 和 summary.json 写入
结果目录。只有这一步通过后，才应开始正式基线训练。
"""

from __future__ import annotations

import argparse
import json
import hashlib
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from dataset import (
    ACTION_REPRESENTATION,
    CLIP_IMAGE_MEAN,
    CLIP_IMAGE_STD,
    POSITION_SCALE_METERS,
    UnifiedRobotDataset,
)


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def restore_rgb(image_tensor) -> np.ndarray:
    image = image_tensor * CLIP_IMAGE_STD + CLIP_IMAGE_MEAN
    return (
        image.clamp(0.0, 1.0).permute(1, 2, 0).mul(255).byte().cpu().numpy()
    )


def evenly_spaced(indices: list[int], count: int) -> list[int]:
    if len(indices) <= count:
        return indices
    positions = np.linspace(0, len(indices) - 1, count, dtype=int)
    return [indices[int(position)] for position in positions]


def sample_identity(sample: dict) -> dict[str, object]:
    source_path = sample.get("file_path", sample.get("action_path", ""))
    trajectory = sample.get("demo_key", sample.get("record_index", "unknown"))
    return {
        "source_path": str(source_path),
        "trajectory": str(trajectory),
        "start_index": int(sample.get("start_index", 0)),
    }


def inspect_bridge_timing(state: np.ndarray, command: np.ndarray, first=None, last=None) -> dict:
    """报告时间候选与空动作，不能用相关性自动决定移动索引。

    控制命令不必等于实际到达位移；误差统计是排查线索，不是时序证明。
    """
    state, command = np.asarray(state), np.asarray(command)
    if state.ndim != 2 or state.shape[1] != 7 or command.shape != state.shape or len(state) < 2:
        raise ValueError("Bridge timing audit requires matching [T,7] state/action, T>=2")
    if not np.isfinite(state).all() or not np.isfinite(command).all():
        raise ValueError("Bridge timing fields contain nonfinite values")
    delta = np.diff(state[:, :3], axis=0)
    result = {"steps": len(state), "first_action_all_zero": bool(np.all(command[0] == 0)),
        "last_action_all_zero": bool(np.all(command[-1] == 0)),
        "first_action": command[0].tolist(), "last_action": command[-1].tolist(),
        "next_delta_vs_action_t_xyz_rmse_native_units": float(np.sqrt(np.mean((delta - command[:-1, :3]) ** 2))),
        "next_delta_vs_action_t_plus_1_xyz_rmse_native_units": float(np.sqrt(np.mean((delta - command[1:, :3]) ** 2))),
        "timestamps_available": False}
    for name, flags, expected in (("is_first", first, 0), ("is_last", last, len(state) - 1)):
        if flags is None:
            result[name + "_indices"] = None
            result[name + "_valid"] = None
        else:
            flags = np.asarray(flags)
            result[name + "_indices"] = np.flatnonzero(flags).tolist()
            result[name + "_valid"] = bool(flags.shape == (len(state),) and np.isin(flags, [0, 1]).all()
                                                and result[name + "_indices"] == [expected])
    return result


def plot_sample(
    output_path: Path,
    schema: str,
    instruction: str,
    image_tensor,
    current_gripper: float,
    actions: np.ndarray,
    predictions: np.ndarray | None = None,
) -> None:
    position_cm = actions[:, :3] * POSITION_SCALE_METERS * 100.0
    quaternion_w = np.clip(np.abs(actions[:, 6]), 0.0, 1.0)
    rotation_deg = 2.0 * np.degrees(np.arccos(quaternion_w))
    canvas = Image.new("RGB", (1200, 850), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    def text(position: tuple[int, int], value: str, fill: str = "black") -> None:
        # 默认字体不保证包含中文；审计图只需要稳定显示字段和数值。
        safe = value.encode("ascii", errors="replace").decode("ascii")
        draw.text(position, safe, fill=fill, font=font)

    def chart(
        box: tuple[int, int, int, int],
        series: list[tuple[np.ndarray, str, str]],
        title: str,
        fixed_range: tuple[float, float] | None = None,
    ) -> None:
        left, top, right, bottom = box
        draw.rectangle(box, outline="#777777", width=2)
        text((left, top - 22), title)
        values = np.concatenate([values.reshape(-1) for values, _, _ in series])
        minimum, maximum = (
            fixed_range
            if fixed_range is not None
            else (float(values.min()), float(values.max()))
        )
        if abs(maximum - minimum) < 1e-8:
            minimum -= 1.0
            maximum += 1.0
        for series_index, (values, color, label) in enumerate(series):
            points = []
            for index, value in enumerate(values):
                x = left + index * (right - left) / max(1, len(values) - 1)
                y = bottom - (float(value) - minimum) * (bottom - top) / (
                    maximum - minimum
                )
                points.append((round(x), round(y)))
            if len(points) > 1:
                draw.line(points, fill=color, width=3)
            if label:
                legend_x = left + 10 + 120 * series_index
                draw.line((legend_x, bottom + 18, legend_x + 25, bottom + 18), fill=color, width=3)
                text((legend_x + 32, bottom + 11), label)
        text((left - 5, bottom + 38), f"min={minimum:.2f}  max={maximum:.2f}")
        text((left, bottom + 55), f"waypoint 1 ... {len(actions)}")

    compact_instruction = " ".join(instruction.split())
    text((35, 20), schema)
    text((35, 42), compact_instruction[:170])
    input_image = Image.fromarray(restore_rgb(image_tensor)).resize((350, 350))
    canvas.paste(input_image, (35, 90))
    text((35, 450), "Input RGB (resized by loader to 224 x 224)")

    position_series = [
        (position_cm[:, 0], "#b2182b", "target x"),
        (position_cm[:, 1], "#167d42", "target y"),
        (position_cm[:, 2], "#452b93", "target z"),
    ]
    if predictions is not None:
        for prediction in predictions:
            for dimension, color in enumerate(("#ed8993", "#84d2a4", "#b5a2e3")):
                position_series.append((prediction[:, dimension] * POSITION_SCALE_METERS * 100, color, ""))
        text((35, 475), "Dark XYZ: target; light XYZ: predictions. All waypoints use the SAME reference frame.")
    chart(
        (440, 90, 1150, 390),
        position_series,
        "Tool-relative translation (cm)",
    )
    rotation_series = [(rotation_deg, "#222222", "angle")]
    if predictions is not None:
        rotation_series = [
            (2 * np.degrees(np.arccos(np.clip(np.abs((prediction[:, 3:7] * actions[:, 3:7]).sum(axis=-1)), 0, 1))), "#5186c4", "error" if i == 0 else "")
            for i, prediction in enumerate(predictions)
        ]
    chart(
        (35, 520, 560, 770),
        rotation_series,
        "Rotation prediction error (degrees)" if predictions is not None else "Relative rotation angle (degrees)",
    )
    gripper_series = [
        (actions[:, 7], "#222222", "target"),
        (np.full(len(actions), current_gripper, dtype=np.float32), "#999999", "current"),
    ]
    if predictions is not None:
        gripper_series.extend((prediction[:, 7], "#5186c4", "prediction" if i == 0 else "") for i, prediction in enumerate(predictions))
    chart(
        (640, 520, 1150, 770),
        gripper_series,
        "Gripper (-1 closed, +1 open)",
        fixed_range=(-1.2, 1.2),
    )
    canvas.save(output_path)


def run_audit(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    exclude_path_parts = tuple(
        value.strip() for value in args.exclude_path_parts.split(",") if value.strip()
    )
    tfrecord_splits = tuple(
        value.strip() for value in args.tfrecord_splits.split(",") if value.strip()
    )
    print("=" * 68)
    print("[审计 1/3] 扫描训练数据；本脚本不会训练或修改 checkpoint")
    dataset = UnifiedRobotDataset(
        data_dir=args.dataset_dir,
        chunk_size=args.chunk_size,
        stride=args.stride,
        sources=("auto",),
        max_samples_per_schema=args.max_samples_per_schema,
        max_tfrecord_episodes_per_schema=args.max_tfrecord_episodes_per_schema,
        min_trajectory_steps=args.min_trajectory_steps,
        exclude_path_parts=exclude_path_parts,
        tfrecord_splits=tfrecord_splits,
        bridge_gripper_policy=args.bridge_gripper_policy,
    )

    indices_by_schema: dict[str, list[int]] = defaultdict(list)
    for index, sample in enumerate(dataset.samples):
        indices_by_schema[str(sample["source"])].append(index)

    report: dict[str, object] = {
        "dataset_dir": str(Path(args.dataset_dir).resolve()),
        "action_representation": ACTION_REPRESENTATION,
        "bridge_gripper_policy": args.bridge_gripper_policy,
        "action_dimensions": 8,
        "position_scale_meters": POSITION_SCALE_METERS,
        "chunk_size": args.chunk_size,
        "stride": args.stride,
        "min_trajectory_steps": args.min_trajectory_steps,
        "excluded_path_parts": list(exclude_path_parts),
        "discovered_chunks": dict(dataset.discovered_counts),
        "retained_chunks": dict(dataset.source_counts),
        "excluded_file_count": dataset.excluded_file_count,
        "incompatible_counts": dict(dataset.incompatible_counts),
        "skipped_files": [
            {"path": path, "reason": reason} for path, reason in dataset.skipped_files
        ],
        "schemas": {},
        "formal_training_blockers": [],
        "acceptance_status": "requires_semantic_review",
        "acceptance_note": "无已记录阻断项不代表动作语义、时间对齐或泛化性能已通过验收。",
    }

    print("[审计 2/3] 解码代表性样本并计算动作范围")
    for schema, indices in sorted(indices_by_schema.items()):
        stats_indices = evenly_spaced(indices, args.stats_samples)
        visual_indices = evenly_spaced(indices, args.samples_per_schema)
        position_values: list[np.ndarray] = []
        rotation_values: list[np.ndarray] = []
        gripper_values: list[np.ndarray] = []
        change_values: list[np.ndarray] = []
        examples: list[dict[str, object]] = []

        decoded: dict[int, tuple] = {}
        for index in sorted(set(stats_indices + visual_indices)):
            decoded[index] = dataset[index]
        for index in stats_indices:
            _, _, current_gripper_tensor, action_tensor, supervision_mask = decoded[index]
            actions = action_tensor.numpy()
            mask = supervision_mask.numpy() > 0.5
            # 缺失的旋转/夹爪是接口占位值，不能计入真实数据统计。
            positions_cm = actions[:, :3] * POSITION_SCALE_METERS * 100.0
            position_values.append(np.where(mask[:, :3], positions_cm, np.nan))
            rotation = (
                2.0
                * np.degrees(np.arccos(np.clip(np.abs(actions[:, 6]), 0.0, 1.0)))
            )
            rotation_values.append(rotation[mask[:, 3:7].all(axis=1)])
            gripper_open = actions[:, 7] > 0.0
            current_open = float(current_gripper_tensor.item()) > 0.0
            gripper_values.append(gripper_open[mask[:, 7]].astype(np.float32))
            change_values.append((gripper_open != current_open)[mask[:, 7]].astype(np.float32))

        for order, index in enumerate(visual_indices, start=1):
            instruction, image, current_gripper, actions, supervision_mask = decoded[index]
            file_name = f"{schema}_sample_{order:02d}.png"
            plot_sample(
                output_dir / file_name,
                schema,
                instruction,
                image,
                float(current_gripper.item()),
                actions.numpy(),
            )
            example = sample_identity(dataset.samples[index])
            example.update({"instruction": instruction, "plot": file_name})
            examples.append(example)

        positions = np.concatenate(position_values, axis=0)
        rotations = np.concatenate(rotation_values, axis=0)
        grippers = np.concatenate(gripper_values, axis=0)
        changes = np.concatenate(change_values, axis=0)
        representative_mask = decoded[stats_indices[0]][4][0]
        supervised_dimensions = torch.where(representative_mask > 0.5)[0].tolist()
        has_full_pose = all(index in supervised_dimensions for index in range(7))
        def position_stat(operation):
            return [
                round(float(operation(column[np.isfinite(column)])), 4)
                if np.isfinite(column).any() else None
                for column in positions.T
            ]

        rotation_mean = float(rotations.mean()) if rotations.size else None
        open_rate = float(grippers.mean()) if grippers.size else None
        change_rate = float(changes.mean()) if changes.size else None
        report["schemas"][schema] = {
            "retained_chunks": len(indices),
            "stats_samples": len(stats_indices),
            "pose_supervision": (
                "position_and_rotation" if has_full_pose else "position_only"
            ),
            "full_action_supervision": len(supervised_dimensions) == 8,
            "training_ready_with_mask": bool(supervised_dimensions),
            "supervised_action_dimensions": supervised_dimensions,
            "position_cm_min": position_stat(np.min),
            "position_cm_max": position_stat(np.max),
            "position_cm_mean": position_stat(np.mean),
            "rotation_deg_mean": round(rotation_mean, 4) if rotation_mean is not None else None,
            "rotation_deg_max": round(float(rotations.max()), 4) if rotations.size else None,
            "gripper_open_rate": round(open_rate, 6) if open_rate is not None else None,
            "gripper_change_rate": round(change_rate, 6) if change_rate is not None else None,
            "examples": examples,
        }
        print(
            f"[审计] {schema}: position="
            f"{position_stat(np.min)}..{position_stat(np.max)} cm, "
            f"rotation_mean={rotation_mean}, open={open_rate}, change={change_rate}"
        )

    cliport_short = int(
        dataset.incompatible_counts.get("cliport_trajectory_too_short", 0)
    )
    if cliport_short:
        report["formal_training_blockers"].append(
            {
                "schema": "cliport",
                "reason": (
                    f"{cliport_short} 条候选演示不足 {args.min_trajectory_steps} 个有效步骤，"
                    "未达到导师规定的轨迹长度门槛。"
                ),
                "recommended_action": "保留原始文件，但不纳入本轮低层轨迹训练。",
            }
        )

    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("[审计 3/3] 第一阶段数据审计文件已生成")
    print(f"[审计] 汇总：{summary_path}")
    print(f"[审计] 轨迹图目录：{output_dir}")
    blocker_count = len(report["formal_training_blockers"])
    print(f"[审计] 正式训练阻断项：{blocker_count} 个；详情见 summary.json")
    print("[审计] 阻断项为 0 仅表示本脚本未记录阻断，不能视为数据语义已验收。")
    print("[审计] 请人工确认图像、指令、位移方向和夹爪切换时刻后再正式训练。")


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="生成统一机器人数据的第一阶段验收报告")
    parser.add_argument("--single-source-check", choices=("bcz", "bridge"),
                        help="独立原始记录/图像/PyBullet转换核对；单一来源，不训练")
    parser.add_argument("--cliport-native-only", action="store_true",
                        help="只核验CLIPort训练分区原生primitive及观测索引，不转换8维目标或运行仿真")
    parser.add_argument("--tfds-reference", action="store_true",
                        help="Bridge单来源审计：用元数据驱动的官方TFDS入口独立对照原始字段")
    parser.add_argument("--bridge-timeline", action="store_true",
                        help="保存Bridge整条演示的画面、原生状态和夹爪命令时序图；不训练")
    parser.add_argument("--bridge-task-plan", action="store_true",
                        help="仅用官方TFDS统计Bridge任务演示数量并生成单任务划分计划，不加载模型")
    parser.add_argument("--bridge-gripper-policy", choices=("threshold_v1", "reverse_scan_v1"), default="threshold_v1")
    parser.add_argument(
        "--dataset-dir", default=str(project_root / "training_cache" / "oxe_core")
    )
    parser.add_argument(
        "--output-dir", default=str(project_root / "results" / "data_audit")
    )
    parser.add_argument(
        "--exclude-path-parts",
        default=(
            "LIBERO,bridge,bridge_data_msr,old1.0.1,"
            "language_table_sim*,language_table*oracle*,"
            "VIOLA-dataset,fanuc_manipulation,cliport"
        ),
    )
    parser.add_argument("--tfrecord-splits", default="train")
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--min-trajectory-steps", type=int, default=10)
    parser.add_argument("--max-samples-per-schema", type=int, default=512)
    parser.add_argument("--max-tfrecord-episodes-per-schema", type=int)
    parser.add_argument("--stats-samples", type=int, default=32)
    parser.add_argument("--samples-per-schema", type=int, default=3)
    return parser.parse_args()


def audit_cliport_native(args):
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Preserve existing reports; choose a new output directory")
    files = sorted(Path(args.dataset_dir).glob("**/*-train/action/*.pkl"))
    if not files:
        raise ValueError("No CLIPort *-train/action/*.pkl files found")
    rows = []
    for path in files:
        episode = UnifiedRobotDataset.read_cliport_native_episode(path)
        positions = []
        for step in episode["executable_step_indices"]:
            action = episode["action"][step]
            serial = {key: [np.asarray(v).tolist() for v in pose] for key, pose in action.items()}
            restored = json.loads(json.dumps(serial, allow_nan=False))
            for key in ("pose0", "pose1"):
                for original, actual in zip(action[key], restored[key]):
                    assert np.array_equal(np.asarray(original), np.asarray(actual))
                positions.append(np.asarray(action[key][0], dtype=float))
        rows.append(dict(file=str(path.relative_to(Path(args.dataset_dir))),
            action_file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            observation_records=len(episode["color"]), primitive_actions=len(episode["executable_step_indices"]),
            executable_step_indices=episode["executable_step_indices"],
            color_shape=list(np.asarray(episode["color"]).shape),
            instructions=list(dict.fromkeys(episode["info"][t]["lang_goal"] for t in episode["executable_step_indices"])),
            native_xyz_min_m=np.min(positions,axis=0).tolist(), native_xyz_max_m=np.max(positions,axis=0).tolist(),
            json_roundtrip_preserves_pose=True, unified_training_compatible=False))
    output.mkdir(parents=True, exist_ok=True)
    report = dict(action_representation="cliport_world_pick_place_v1", training_episodes=len(rows),
        primitive_actions=sum(r["primitive_actions"] for r in rows), native_record_checks_passed=True,
        trained=False, test_targets_used=False, simulation_executed=False, unified_conversion_verified=False,
        note="Existing unified loader already excludes CLIPort; native checks do not prove physical replay success.", rows=rows)
    (output/"cliport_native.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"CLIPORT NATIVE RECORD CHECK: PASSED; episodes={len(rows)}, primitives={report['primitive_actions']}; no rollout")


def prepare_bridge_task_plan(args: argparse.Namespace) -> None:
    """先按指令/来源元数据固定任务与完整演示分区，不按预测误差挑任务。

    计划不是train.py的split_manifest：还未接通训练筛选和输入编码契约。
    不利用状态/动作数值挑选任务，不作物理成功判断。
    """
    import random
    import tensorflow_datasets as tfds
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError("输出目录非空，请指定新目录，保留旧报告")
    limit = args.max_tfrecord_episodes_per_schema or 256
    if limit < 1:
        raise ValueError("扫描episode数量必须大于0")
    builder = tfds.builder_from_directory(str(Path(args.dataset_dir).resolve()))
    if builder.info.name != "bridge_data_v2":
        raise ValueError("单任务计划目前只接受已核查的bridge_data_v2格式")
    split = builder.info.splits["train"]
    limit = min(limit, split.num_examples)
    official = builder.as_dataset(split=f"train[:{limit}]", shuffle_files=False,
        decoders={"steps": tfds.decode.SkipDecoding()},
        read_config=tfds.ReadConfig(interleave_cycle_length=1, interleave_block_length=1))
    tasks = defaultdict(list)
    skipped = defaultdict(int)
    seen = set()
    scanned = 0
    shard_lengths = list(split.shard_lengths)
    shard_names = list(split.filenames)
    shard_index, shard_start = 0, 0
    print(f"[单任务准备] 官方TFDS扫描前{limit}条原始train演示；不读取模型或选择权重", flush=True)
    for index, episode in enumerate(tfds.as_numpy(official)):
        scanned += 1
        while index >= shard_start + shard_lengths[shard_index]:
            shard_start += shard_lengths[shard_index]
            shard_index += 1
        steps, meta = episode["steps"], episode["episode_metadata"]
        n = len(steps["language_instruction"])
        if n < args.min_trajectory_steps:
            skipped["too_short"] += 1
            continue
        texts = {" ".join(v.decode("utf-8", errors="replace").lower().split())
                 for v in steps["language_instruction"] if v.strip()}
        if len(texts) != 1:
            skipped["empty_or_multiple_instructions"] += 1
            continue
        if any(not v.strip() for v in steps["language_instruction"]):
            skipped["partially_missing_instruction"] += 1
            continue
        first, last = steps["is_first"], steps["is_last"]
        if np.flatnonzero(first).tolist() != [0] or np.flatnonzero(last).tolist() != [n - 1]:
            skipped["invalid_boundaries"] += 1
            continue
        origin = meta["file_path"].decode("utf-8", errors="replace")
        episode_id = int(meta["episode_id"])
        identity = (origin, episode_id)
        if not origin.strip() or identity in seen:
            skipped["missing_or_duplicate_origin_identity"] += 1
            continue
        seen.add(identity)
        tasks[next(iter(texts))].append({"shard": shard_names[shard_index],
            "record_index": index - shard_start, "origin_file_path": origin,
            "episode_id": episode_id, "steps": n})
        if scanned % 64 == 0:
            print(f"[单任务准备] 已检查{scanned}/{limit}条，当前{len(tasks)}种规范化指令", flush=True)
    counts = {task: len(rows) for task, rows in sorted(tasks.items(), key=lambda item: (-len(item[1]), item[0]))}
    eligible = [task for task, count in counts.items() if count >= 20]
    selected = eligible[0] if eligible else None
    partitions = {}
    if selected:
        rows = list(tasks[selected])
        random.Random(42).shuffle(rows)
        n = len(rows)
        train_end, validation_end = int(n * .7), int(n * .7) + int(n * .15)
        partitions = {"train": rows[:train_end], "validation": rows[train_end:validation_end],
                      "test": rows[validation_end:]}
    output.mkdir(parents=True, exist_ok=True)
    report = {"purpose": "bridge_single_task_metadata_plan_not_training_manifest",
        "dataset_dir": str(Path(args.dataset_dir).resolve()), "version": str(builder.info.version),
        "scan": "deterministic_prefix_of_original_train_split", "scanned_episodes": scanned,
        "task_episode_counts": counts, "skipped": dict(skipped), "selected_instruction": selected,
        "selection_rule": "at_least_20_episodes_then_largest_count_lexical_tie_break; no_model_errors",
        "seed": 42, "partitions": partitions,
        "contract": {"output": "xyz3 + xyzw4 + gripper_command1 = 8",
            "pose_target": "reached_observation_relative_to_input; not_native_motion_command",
            "gripper_target": "reverse_scan_v1 command[j-1], not observed_state[j,6]",
            "recommended_gripper_target_mode": "state (absolute command), not transition",
            "current_input": "continuous measured opening proposed; not yet implemented for Bridge",
            "legacy_input_warning": "state[:,6]>=0.5 does NOT establish prior commanded open/closed",
            "physical_semantics_verified": False},
        "ready_to_train": False,
        "limitations": ["按完整来源演示身份去重隔离，不能证明视觉/场景独立或别名任务等价。",
            "仅扫描前缀，计数不代表全数据；计划不采用模型误差选任务。",
            "不利用动作或状态数值选任务，未训练；尚需版本化输入契约与训练筛选入口。",
            "原始train split可能含项目旧内部留出集；本计划是新的单任务诊断协议，不是最终零样本测试。"]}
    target = output / "bridge_task_plan.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[单任务准备] 任务={selected!r}，演示分区="
          f"{ {k: len(v) for k, v in partitions.items()} }；未训练。报告：{target.resolve()}", flush=True)


def compare_bridge_tfds(data_dir: str, wanted: dict, records: dict) -> dict:
    """官方入口与手动字段解析对照；只读取选中记录所在的局部分片范围。

    SkipDecoding保留JPEG字节，先排除不同JPEG解码器带来的像素差异。
    TFDS确认序列化/顺序，不提供该导出版本的物理单位或坐标标定证明。
    """
    import tensorflow_datasets as tfds
    builder = tfds.builder_from_directory(str(Path(data_dir).resolve()))
    split = builder.info.splits["train"]
    root = Path(data_dir).resolve()
    shard_paths = [(Path(str(path)) if Path(str(path)).is_absolute()
                    else root / str(path)).resolve() for path in split.filenames]
    lengths = list(split.shard_lengths)
    if len(shard_paths) != len(lengths):
        raise ValueError("TFDS分片清单与episode数量不一致")
    offsets = {}
    offset = 0
    for path, length in zip(shard_paths, lengths):
        offsets[path] = (offset, int(length))
        offset += int(length)
    checked = 0
    read_episodes = 0
    checked_fields = ("action", "observation/state", "language_instruction",
                      "is_first", "is_last", "is_terminal", "observation/image_0")
    for path, indices in wanted.items():
        absolute = Path(path).resolve()
        if absolute not in offsets:
            raise ValueError(f"所选分片不属于TFDS train清单：{path}")
        base, length = offsets[absolute]
        lo, hi = min(indices), max(indices) + 1
        if hi > length:
            raise ValueError("记录索引超出TFDS元数据的分片长度")
        print(f"[TFDS对照] {absolute.name} records={lo}:{hi}，不读取整个数据集", flush=True)
        official = builder.as_dataset(split=f"train[{base + lo}:{base + hi}]",
            shuffle_files=False, decoders={"steps": tfds.decode.SkipDecoding()},
            read_config=tfds.ReadConfig(interleave_cycle_length=1, interleave_block_length=1))
        count = 0
        for count, episode in enumerate(tfds.as_numpy(official), start=1):
            read_episodes += 1
            number = lo + count - 1
            if number not in indices:
                continue
            fields = records[path, number]
            steps = episode["steps"]
            for name in checked_fields:
                value = steps
                for part in name.split("/"):
                    value = value[part]
                field = fields["steps/" + name]
                kind = field.WhichOneof("kind")
                raw = np.asarray(getattr(field, kind).value)
                expected = raw.reshape(value.shape)
                if not np.array_equal(value, expected):
                    raise ValueError(f"TFDS与手动解析不一致：{path} episode={number} field={name}")
            if not bool(steps["is_last"][-1]) or np.count_nonzero(steps["is_last"]) != 1:
                raise ValueError("RLDS末步标记无效，不能确认动作窗口边界")
            checked += 1
        if count != hi - lo:
            raise ValueError("TFDS实际读取的episode数量与请求范围不一致")
    return {"entry_point": "tfds.builder_from_directory().as_dataset()",
            "tfds_version": tfds.__version__, "dataset_name": builder.info.name,
            "dataset_version": str(builder.info.version), "split": "train",
            "checked_episodes": checked, "read_episodes": read_episodes,
            "checked_fields": list(checked_fields), "exact_raw_field_match": True,
            "jpeg_comparison": "encoded_bytes_exact; decoder_pixel_difference_reported_separately",
            "last_step_action": "excluded: reached target j uses command j-1, never terminal command",
            "physical_semantics_verified": False}


def plot_bridge_timeline(fields, output: Path, record_index: int) -> dict:
    """全episode原始时序，不调用项目位姿转换，不把步号当物理时间。

    原始夹爪命令与测量可能滞后；末步命令无效，仅将测量绘到末步。
    """
    import io
    images = fields["steps/observation/image_0"].bytes_list.value
    n = len(images)
    state = np.asarray(fields["steps/observation/state"].float_list.value).reshape(n, 7)
    commands = np.asarray(fields["steps/action"].float_list.value).reshape(n, 7)
    instruction = fields["steps/language_instruction"].bytes_list.value[0].decode("utf-8", errors="replace")
    if n < 2 or not np.isfinite(state).all() or not np.isfinite(commands).all():
        raise ValueError("时序图需要至少两步且全部状态/命令有限")
    # 只用于选择查看的帧，不改变训练标签。覆盖原始命令变化附近及首末步。
    transitions = (np.flatnonzero(np.abs(np.diff(commands[:-1, 6])) > 1e-6) + 1).tolist()
    frames = sorted(set([0, n - 1] + transitions + [max(0, t - 1) for t in transitions]))
    if len(frames) > 8:
        frames = [frames[i] for i in np.linspace(0, len(frames) - 1, 8, dtype=int)]
    for t in np.linspace(0, n - 1, 8, dtype=int):
        if len(frames) < min(8, n) and int(t) not in frames:
            frames.append(int(t))
    frames.sort()
    canvas = Image.new("RGB", (1480, 1080), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    draw.text((24, 15), f"Bridge raw episode {record_index}: {instruction[:140]}", fill="black", font=font)
    draw.text((24, 42), "Step indices, NOT seconds. Native state units / frame not independently verified. No model predictions.", fill="#555555", font=font)
    for col, t in enumerate(frames):
        x = 24 + col * 180
        rgb = Image.open(io.BytesIO(images[t])).convert("RGB")
        canvas.paste(rgb.resize((170, 170)), (x, 80))
        draw.text((x, 256), f"step {t}" + (" (last)" if t == n - 1 else ""), fill="black", font=font)
        label = "command invalid" if t == n - 1 else f"cmd={commands[t, 6]:.2f}"
        draw.text((x, 280), label, fill="#555555", font=font)
        draw.text((x, 303), f"obs={state[t, 6]:.2f}", fill="#555555", font=font)

    def chart(top, title, series, fixed=None, stairs=False):
        left, right, bottom = 115, 1440, top + 145
        values = np.concatenate([np.asarray(v) for v, _, _ in series])
        low, high = fixed or (float(values.min()), float(values.max()))
        if high - low < 1e-8:
            low, high = low - .01, high + .01
        pad = (high - low) * .08
        low, high = low - pad, high + pad
        draw.text((24, top - 30), title, fill="black", font=font)
        draw.rectangle((left, top, right, bottom), outline="#777777")
        for value in np.linspace(low, high, 4):
            y = bottom - (value - low) / (high - low) * (bottom - top)
            draw.line((left, y, right, y), fill="#eeeeee")
            draw.text((24, y - 8), f"{value:.3f}", fill="#555555", font=font)
        for t in frames:
            x = left + t / (n - 1) * (right - left)
            draw.line((x, top, x, bottom), fill="#dddddd")
            draw.text((x - 8, bottom + 5), str(t), fill="#555555", font=font)
        for j, (values, color, label) in enumerate(series):
            points = []
            for t, value in enumerate(values):
                x = left + t / (n - 1) * (right - left)
                y = bottom - (value - low) / (high - low) * (bottom - top)
                if stairs and points:
                    points.append((x, points[-1][1]))
                points.append((x, y))
            draw.line(points, fill=color, width=3)
            draw.text((left + j * 300, bottom + 28), label, fill=color, font=font)
    chart(360, "Observed XYZ minus first observation (native units)",
          [(state[:, i] - state[0, i], color, label) for i, color, label in
           ((0, "#c43131", "X"), (1, "#198547", "Y"), (2, "#344abd", "Z"))])
    chart(590, "Observed orientation columns 3:6 (raw values; wraps may occur)",
          [(state[:, i], color, label) for i, color, label in
           ((3, "#c43131", "column 3"), (4, "#198547", "column 4"), (5, "#344abd", "column 5"))])
    chart(820, "Raw gripper: command[t] vs observation[t] (not shifted); final command excluded",
          [(commands[:-1, 6], "#ba3c16", "raw command (0 closed, 1 open)"),
           (state[:, 6], "#2863a8", "observed opening")], fixed=(0, 1), stairs=True)
    draw.text((115, 1020), "Vertical guides match image step indices. Command != reached state; this is not a controller replay.", fill="#555555", font=font)
    name = f"episode_{record_index}_timeline.png"
    canvas.save(output / name)
    series_name = f"episode_{record_index}_raw_series.json"
    (output / series_name).write_text(json.dumps({"instruction": instruction,
        "step_indices": list(range(n)), "observed_state": state.tolist(),
        "valid_commands": commands[:-1].tolist(), "last_command_excluded": True,
        "timestamps_available": False}, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"image": name, "raw_series": series_name, "steps": n,
            "image_step_indices": frames, "raw_command_change_steps": transitions,
            "note": "原生数值/步号可视化，不独立验证单位、坐标、延迟或任务成功。"}


def audit_single_source(args: argparse.Namespace) -> None:
    """独立读取原始TFRecord，避免仅用同一套编码/解码函数自证正确。

    通过只代表抽样的实现一致性；单位、控制时序、真实机器人标定仍需验收。
    Bridge采用到达状态监督，BC-Z采用原生10命令，不能混为同一种标签。
    """
    import io
    import tensorflow as tf
    import pybullet as bullet
    from dataset import decode_relative_pose, rotation_vector_to_quaternion, euler_xyz_to_quaternion

    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError("输出目录非空，请指定新目录，保留原报告")
    native = args.single_source_check == "bcz"
    if (args.tfds_reference or args.bridge_timeline) and native:
        raise ValueError("TFDS对照/Bridge时序图目前仅支持Bridge单来源审计")
    schema = "tfrecord_bc_z_pose" if native else "tfrecord_bridge_state_action"
    dataset = UnifiedRobotDataset(args.dataset_dir, chunk_size=10 if native else args.chunk_size,
        stride=args.stride, sources=("tfrecord",), max_samples_per_schema=args.max_samples_per_schema,
        max_tfrecord_episodes_per_schema=args.max_tfrecord_episodes_per_schema or 4,
        min_trajectory_steps=args.min_trajectory_steps,
        tfrecord_splits=("train",), bcz_target="native_commands" if native else "reached",
        bcz_current_gripper="continuous" if native else "binary",
        bridge_gripper_policy=args.bridge_gripper_policy)
    if set(dataset.source_counts) != {schema}:
        raise ValueError(f"必须单独提供{schema}目录，当前发现{dict(dataset.source_counts)}")
    if args.samples_per_schema < 1:
        raise ValueError("抽样数量必须大于0")
    output.mkdir(parents=True, exist_ok=True)
    # 优先覆盖独立episode，避免扩大窗口数量却反复检查同一条演示。
    episode_groups = defaultdict(list)
    for index in range(len(dataset)):
        episode_groups[dataset.group_key(index)].append(index)
    groups = list(episode_groups.values())
    chosen_groups = evenly_spaced(list(range(len(groups))), args.samples_per_schema)
    selected = [groups[g][len(groups[g]) // 2] for g in chosen_groups]
    if len(selected) < args.samples_per_schema:
        remaining = [i for i in range(len(dataset)) if i not in selected]
        selected.extend(evenly_spaced(remaining, args.samples_per_schema - len(selected)))
    wanted = defaultdict(set)
    for index in selected:
        sample = dataset.samples[index]
        wanted[sample["file_path"]].add(sample["record_index"])
    records = {}
    # 不调用加载器的按字节offset读取，独立顺序读取核对是否取错episode。
    for path, indices in wanted.items():
        for number, serialized in enumerate(tf.data.TFRecordDataset(path)):
            if number in indices:
                records[path, number] = tf.train.Example.FromString(bytes(serialized.numpy())).features.feature
            if number >= max(indices):
                break

    tfds_reference = compare_bridge_tfds(args.dataset_dir, wanted, records) if args.tfds_reference else None
    timelines = []
    if args.bridge_timeline:
        for (path, number), fields in records.items():
            # 文件编号防止不同分片内相同record_index覆盖图像。
            subdir = output / Path(path).name
            subdir.mkdir(exist_ok=True)
            timelines.append({"source_path": path, "record_index": number,
                "directory": subdir.name, **plot_bridge_timeline(fields, subdir, number)})

    def numbers(fields, key):
        field = fields[key]
        kind = field.WhichOneof("kind")
        if kind not in ("float_list", "int64_list"):
            raise ValueError(f"非数值字段：{key}")
        return np.asarray(getattr(field, kind).value, dtype=np.float64)

    def matrix(q):
        return np.asarray(bullet.getMatrixFromQuaternion(np.asarray(q).tolist())).reshape(3, 3)

    def rotvec_matrix(v):
        angle = float(np.linalg.norm(v))
        q = (0, 0, 0, 1) if angle < 1e-12 else bullet.getQuaternionFromAxisAngle((v / angle).tolist(), angle)
        return matrix(q)

    rows = []
    for index in selected:
        sample = dataset.samples[index]
        fields = records[sample["file_path"], sample["record_index"]]
        start = sample["start_index"]
        image_key = "steps/observation/image" if native else "steps/observation/image_0"
        text_key = "steps/observation/natural_language_instruction" if native else "steps/language_instruction"
        images = fields[image_key].bytes_list.value
        count = len(images)
        text, image_tensor, current, targets, mask = dataset[index]
        assert text == fields[text_key].bytes_list.value[start].decode("utf-8", errors="replace")
        if native:
            positions = numbers(fields, "steps/observation/present/xyz").reshape(count, 3)
            angles = numbers(fields, "steps/observation/present/axis_angle").reshape(count, 3)
            rotation = rotvec_matrix(angles[start])
            xyz = positions[start] + numbers(fields, "steps/action/future/xyz_residual").reshape(count, 10, 3)[start]
            target_angles = angles[start] + numbers(fields, "steps/action/future/axis_angle_residual").reshape(count, 10, 3)[start]
            rotations = np.stack([rotvec_matrix(v) for v in target_angles])
            closes = numbers(fields, "steps/action/future/target_close").reshape(count, 10)[start]
            grippers = 1 - 2 * closes
            observed = numbers(fields, "steps/observation/present/sensed_close")[start]
            expected_current = 1 - 2 * observed
            project_q = rotation_vector_to_quaternion(angles[start])
            intermediate = None
            official_gripper_difference = None
            official_difference_indices = []
            target_gripper_differences = None
            timing = None
        else:
            state = numbers(fields, "steps/observation/state").reshape(count, 7)
            command = numbers(fields, "steps/action").reshape(count, 7)
            flags = lambda key: numbers(fields, key) if key in fields else None
            timing = inspect_bridge_timing(state, command, flags("steps/is_first"), flags("steps/is_last"))
            camera_key = "episode_metadata/has_image_0"
            timing["has_image_0"] = bool(numbers(fields, camera_key)[0]) if camera_key in fields else None
            positions = state[:, :3]
            rotation = matrix(bullet.getQuaternionFromEuler(state[start, 3:6].tolist()))
            steps = np.minimum(np.arange(start + 1, start + 1 + dataset.chunk_size), count - 1)
            xyz = positions[steps]
            rotations = np.stack([matrix(bullet.getQuaternionFromEuler(v.tolist())) for v in state[steps, 3:6]])
            grippers = np.where(command[steps - 1, 6] >= .5, 1, -1)
            expected_current = 1 if state[start, 6] >= .5 else -1
            project_q = euler_xyz_to_quaternion(state[start, 3:6])
            intermediate = int(((command[:, 6] >= .05) & (command[:, 6] <= .95)).sum())
            # 对照官方向后扫描策略，仅报告差异，不悄悄改旧标签/权重语义。
            official = command[:, 6].copy()
            carry = official[-1]
            for t in reversed(range(count)):
                if official[t] > .95:
                    carry = 1.0
                elif official[t] < .05:
                    carry = 0.0
                official[t] = carry
            official_gripper_difference = int(np.count_nonzero((command[:, 6] >= .5) != (official >= .5)))
            official_difference_indices = np.flatnonzero((command[:, 6] >= .5) != (official >= .5)).tolist()
            target_gripper_differences = int(np.count_nonzero((grippers > 0) != (official[steps - 1] >= .5)))
            if args.bridge_gripper_policy == "reverse_scan_v1":
                if not np.isin(official, [0, 1]).all():
                    raise ValueError("独立官方参考包含末尾非二值，不能视为二值监督验收通过")
                grippers = 2 * official[steps - 1] - 1
        expected_xyz = (xyz - positions[start]) @ rotation / POSITION_SCALE_METERS
        expected_rotation = rotation.T @ rotations
        actual = targets.numpy()
        xyz_error = float(np.abs(actual[:, :3] - expected_xyz).max())
        rotation_error = float(np.abs(np.stack([matrix(q) for q in actual[:, 3:7]]) - expected_rotation).max())
        matrix_error = float(np.abs(matrix(project_q) - rotation).max())
        decoded = [decode_relative_pose(positions[start], project_q, action) for action in actual]
        world_error = float(np.abs(np.stack([p for p, q in decoded]) - xyz).max())
        world_rotation_error = float(np.abs(np.stack([matrix(q) for p, q in decoded]) - rotations).max())
        assert xyz_error < 2e-5 and rotation_error < 2e-5 and matrix_error < 2e-5
        assert world_error < 2e-5 and world_rotation_error < 2e-5
        assert np.array_equal(actual[:, 7], grippers) and abs(current.item() - expected_current) < 1e-6
        assert bool((mask == 1).all()) and bool(torch.isfinite(image_tensor).all())
        raw = np.asarray(Image.open(io.BytesIO(images[start])).convert("RGB"))
        if timing is not None:
            timing["input_image_all_zero"] = bool(np.all(raw == 0))
            timing["input_image_std"] = float(raw.std())
            timing["camera_flag_contradicts_nonzero_pixels"] = bool(timing["has_image_0"] is False and np.any(raw != 0))
        tf_rgb = tf.io.decode_image(images[start], channels=3, expand_animations=False).numpy()
        assert tf_rgb.shape == raw.shape
        # 用TensorFlow独立实现相同letterbox，核对通道/位置/缩放/归一化。
        height, width = raw.shape[:2]
        scale = min(224 / height, 224 / width)
        rh, rw = max(1, min(224, round(height * scale))), max(1, min(224, round(width * scale)))
        resized = tf.image.resize(raw.astype(np.float32) / 255, (rh, rw), method="bilinear").numpy()
        reference_image = np.broadcast_to(CLIP_IMAGE_MEAN.numpy().reshape(1, 1, 3), (224, 224, 3)).copy()
        top, left = (224 - rh) // 2, (224 - rw) // 2
        reference_image[top:top+rh, left:left+rw] = resized
        restored = (image_tensor * CLIP_IMAGE_STD + CLIP_IMAGE_MEAN).permute(1, 2, 0).numpy()
        image_error = float(np.abs(restored - reference_image).max())
        assert image_error < 2e-5
        prefix = f"sample_{index}_t{start}"
        for t in sorted({max(0, start - 1), start, min(count - 1, start + 1)}):
            Image.open(io.BytesIO(images[t])).convert("RGB").save(output / f"{prefix}_raw_t{t}.png")
        Image.fromarray(restore_rgb(image_tensor)).save(output / f"{prefix}_model_input.png")
        rows.append({**sample_identity(sample), "index": index, "instruction": text,
            "image_shape": list(raw.shape), "image_dtype": str(raw.dtype), "raw_pixel_max": int(raw.max()),
            "independent_resize_max_error": image_error, "pil_tf_decode_max_pixel_difference": int(np.abs(raw.astype(int) - tf_rgb.astype(int)).max()),
            "local_position_max_error": xyz_error, "local_rotation_matrix_max_error": rotation_error,
            "pybullet_matrix_max_error": matrix_error, "decoded_position_max_error_native_units": world_error,
            "decoded_rotation_matrix_max_error": world_rotation_error, "bridge_intermediate_gripper_commands": intermediate,
            "bridge_threshold_vs_official_scan_differences_episode": official_gripper_difference,
            "bridge_official_difference_command_indices": official_difference_indices,
            "bridge_gripper_label_difference_targets": target_gripper_differences,
            "bridge_timing": timing})
        print(f"[独立验收] {schema} sample={index} t={start} 图像/指令/动作对应、旋转库核对通过", flush=True)
    report = {"schema": schema, "sampled_windows": len(rows), "implementation_checks_passed": True,
        "tfds_reference": tfds_reference,
        "bridge_timelines": timelines,
        "bridge_gripper_policy": args.bridge_gripper_policy,
        "sampled_episodes": len({dataset.group_key(i) for i in selected}),
        "sampled_shards": len(wanted), "scanned_episodes": dict(dataset.tfrecord_episode_counts),
        "sampling": "deterministic_episode_first_then_fill_windows",
        "physical_semantics_fully_validated": False, "samples": rows,
        "limitations": ["只检查少量记录，不代表全数据；未训练或选择模型。",
            "只读原始TFRecord train split，未按checkpoint内部train/validation/test重新过滤；不能称仅访问项目训练分区。报告只用于实现/语义审计，不用于调参选模型。",
            "图像不重新JPEG压缩；224 letterbox可损失细节，PNG需人工确认方向/颜色/可辨识性。",
            "PyBullet独立轴角/Euler/旋转矩阵验证数学与xyzw接口，不验证机器人坐标标定、单位或控制周期。",
            "BC-Z原生残差不等于下一观测；不能把命令与下一帧不一致直接判错。",
            f"Bridge策略={args.bridge_gripper_policy}；旧阈值策略不等价于官方扫描，新扫描通过仅说明抽样二值标签一致。",
            "当前PyBullet任务不是数据采集环境，不能用其成功率证明该数据加载器正确或错误。"]}
    if not native:
        timings = [r["bridge_timing"] for r in rows]
        report["bridge_timing_summary"] = {
            "first_flags_valid": sum(t["is_first_valid"] is True for t in timings),
            "last_flags_valid": sum(t["is_last_valid"] is True for t in timings),
            "first_action_all_zero": sum(t["first_action_all_zero"] for t in timings),
            "last_action_all_zero": sum(t["last_action_all_zero"] for t in timings),
            "camera_flag_contradictions": sum(t["camera_flag_contradicts_nonzero_pixels"] for t in timings),
            "input_images_all_zero": sum(t["input_image_all_zero"] for t in timings),
            "timing_verified_by_matched_exporter": False,
            "note": "边界正确或t候选误差较低不独立证明物理时序；不能据has_image_0=false直接过滤实际有图像的演示。"}
    (output / "single_source_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[独立验收] 完成：{output.resolve()}；实现一致性通过不等于物理语义全部验收")


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.cliport_native_only:
        if arguments.single_source_check or arguments.bridge_task_plan or arguments.tfds_reference or arguments.bridge_timeline:
            raise ValueError("Native CLIPort audit cannot combine with Bridge/BC-Z modes")
        audit_cliport_native(arguments)
        sys.exit(0)
    if arguments.bridge_task_plan:
        if arguments.tfds_reference or arguments.bridge_timeline or arguments.single_source_check:
            raise ValueError("单任务计划必须单独运行，不能与窗口审计混用")
        prepare_bridge_task_plan(arguments)
        sys.exit(0)
    if (arguments.tfds_reference or arguments.bridge_timeline) and not arguments.single_source_check:
        raise ValueError("TFDS对照/Bridge时序图必须与--single-source-check bridge一起使用")
    if arguments.single_source_check:
        audit_single_source(arguments)
    else:
        run_audit(arguments)

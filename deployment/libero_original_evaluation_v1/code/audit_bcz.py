"""审计本地 RLDS BC-Z：不改标签、不训练，比较命令与实际到达状态。"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def values(features, key):
    if key not in features:
        return np.asarray([])
    feature = features[key]
    kind = feature.WhichOneof("kind")
    return np.asarray(getattr(feature, kind).value) if kind else np.asarray([])


def inspect_command_neighborhood(features, start, radius=4):
    """错误训练窗口的原始邻域；数值近邻不能当作目标时间标定。"""
    from dataset import rotation_vector_to_quaternion
    sensed = values(features, "steps/observation/present/sensed_close")
    n = len(sensed)
    if not 0 <= start < n or radius < 0:
        raise ValueError("错误窗口的输入步或邻域半径非法")
    positions = values(features, "steps/observation/present/xyz").reshape(n, 3)
    angles = values(features, "steps/observation/present/axis_angle").reshape(n, 3)
    residuals = values(features, "steps/action/future/xyz_residual").reshape(n, 10, 3)
    angular = values(features, "steps/action/future/axis_angle_residual").reshape(n, 10, 3)
    targets = values(features, "steps/action/future/target_close").reshape(n, 10)
    future_indices = np.arange(start, n)
    observed_q = np.stack([rotation_vector_to_quaternion(a) for a in angles[start:]])
    matches = []
    for k in range(10):
        goal = positions[start] + residuals[start, k]
        goal_q = rotation_vector_to_quaternion(angles[start] + angular[start, k])
        errors = np.linalg.norm(positions[start:] - goal, axis=-1)
        nearest = int(future_indices[np.argmin(errors)])
        compatible = (errors <= 1e-5) & (np.abs(observed_q @ goal_q) >= 1 - 1e-5)
        matched_indices = future_indices[compatible].tolist()
        matches.append({"waypoint_zero_based": k, "native_target_close": float(targets[start, k]),
                        "nearest_future_xyz_observation_index": nearest,
                        "nearest_xyz_error_native_units": float(errors.min()),
                        "pose_matching_future_observation_indices": matched_indices,
                        "sensed_close_at_pose_matches": sensed[matched_indices].tolist()})
    return {"input_observation_index": start,
            "target_feature_kind": features["steps/action/future/target_close"].WhichOneof("kind"),
            "neighboring_observations": [{"index": i, "sensed_close": float(sensed[i]),
                "native_target_closes": targets[i].tolist()} for i in range(max(0, start-radius), min(n, start+radius+1))],
            "waypoint_pose_matches": matches,
            "note": "观测索引不是秒数；静止或重复位姿可能产生多个匹配；匹配/不匹配均不能独自证明标签正确或错误"}


def inspect_gripper(features):
    """描述性阈值/偏移扫描：不能用最高一致率反推官方标定或改训练标签。"""
    sensed = values(features, "steps/observation/present/sensed_close")
    target_key = "steps/action/future/target_close"
    targets = values(features, target_key)
    if not sensed.size or targets.size != sensed.size * 10:
        raise ValueError("夹爪观测与10个目标长度不一致")
    if not np.isfinite(sensed).all() or not np.isfinite(targets).all():
        raise ValueError("夹爪存在非有限值")
    targets = targets.reshape(-1, 10)
    rows = []
    for threshold in (.25, .5, .75, .95, .999):
        for offset in (-3, -2, -1, 0, 1, 2, 3):
            # 正偏移比较 action[t] 与 sensed[t+offset]；不是假定控制延迟。
            start = max(0, -offset)
            stop = max(start, min(len(sensed), len(sensed) - offset))
            current_closed = sensed[start + offset:stop + offset] >= threshold
            for waypoint in range(10):
                goal_closed = targets[start:stop, waypoint] >= .5
                rows.append({"threshold": threshold, "offset": offset, "waypoint": waypoint,
                             "open_open": int((~current_closed & ~goal_closed).sum()),
                             "open_closed": int((~current_closed & goal_closed).sum()),
                             "closed_open": int((current_closed & ~goal_closed).sum()),
                             "closed_closed": int((current_closed & goal_closed).sum())})
    adjacent = sensed >= .5
    return {"steps": len(sensed), "target_feature_kind": features[target_key].WhichOneof("kind"),
            "sensed_quantiles": np.quantile(sensed, [0, .1, .25, .5, .75, .9, 1]).tolist(),
            "target_value_counts": dict(Counter(str(v) for v in targets.ravel().tolist())),
            "first_target_closed_sensed_range": [float(sensed[targets[:, 0] >= .5].min()),
                                                    float(sensed[targets[:, 0] >= .5].max())]
                if np.any(targets[:, 0] >= .5) else None,
            "observed_adjacent_open_closed": int((~adjacent[:-1] & adjacent[1:]).sum()),
            "observed_adjacent_closed_open": int((adjacent[:-1] & ~adjacent[1:]).sum()),
            "sensitivity_counts": rows}


def inspect_episode(features):
    """仅比较数值；offset 是待验证的对齐假设，不据此改训练标签。"""
    prefix = "steps/observation/"
    xyz = values(features, prefix + "present/xyz")
    close = values(features, prefix + "present/sensed_close")
    if xyz.size % 3 or xyz.size // 3 != close.size or close.size < 2:
        raise ValueError("位置/夹爪长度不一致或轨迹过短")
    xyz = xyz.reshape(-1, 3)
    n = len(xyz)
    angles = values(features, prefix + "present/axis_angle")
    if angles.size != n * 3:
        raise ValueError("轴角长度不一致")
    residual = values(features, "steps/action/future/xyz_residual")
    targets = values(features, "steps/action/future/target_close")
    if residual.size != n * 30 or targets.size != n * 10:
        raise ValueError("未来动作字段不是每步 10 个 waypoint")
    residual = residual.reshape(n, 10, 3)
    targets = targets.reshape(n, 10)
    from dataset import bcz_first_command_action, decode_relative_pose, rotation_vector_to_quaternion
    angle_commands = values(features, "steps/action/future/axis_angle_residual")
    if angle_commands.size != n * 30:
        raise ValueError("未来轴角命令长度不一致")
    angle_commands = angle_commands.reshape(n, 10, 3)
    angles = angles.reshape(n, 3)
    command_checks = []
    for start in sorted({0, n // 2, n - 1}):
        action = bcz_first_command_action(xyz[start], angles[start], residual[start, 0],
                                          angle_commands[start, 0], targets[start, 0])
        decoded_p, decoded_q = decode_relative_pose(xyz[start], rotation_vector_to_quaternion(angles[start]), action)
        expected_q = rotation_vector_to_quaternion(angles[start] + angle_commands[start, 0])
        command_checks.append({"position_roundtrip_error_native_units": float(np.linalg.norm(decoded_p - xyz[start] - residual[start, 0])),
                               "quaternion_agreement": float(abs(np.dot(decoded_q, expected_q))),
                               "gripper_matches_command": bool((action[7] < 0) == (targets[start, 0] == 1))})
    if not all(np.isfinite(a).all() for a in (xyz, close, angles, residual, targets)):
        raise ValueError("包含 NaN/Inf")
    comparisons = {}
    for offset in (0, 1, 2):
        count = n - offset
        actual_delta = xyz[offset:] - xyz[:count]
        comparisons[str(offset)] = {
            "count": count,
            "first_waypoint_vs_observed_delta_cm_sum": float(
                np.linalg.norm(residual[:count, 0] - actual_delta, axis=1).sum() * 100
            ),
            "command_vs_sensed_agreements": int(
                ((targets[:count, 0] >= 0.5) == (close[offset:] >= 0.5)).sum()
            ),
        }
    timestep_keys = [k for k in features if k.endswith("present/timestep_count")]
    time = values(features, timestep_keys[0]).reshape(-1) if timestep_keys else np.asarray([])
    success = values(features, prefix + "episode_success")
    autonomous = values(features, prefix + "present/autonomous")
    intervention = values(features, prefix + "present/intervention")
    identity = values(features, "episode_id")
    return {
        "steps": n,
        "first_command_roundtrip_checks": command_checks,
        "episode_id": bytes(identity[0]).decode("utf-8", errors="replace") if identity.size else None,
        "success_values": np.unique(success).tolist(),
        "sensed_close_quantiles": np.quantile(close, [0, .1, .5, .9, 1]).tolist(),
        "target_close_values": np.unique(targets).tolist(),
        "sensed_closed_rate": float((close >= .5).mean()),
        "target_closed_rate": float((targets[:, 0] >= .5).mean()),
        "observed_adjacent_translation_cm_max": float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).max() * 100),
        "timestep_field": timestep_keys[0] if timestep_keys else None,
        "timestep_gaps": int((np.diff(time) > 1).sum()) if time.size else None,
        "timestep_nonincreasing": int((np.diff(time) <= 0).sum()) if time.size else None,
        "autonomous_values": np.unique(autonomous).tolist(),
        "intervention_values": np.unique(intervention).tolist(),
        "autonomous_intervention_complement": bool(np.all(autonomous + intervention == 1)) if autonomous.size == intervention.size == n else None,
        "alignment_hypotheses": comparisons,
    }


def inspect_native_sequence(features):
    """所有输入步×原生10目标往返验证；不是仿真或物理时序验证。"""
    from dataset import (bcz_native_command_chunk, bcz_continuous_gripper_observation,
                         decode_relative_pose, rotation_vector_to_quaternion)
    sensed = values(features, "steps/observation/present/sensed_close")
    n = sensed.size
    xyz = values(features, "steps/observation/present/xyz").reshape(n, 3)
    angles = values(features, "steps/observation/present/axis_angle").reshape(n, 3)
    residuals = values(features, "steps/action/future/xyz_residual").reshape(n, 10, 3)
    angular = values(features, "steps/action/future/axis_angle_residual").reshape(n, 10, 3)
    targets = values(features, "steps/action/future/target_close").reshape(n, 10)
    continuous = bcz_continuous_gripper_observation(sensed)
    rows = [{"waypoint": k, "checked_targets": 0, "max_position_roundtrip_error_native_units": 0.,
             "min_quaternion_abs_dot": 1., "gripper_mismatches": 0} for k in range(10)]
    for start in range(n):
        actions = bcz_native_command_chunk(xyz[start], angles[start], residuals[start], angular[start], targets[start])
        reference_q = rotation_vector_to_quaternion(angles[start])
        for k, action in enumerate(actions):
            p, q = decode_relative_pose(xyz[start], reference_q, action)
            row = rows[k]
            row["checked_targets"] += 1
            row["max_position_roundtrip_error_native_units"] = max(row["max_position_roundtrip_error_native_units"],
                float(np.linalg.norm(p - (xyz[start] + residuals[start, k]))))
            expected_q = rotation_vector_to_quaternion(angles[start] + angular[start, k])
            row["min_quaternion_abs_dot"] = min(row["min_quaternion_abs_dot"], float(abs(np.dot(q, expected_q))))
            row["gripper_mismatches"] += int(action[7] != (1 if targets[start, k] == 0 else -1))
    current_binary = sensed >= .5
    return {"input_steps": n, "output_shape_per_input": [10, 8], "waypoints": rows,
            "continuous_input_encoded_range": [float(continuous.min()), float(continuous.max())],
            "continuous_input_roundtrip_max_error": float(np.max(np.abs((1. - continuous) / 2. - sensed))),
            "distinct_sensed_values_per_binary_bucket": {
                "below_0.5": int(np.unique(sensed[~current_binary]).size),
                "at_or_above_0.5": int(np.unique(sensed[current_binary]).size)},
            "target_switches_between_waypoints": int((targets[:, 1:] != targets[:, :-1]).sum()),
            "note": "数值一致性不证明未来目标时刻、夹爪宽度或控制器可执行性；连续输入只用于审计，默认加载器未改变"}


def run(args):
    import tensorflow as tf
    root = Path(args.dataset_dir).resolve()
    files = sorted(root.rglob("*.tfrecord*"))
    if not files:
        raise RuntimeError(f"未发现 TFRecord：{root}")
    selected = None
    if args.selection_manifest:
        manifest = json.loads(Path(args.selection_manifest).read_text(encoding="utf-8"))
        if manifest.get("purpose") != "training_fit_only_not_generalization":
            raise ValueError("只接受标记为训练拟合诊断的选样清单，避免误扫留出分区")
        selected = {}
        for item in manifest["selected_windows"]:
            path = str(Path(item["file_path"]).resolve())
            selected.setdefault(path, set()).add(int(item["record_index"]))
        available = {str(f) for f in files}
        if not selected or not set(selected).issubset(available):
            raise ValueError("清单为空或包含数据根目录之外的文件")
        files = [f for f in files if str(f) in selected]
    report = {
        "dataset_dir": str(root), "episodes_per_file_limit": args.episodes_per_file,
        "scope": "每个分片前若干 episode，结果不能代表完整数据集",
        "gripper_threshold": .5,
        "threshold_note": "0.5 是当前加载器采用的候选阈值，不是已验证的官方阈值",
        "files": [], "episodes": [], "duplicate_content": [], "repeated_episode_ids": [],
        "errors": [], "field_keys": [],
    }
    if selected is not None:
        report["scope"] = "固定训练拟合清单所涉及episode的全部步骤；不读取验证/测试episode进行指标诊断"
        report["selection_manifest"] = str(Path(args.selection_manifest).resolve())
    hashes, ids = {}, {}
    for file in files:
        scope_note = f"清单指定 {len(selected[str(file)])} 条轨迹" if selected is not None else f"最多 {args.episodes_per_file} 条轨迹"
        print(f"[BC-Z审计] {file.name}：{scope_note}", flush=True)
        count = 0
        limit = max(selected[str(file)]) + 1 if selected is not None else args.episodes_per_file
        for index, raw in enumerate(tf.data.TFRecordDataset([str(file)]).take(limit)):
            # 顺序容器可能经过未选记录，但不解析其字段、不计算指标。
            if selected is not None and index not in selected[str(file)]:
                continue
            example = tf.train.Example.FromString(bytes(raw.numpy()))
            features = example.features.feature
            if not report["field_keys"]:
                report["field_keys"] = sorted(features)
            where = {"file": str(file), "record_index": index}
            try:
                result = inspect_episode(features)
                result["gripper_diagnostic"] = inspect_gripper(features)
                if args.native_sequence_check:
                    result["native_sequence_diagnostic"] = inspect_native_sequence(features)
            except ValueError as error:
                report["errors"].append({**where, "reason": str(error)})
                continue
            # 哈希图像、语言、状态和动作，排除元数据/episode_id 的影响。
            digest = hashlib.sha256()
            for key in sorted(features):
                if key.startswith(("steps/observation/", "steps/action/")):
                    digest.update(key.encode())
                    digest.update(features[key].SerializeToString(deterministic=True))
            fingerprint = digest.hexdigest()
            if fingerprint in hashes:
                report["duplicate_content"].append({"first": hashes[fingerprint], "repeat": where})
            hashes[fingerprint] = where
            eid = result["episode_id"]
            if eid:
                if eid in ids:
                    report["repeated_episode_ids"].append({"episode_id": eid, "first": ids[eid], "repeat": where})
                ids[eid] = where
            report["episodes"].append({**where, **result})
            count += 1
        report["files"].append({"path": str(file), "valid_audited_episodes": count})
    if selected is not None and len(report["episodes"]) + len(report["errors"]) != sum(map(len, selected.values())):
        raise RuntimeError("未读全清单指定的episode，拒绝输出不完整诊断")
    sensitivity = {}
    for episode in report["episodes"]:
        for row in episode["gripper_diagnostic"]["sensitivity_counts"]:
            key = (row["threshold"], row["offset"], row["waypoint"])
            counts = sensitivity.setdefault(key, Counter())
            counts.update({k: row[k] for k in ("open_open", "open_closed", "closed_open", "closed_closed")})
    report["gripper_sensitivity_summary"] = [
        {"threshold": t, "offset": o, "waypoint": w, **counts}
        for (t, o, w), counts in sorted(sensitivity.items())]
    report["gripper_diagnostic_note"] = "阈值和±3步偏移是描述性假设，不是标定或控制延迟估计；整数字段不能证明转换过程发生截断。"
    if args.native_sequence_check:
        checks = [e["native_sequence_diagnostic"] for e in report["episodes"]]
        if not checks:
            raise RuntimeError("无有效原生序列检查，不能生成通过结论")
        report["native_sequence_summary"] = {
            "checked_input_steps": sum(c["input_steps"] for c in checks),
            "checked_8d_targets": sum(r["checked_targets"] for c in checks for r in c["waypoints"]),
            "max_position_roundtrip_error_native_units": max(r["max_position_roundtrip_error_native_units"] for c in checks for r in c["waypoints"]),
            "min_quaternion_abs_dot": min(r["min_quaternion_abs_dot"] for c in checks for r in c["waypoints"]),
            "gripper_mismatches": sum(r["gripper_mismatches"] for c in checks for r in c["waypoints"]),
            "continuous_input_roundtrip_max_error": max(c["continuous_input_roundtrip_max_error"] for c in checks),
            "target_switches_between_waypoints": sum(c["target_switches_between_waypoints"] for c in checks),
            "activation": "diagnostic_only_not_connected_to_training_or_control",
        }
    report["observed_adjacent_transitions"] = {
        key: sum(e["gripper_diagnostic"][key] for e in report["episodes"])
        for key in ("observed_adjacent_open_closed", "observed_adjacent_closed_open")}
    report["gripper_reference_contract"] = {
        "sensed_close": "continuous measured closure, nominal 0..1, documented observed range about 0.2..1; no official binary 0.5 calibration established",
        "target_close": "absolute desired closure, not a delta; local field is int64, raw source example supplied by user uses float32; conversion code not verified",
        "paper_timing": "Appendix C: adaptive future state selection using gripper change >0.01 or joint delta L2 >0.05; not a fixed adjacent-state interval",
        "reference_implementation": "OpenVLA selects first native XYZ/axis-angle residual and inverts first native target_close; does not replace target by sensed threshold",
        "version_caveat": "local metadata is BC-Z 1.0.0/google3 builder; public TFDS catalog/importer documents 0.1.0 and is not proof of local conversion internals",
        "sources": ["https://www.tensorflow.org/datasets/catalog/bc_z",
                    "https://arxiv.org/html/2202.02005#A3",
                    "https://github.com/openvla/openvla/blob/main/prismatic/vla/datasets/rlds/oxe/transforms.py",
                    "https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/robotics/rtx/rtx.py"]}
    aggregate = {}
    for offset in ("0", "1", "2"):
        entries = [e["alignment_hypotheses"][offset] for e in report["episodes"]]
        n = sum(e["count"] for e in entries)
        aggregate[offset] = {
            "steps": n,
            "position_error_cm": sum(e["first_waypoint_vs_observed_delta_cm_sum"] for e in entries) / n if n else None,
            "gripper_agreement": sum(e["command_vs_sensed_agreements"] for e in entries) / n if n else None,
        }
    report["alignment_hypotheses_summary"] = aggregate
    report["success_value_counts_by_episode"] = dict(Counter(str(e["success_values"]) for e in report["episodes"]))
    report["label_contract"] = {
        "current_training_default": "future_reached_pose_in_input_tool_frame",
        "first_command_option": "current_xyz_plus_first_xyz_residual; current_axis_angle_plus_first_axis_angle_residual; binary_target_close",
        "rotation_conversion": "add in axis-angle vector space, then convert absolute orientation to quaternion, then input-tool-relative quaternion",
        "physical_timing": "not independently verified for local export; do not equate first command to next observed state or 0.1s",
        "position_scale": "existing project convention treats positions as meters; roundtrip check itself uses native units",
        "sources": ["local features.json", "https://www.tensorflow.org/datasets/catalog/bc_z", "https://arxiv.org/html/2202.02005#S0.SS3"]}
    report["limitations"] = [
        "没有 timestep_count 时，不能判定转换前是否存在时间缺口。",
        "offset 比较不能证明未来 waypoint 的控制时间间隔，也不能证明坐标系正确。",
        "没有扫描的轨迹不在重复检测覆盖范围内。",
        "未自动筛除轨迹或更改已有 checkpoint 的数据划分。",
    ]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[BC-Z审计] 有效={len(report['episodes'])}，异常={len(report['errors'])}，重复内容={len(report['duplicate_content'])}")
    print(f"[BC-Z审计] 对齐假设：{aggregate}")
    print(f"[BC-Z审计] 报告：{output.resolve()}")


def export_saved_prediction_audit(dataset, diagnostic_path, output_dir, task_names):
    """复核已保存验证预测，不加载模型、不重采样、不改标签。"""
    from dataset import decode_relative_pose, rotation_vector_to_quaternion
    from audit_dataset import plot_sample
    report = json.loads(Path(diagnostic_path).read_text(encoding="utf-8"))["partitions"]["validation"]
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError("原始轨迹核对需使用新目录，不覆盖旧图")
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for order, task in enumerate(task_names):
        row = report["instructions"].index(task)  # 每种指定任务取固定验证顺序的第一条。
        index = report["indices"][row]
        instruction, image, gripper, target, mask = dataset[index]
        assert instruction == task and dataset.samples[index] == report["samples"][row]
        target_array = target.numpy()
        assert np.allclose(target_array, report["targets"][row], atol=1e-6, rtol=0)
        assert bool((mask > .5).all())
        prediction = np.asarray(report["predictions"]["original"][row], dtype=np.float32)
        sample = dataset.samples[index]
        feature = dataset._load_tfrecord_example(sample["file_path"], sample["record_index"]).features.feature
        n = values(feature, "steps/observation/present/sensed_close").size
        start = int(sample["start_index"])
        position = values(feature, "steps/observation/present/xyz").reshape(n, 3)[start]
        angle = values(feature, "steps/observation/present/axis_angle").reshape(n, 3)[start]
        reference_q = rotation_vector_to_quaternion(angle)
        residuals = values(feature, "steps/action/future/xyz_residual").reshape(n, 10, 3)[start]
        angular = values(feature, "steps/action/future/axis_angle_residual").reshape(n, 10, 3)[start]
        closes = values(feature, "steps/action/future/target_close").reshape(n, 10)[start]
        decoded = [decode_relative_pose(position, reference_q, action) for action in target_array]
        xyz_error = max(float(np.linalg.norm(p - (position + residuals[k]))) for k, (p, q) in enumerate(decoded))
        q_dot = min(float(abs(np.dot(q, rotation_vector_to_quaternion(angle + angular[k])))) for k, (p, q) in enumerate(decoded))
        assert xyz_error < 1e-5 and q_dot > 1 - 1e-5
        assert np.array_equal(target_array[:, 7] < 0, closes > .5)
        plot_sample(output / f"case_{order + 1:02d}.png", f"fixed validation index={index}", instruction,
                    image, float(gripper.item()), target_array, prediction[None])
        records.append({"index": index, "instruction": instruction, "sample": sample,
            "plot": f"case_{order + 1:02d}.png", "reference_position": position.tolist(),
            "reference_axis_angle": angle.tolist(), "raw_xyz_residual_m": residuals.tolist(),
            "raw_target_close": closes.tolist(), "target": target_array.tolist(), "prediction": prediction.tolist(),
            "max_roundtrip_position_error_m": xyz_error, "min_roundtrip_quaternion_abs_dot": q_dot,
            "position_error_cm": float((np.linalg.norm((prediction - target_array)[:, :3], axis=-1)*10).mean()),
            "target_motion_cm": float((np.linalg.norm(target_array[:, :3], axis=-1)*10).mean()),
            "predicted_motion_cm": float((np.linalg.norm(prediction[:, :3], axis=-1)*10).mean()),
            "target_open": (target_array[:, 7] >= 0).tolist(), "predicted_open": (prediction[:, 7] >= 0).tolist()})
    result = {"purpose": "saved_validation_prediction_vs_native_fields_not_execution",
              "selection": "first_fixed_validation_window_for_each_prespecified_task",
              "camera_extrinsics_available": False, "physical_waypoint_timing_verified": False,
              "records": records}
    (output / "raw_prediction_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[原始预测核对] 案例={len(records)}；报告与图像：{output.resolve()}")
    return result


def audit_rotation_consistency(checkpoint_path, output_path):
    """BC-Z固定探针或Bridge训练演示核对原始/推理四元数，不优化或改标签。"""
    import torch
    from transformers import CLIPTokenizer
    from dataset import UnifiedRobotDataset
    from models import RobotAdapterModel
    from train import collate_batch, tokenise, dataset_split_identity
    output = Path(output_path)
    if output.exists():
        raise ValueError("请使用新报告路径，不覆盖已有诊断")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    bridge_fit = (checkpoint.get("experiment_kind") == "training_fit_diagnostic"
        and bool(checkpoint.get("data_config", {}).get("bridge_episode_selection")))
    bridge = checkpoint.get("experiment_kind") == "bridge_single_task_offline_diagnostic" or bridge_fit
    if not bridge and (checkpoint.get("experiment_kind") != "training_fit_diagnostic" or not checkpoint.get("pool_fit_from")
            or checkpoint["config"]["model"]["decoder_type"] != "regression"
            or checkpoint["data_config"]["bcz_target"] != "native_commands"):
        raise ValueError("旋转核对仅接受源完整池已见探针的回归拟合诊断")
    if bridge and checkpoint["config"]["model"]["decoder_type"] != "regression":
        raise ValueError("Bridge旋转核对必须是固定单任务回归诊断")
    config = checkpoint["data_config"]
    dataset = UnifiedRobotDataset(data_dir=config["dataset_dir"], **{
        key: config[key] for key in ("chunk_size", "stride", "sources", "max_samples", "max_samples_per_schema",
            "max_tfrecord_episodes", "max_tfrecord_episodes_per_schema", "min_trajectory_steps",
            "exclude_path_parts", "exclude_schemas", "tfrecord_splits", "bcz_target", "bcz_current_gripper",
            "bridge_gripper_policy", "bridge_current_gripper", "bridge_episode_selection") if key in config})
    indices = checkpoint["split_indices"]["train"]
    independent_rows = []
    if bridge:
        import pybullet as bullet
        from collections import defaultdict
        from train import bridge_plan_splits
        planned = bridge_plan_splits(dataset)
        if dataset_split_identity(dataset) != checkpoint.get("dataset_identity"):
            raise ValueError("Bridge数据身份或固定演示分区不同")
        if bridge_fit:
            from train import select_bridge_fit_windows, select_bridge_gripper_candidates
            fit = json.loads((Path(checkpoint_path).parent / "overfit_manifest.json").read_text(encoding="utf-8"))
            candidate_report = checkpoint.get("run_arguments", {}).get("bridge_gripper_fit_report")
            if candidate_report:
                expected, _ = select_bridge_gripper_candidates(dataset, planned["train"], candidate_report)
                if (fit.get("selection_rule") != "audited_training_gripper_category_coverage_v1"
                        or fit.get("candidate_report_sha256") != hashlib.sha256(Path(candidate_report).read_bytes()).hexdigest()):
                    raise ValueError("夹爪覆盖拟合源候选报告已变化")
            else:
                expected = select_bridge_fit_windows(dataset, planned["train"], len(indices))
            if (indices != expected or fit.get("purpose") != "bridge_fixed_train_fit_only"
                    or fit.get("dataset_identity") != checkpoint["dataset_identity"]
                    or fit.get("selected_indices") != indices
                    or len(fit.get("selected_windows", [])) != len(indices)
                    or any(dataset.samples[i] != s for i, s in zip(indices, fit["selected_windows"]))
                    or checkpoint["split_indices"]["validation"] != planned["validation"]
                    or checkpoint["split_indices"]["test"] != planned["test"]):
                raise ValueError("Bridge拟合窗口/留出分区与固定计划不一致")
        elif checkpoint["split_indices"] != planned:
            raise ValueError("Bridge固定演示分区不同")
        groups = defaultdict(list)
        for index in indices:
            groups[dataset.group_key(index)].append(index)
        indices = [values[len(values) // 2] for values in groups.values()]
        # 每条训练演示取按索引排序的中间窗口，选择不依赖目标或误差。
        # 对照独立旋转库，只确认声明的Euler解释/坐标数学，不证明原始物理语义。
        matrix = lambda q: np.asarray(bullet.getMatrixFromQuaternion(q)).reshape(3, 3)
        for index in indices:
            sample = dataset.samples[index]
            fields = dataset._load_tfrecord_example(sample["file_path"], sample["record_index"]).features.feature
            state = np.asarray(fields["steps/observation/state"].float_list.value).reshape(-1, 7)
            start = sample["start_index"]
            reference = matrix(bullet.getQuaternionFromEuler(state[start, 3:6].tolist()))
            target = dataset[index][3].numpy()
            raw_eulers = state[start + 1:start + 1 + dataset.chunk_size, 3:6]
            expected = np.stack([reference.T @ matrix(bullet.getQuaternionFromEuler(e.tolist())) for e in raw_eulers])
            actual = np.stack([matrix(q.tolist()) for q in target[:, 3:7]])
            difference = float(np.abs(expected - actual).max())
            if difference > 2e-5:
                raise ValueError("Bridge原始Euler经独立旋转库核对与目标矩阵不同")
            independent_rows.append({"index": index, "sample": sample,
                "reference_raw_euler": state[start, 3:6].tolist(),
                "future_raw_eulers": raw_eulers.tolist(),
                "target_relative_quaternions_xyzw": target[:, 3:7].tolist(),
                "independent_target_matrix_max_difference": difference})
    else:
        source = torch.load(checkpoint["pool_fit_from"], map_location="cpu", weights_only=False)
        manifest = json.loads((Path(checkpoint["pool_fit_from"]).parent / "split_manifest.json").read_text(encoding="utf-8"))
        if dataset_split_identity(dataset) != manifest["dataset_identity"] or indices != source["fit_probe_indices"]:
            raise ValueError("数据身份或固定训练探针不同")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RobotAdapterModel(checkpoint["config"], cache_dir=checkpoint["run_arguments"]["cache_dir"]).to(device).eval()
    from train import trainable_state_dict
    if set(trainable_state_dict(model)) != set(checkpoint["trainable_state_dict"]):
        raise ValueError("可训练权重不完整")
    model.load_state_dict(checkpoint["trainable_state_dict"], strict=False)
    tokenizer = CLIPTokenizer.from_pretrained(checkpoint["config"]["model"]["name"], cache_dir=checkpoint["run_arguments"]["cache_dir"])
    raw_values, target_values, sampled_values = [], [], []
    with torch.no_grad():
        for start in range(0, len(indices), 4):
            texts, images, current, target, masks = collate_batch([dataset[i] for i in indices[start:start+4]])
            if not bool((masks > .5).all()):
                raise ValueError("旋转目标监督不完整")
            tokens = tokenise(tokenizer, texts, device)
            context = model.get_context_vector(images.to(device), tokens["input_ids"], tokens.get("attention_mask"))
            raw = model.regression_head(context).reshape(-1, dataset.chunk_size, 7)
            prediction = model.sample(context, current.to(device))
            raw_values.append(raw.cpu()); target_values.append(target); sampled_values.append(prediction.cpu())
    raw = torch.cat(raw_values); targets = torch.cat(target_values); sampled = torch.cat(sampled_values)
    if not all(bool(torch.isfinite(t).all()) for t in (raw, targets, sampled)):
        raise ValueError("输出或目标非有限")
    q = torch.nn.functional.normalize(raw[...,3:7], dim=-1)
    target_q = torch.nn.functional.normalize(targets[...,3:7], dim=-1)
    similarity = (q * target_q).sum(-1).abs().clamp(0,1)
    proxy = 1 - similarity
    degrees = 2 * torch.acos(similarity) * 180 / np.pi
    inference_similarity = (sampled[...,3:7] * target_q).sum(-1).abs().clamp(0,1)
    inference_degrees = 2 * torch.acos(inference_similarity) * 180 / np.pi
    # 仅对保存的原始输出求导，检查损失信号；不反传模型/不执行optimizer.step。
    output_variable = raw.clone().requires_grad_(True)
    q_var = torch.nn.functional.normalize(output_variable[...,3:7], dim=-1)
    rotation_loss = (1 - (q_var * target_q).sum(-1).abs().clamp(max=1)).mean()
    xyz_loss = (output_variable[...,:3] - targets[...,:3]).square().mean()
    rotation_grad = torch.autograd.grad(rotation_loss, output_variable, retain_graph=True)[0]
    xyz_grad = torch.autograd.grad(xyz_loss, output_variable)[0]
    norm = raw[...,3:7].norm(dim=-1)
    angle_bins = {}
    for label, low, high in (("0_to_5",0,5),("5_to_15",5,15),("15_to_30",15,30),("30_to_180",30,181)):
        mask = (degrees >= low) & (degrees < high)
        angle_bins[label] = {"targets": int(mask.sum()), "mean_proxy_loss": float(proxy[mask].mean()) if mask.any() else None}
    report = {"purpose": "rotation_training_inference_consistency_no_training", "checkpoint": str(checkpoint_path),
        "windows": len(indices), "targets": int(degrees.numel()), "indices": indices,
        "raw_quaternion_norm_min_mean_max": [float(norm.min()),float(norm.mean()),float(norm.max())],
        "identity_fallback_targets": int((norm <= 1e-6).sum()),
        "target_quaternion_norm_max_deviation": float((targets[...,3:7].norm(dim=-1)-1).abs().max()),
        "normalized_raw_vs_sample_quaternion_max_component_difference": float((q-sampled[...,3:7]).abs().max()),
        "raw_rotation_error_deg": float(degrees.mean()), "sample_rotation_error_deg": float(inference_degrees.mean()),
        "training_rotation_proxy_loss": float(rotation_loss.detach()), "training_xyz_loss": float(xyz_loss.detach()),
        "regression_rotation_weight": model.regression_rotation_weight,
        "weighted_training_rotation_loss": float(rotation_loss.detach()) * model.regression_rotation_weight,
        "raw_output_rotation_gradient_l2": float(rotation_grad.norm()), "raw_output_xyz_gradient_l2": float(xyz_grad.norm()),
        "finite_output_gradients": bool(torch.isfinite(rotation_grad).all() and torch.isfinite(xyz_grad).all()),
        "sign_flip_loss_max_difference": float(((1-((-q)*target_q).sum(-1).abs().clamp(max=1))-proxy).abs().max()),
        "angle_bins": angle_bins, "rotation_error_by_waypoint_deg": degrees.mean(0).tolist(),
        "source": "bridge" if bridge else "bcz", "checkpoint_epoch": checkpoint.get("epoch"),
        "sampling": "one_middle_window_per_training_episode" if bridge else "original_fixed_seen_probe",
        "validation_test_targets_inspected": False,
        "identity_rotation_baseline_deg": float((2 * torch.acos(target_q[..., 3].abs().clamp(0, 1)) * 180 / np.pi).mean()),
        "prediction_relative_rotation_magnitude_deg": float((2 * torch.acos(q[..., 3].abs().clamp(0, 1)) * 180 / np.pi).mean()),
        "bridge_independent_target_rows": independent_rows,
        "limitations": ["输出空间梯度不是Adapter参数梯度，位置/旋转量纲不同，范数不能直接作为最优权重依据。",
            "该代理损失为1-cos(theta/2)，与报告角度不是同一数值尺度；小损失不等于小角度。",
            "数值一致性不独立证明原生字段物理坐标/时间或轴角残差语义正确。"]}
    if bridge:
        for row, prediction, errors in zip(independent_rows, sampled[..., 3:7], degrees):
            row["predicted_relative_quaternions_xyzw"] = prediction.tolist()
            row["rotation_errors_deg"] = errors.tolist()
        report["limitations"].append(f"Bridge每演示1个中间窗口，共{len(indices)}窗口，与源完整池窗口指标不是同一范围；窗口内目标相互相关。")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    printable = {key: value for key, value in report.items() if key != "bridge_independent_target_rows"}
    print(json.dumps(printable, ensure_ascii=False, indent=2), flush=True)
    if bridge:
        print(f"[旋转核对] 原始Euler/目标/预测逐窗口明细保存在：{output.resolve()}；未训练", flush=True)
    return report


def audit_pool_generalization(checkpoint_path, output_path):
    """完整池只读诊断：全训练/验证目标分布＋固定探针误差；不读取测试标签。"""
    import torch
    import tensorflow as tf
    from collections import defaultdict
    from transformers import CLIPTokenizer
    from dataset import UnifiedRobotDataset, POSITION_SCALE_METERS
    from models import RobotAdapterModel
    from train import collate_batch, tokenise

    output = Path(output_path)
    if output.exists():
        raise ValueError("诊断报告已存在，请用新输出路径，避免覆盖")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["data_config"]
    if config.get("bcz_target") != "native_commands":
        raise ValueError("该诊断仅支持BC-Z原生10目标完整池")
    dataset = UnifiedRobotDataset(data_dir=config["dataset_dir"], **{
        key: config[key] for key in ("chunk_size", "stride", "sources", "max_samples", "max_samples_per_schema",
            "max_tfrecord_episodes", "max_tfrecord_episodes_per_schema", "min_trajectory_steps",
            "exclude_path_parts", "exclude_schemas", "tfrecord_splits", "bcz_target", "bcz_current_gripper") if key in config})
    if len(dataset) != checkpoint["dataset_size"]:
        raise ValueError("数据规模与checkpoint不同")
    manifest_path = Path(checkpoint_path).parent / "split_manifest.json"
    training_fit = checkpoint.get("experiment_kind") == "training_fit_diagnostic"
    if training_fit:
        if not checkpoint.get("pool_fit_from"):
            raise ValueError("这里只审计源完整池固定探针的继续拟合权重")
        manifest_path = Path(checkpoint["pool_fit_from"]).parent / "split_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    from train import dataset_split_identity
    if dataset_split_identity(dataset) != manifest["dataset_identity"]:
        raise ValueError("数据身份不同，不能复用探针")
    splits = checkpoint["split_indices"]
    lookup = defaultdict(lambda: defaultdict(list))
    for partition in ("train", "validation"):
        for index in splits[partition]:
            sample = dataset.samples[index]
            if sample["source"] != "tfrecord_bc_z_pose":
                raise ValueError("这里只支持BC-Z schema")
            lookup[sample["file_path"]][sample["record_index"]].append((partition, index, sample["start_index"]))
    metadata = {}
    for file_path, wanted in lookup.items():
        for record_index, serialized in enumerate(tf.data.TFRecordDataset(file_path)):
            if record_index not in wanted:
                continue  # 未选episode（含test）不解析字段/标签。
            feature = tf.train.Example.FromString(bytes(serialized.numpy())).features.feature
            texts = feature["steps/observation/natural_language_instruction"].bytes_list.value
            residual = values(feature, "steps/action/future/xyz_residual").reshape(len(texts), 10, 3)
            closes = values(feature, "steps/action/future/target_close").reshape(len(texts), 10)
            for partition, index, start in wanted[record_index]:
                metadata[index] = {"partition": partition,
                    "task": " ".join(texts[start].decode("utf-8", errors="replace").lower().split()),
                    "episode": dataset.group_key(index), "motion_cm": (np.linalg.norm(residual[start], axis=-1) * 100).tolist(),
                    "max_axis_cm": float(np.abs(residual[start]).max() * 100),
                    "open_count": int((closes[start] <= .5).sum())}
        print(f"[覆盖审计] 已读取训练/验证元数据窗口={len(metadata)}", flush=True)
    distributions = {}
    for partition in ("train", "validation"):
        rows = [metadata[i] for i in splits[partition]]
        motion = np.asarray([r["motion_cm"] for r in rows])
        tasks = Counter(r["task"] for r in rows)
        episodes = {task: len({r["episode"] for r in rows if r["task"] == task}) for task in tasks}
        distributions[partition] = {"windows": len(rows), "episodes": len({r["episode"] for r in rows}),
            "task_windows": dict(tasks), "task_episodes": episodes,
            "motion_cm_quantiles": dict(zip(("min", "p50", "p90", "p95", "p99", "max"), np.quantile(motion, [0,.5,.9,.95,.99,1]).tolist())),
            "mean_motion_cm": float(motion.mean()), "zero_target_fraction": float((motion <= 1e-5).mean()),
            "mean_motion_by_waypoint_cm": motion.mean(axis=0).tolist(),
            "windows_exceeding_30cm_axis": sum(r["max_axis_cm"] > 30 for r in rows),
            "open_fraction": sum(r["open_count"] for r in rows) / motion.size}
    train_tasks = set(distributions["train"]["task_windows"])
    unseen = sorted(set(distributions["validation"]["task_windows"]) - train_tasks)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RobotAdapterModel(checkpoint["config"], cache_dir=checkpoint["run_arguments"]["cache_dir"]).to(device)
    incompatible = model.load_state_dict(checkpoint["trainable_state_dict"], strict=False)
    trainable_names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    if incompatible.unexpected_keys or trainable_names.intersection(incompatible.missing_keys):
        raise ValueError("可训练权重未完整加载")
    model.eval()
    tokenizer = CLIPTokenizer.from_pretrained(checkpoint["config"]["model"]["name"], cache_dir=checkpoint["run_arguments"]["cache_dir"])
    probes = ({"train": splits["train"]} if training_fit else
              {"train": checkpoint["fit_probe_indices"], "validation": checkpoint["validation_probe_indices"]})
    errors = {}
    with torch.no_grad():
        for partition, indices in probes.items():
            if not indices or len(indices) != len(set(indices)) or not set(indices).issubset(splits[partition]):
                raise ValueError("固定探针为空、重复或越过分区")
            rows = []
            for start in range(0, len(indices), 4):
                chosen = indices[start:start+4]
                texts, images, grippers, targets, masks = collate_batch([dataset[i] for i in chosen])
                if not bool((masks > .5).all()) or not bool(torch.isfinite(targets).all()):
                    raise ValueError("原生探针监督不完整或非有限")
                tokens = tokenise(tokenizer, texts, device)
                context = model.get_context_vector(images.to(device), tokens["input_ids"], tokens.get("attention_mask"))
                predicted = model.sample(context, grippers.to(device)).cpu()
                position = torch.linalg.vector_norm(predicted[..., :3] - targets[..., :3], dim=-1) * POSITION_SCALE_METERS * 100
                q1 = torch.nn.functional.normalize(predicted[..., 3:7], dim=-1)
                q2 = torch.nn.functional.normalize(targets[..., 3:7], dim=-1)
                rotation = 2 * torch.acos((q1 * q2).sum(-1).abs().clamp(0, 1)) * 180 / np.pi
                for row, index in enumerate(chosen):
                    rows.append({"index": index, "task": metadata[index]["task"], "episode": metadata[index]["episode"],
                        "position_error_cm": position[row].tolist(), "rotation_error_deg": rotation[row].tolist(),
                        "target_motion_cm": metadata[index]["motion_cm"],
                        "gripper_correct": ((predicted[row,:,7] >= 0) == (targets[row,:,7] >= 0)).tolist(),
                        "target_open": (targets[row,:,7] >= 0).tolist(), "predicted_open": (predicted[row,:,7] >= 0).tolist()})
                print(f"[误差审计] {partition}固定探针={len(rows)}/{len(indices)}", flush=True)
            task_summary = {}
            for task in sorted({r["task"] for r in rows}):
                selected = [r for r in rows if r["task"] == task]
                task_summary[task] = {"windows": len(selected), "episodes": len({r["episode"] for r in selected}),
                    "position_error_cm": float(np.mean([r["position_error_cm"] for r in selected])),
                    "zero_motion_error_cm": float(np.mean([r["target_motion_cm"] for r in selected])),
                    "rotation_error_deg": float(np.mean([r["rotation_error_deg"] for r in selected])),
                    "gripper_accuracy": float(np.mean([r["gripper_correct"] for r in selected]))}
            errors[partition] = {"rows": rows, "tasks": task_summary,
                "position_error_by_waypoint_cm": np.mean([r["position_error_cm"] for r in rows], axis=0).tolist(),
                "target_motion_by_waypoint_cm": np.mean([r["target_motion_cm"] for r in rows], axis=0).tolist(),
                "rotation_error_by_waypoint_deg": np.mean([r["rotation_error_deg"] for r in rows], axis=0).tolist(),
                "gripper_accuracy_by_waypoint": np.mean([r["gripper_correct"] for r in rows], axis=0).tolist()}
    result = {"purpose": ("seen_training_fit_error_audit_not_generalization" if training_fit else
                          "fixed_pool_generalization_audit_no_training_no_test_targets"), "checkpoint": str(checkpoint_path),
        "epoch": checkpoint["epoch"], "distributions": distributions,
        "validation_tasks_absent_from_train": unseen,
        "validation_windows_with_absent_task": sum(metadata[i]["task"] in unseen for i in splits["validation"]),
        "errors": errors, "limitations": ["任务按规范化原始指令字符串匹配，不是语义任务分类。",
            "max_axis_cm及windows_exceeding_30cm_axis描述原生残差坐标轴，不等于工具相对坐标推理裁剪越界。",
            "全分区分布按窗口统计，长轨迹权重较大；误差仅覆盖保存的固定探针。",
            "waypoint是目标序号，没有物理时间标定，不能当成等间隔时间。",
            "任务探针样本少，不能据其均值确定可靠难度排名；不自动调参数或选模型。"]}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[审计完成] 报告：{output.resolve()}", flush=True)
    return result


def bridge_gripper_window_category(commands):
    """按窗口内真实命令分类，不把测量阈值代理当成命令切换。"""
    commands = np.asarray(commands)
    if commands.ndim != 1 or len(commands) < 2 or not np.isin(commands, [0., 1.]).all():
        raise ValueError("夹爪诊断要求完整的二值命令序列")
    changes = np.flatnonzero(commands[1:] != commands[:-1]) + 1
    if not len(changes):
        return "hold_open" if commands[0] == 1 else "hold_closed"
    if len(changes) > 1:
        return "multiple_switches"
    return "open_to_closed" if commands[changes[0]] == 0 else "closed_to_open"


def check_bridge_gripper_candidates(dataset, candidates, output_dir):
    """原始字段与加载输出对照；静态图片仅帮助人工观察，不能标定控制延迟。"""
    import io
    import torch
    from PIL import Image, ImageDraw, ImageFont
    from dataset import prepare_image

    output_dir = Path(output_dir)
    if output_dir.exists():
        raise ValueError("候选图片目录已存在，请使用新输出路径")
    output_dir.mkdir(parents=True)
    font_path = Path("C:/Windows/Fonts/arial.ttf")
    font = ImageFont.truetype(str(font_path), 16) if font_path.exists() else ImageFont.load_default()
    checks, seen_inputs = [], {}
    for category, rows in candidates.items():
        for row in rows:
            index, sample = row["index"], row["sample"]
            fields = dataset._load_tfrecord_example(sample["file_path"], sample["record_index"]).features.feature
            state = values(fields, "steps/observation/state").reshape(-1, 7)
            native_actions = values(fields, "steps/action").reshape(-1, 7)
            images = fields["steps/observation/image_0"].bytes_list.value
            start = sample["start_index"]
            text, image, current, target, mask = dataset[index]
            expected = torch.tensor(row["commands_open_positive"], dtype=target.dtype) * 2 - 1
            raw_image = Image.open(io.BytesIO(images[start])).convert("RGB")
            expected_image = prepare_image(np.asarray(raw_image))
            expected_current = 2 * float(state[start, 6]) - 1
            if (not all(bool(torch.isfinite(t).all()) for t in (image, current, target, mask))
                    or not bool((mask > .5).all()) or not torch.equal(target[:, 7], expected)
                    or abs(float(current.item()) - expected_current) > 1e-6
                    or not torch.equal(image, expected_image)
                    or text != fields["steps/language_instruction"].bytes_list.value[start].decode("utf-8", errors="replace")):
                raise ValueError(f"候选{index}图像/文本/夹爪输入或目标与原始字段不一致")
            signature = hashlib.sha256(text.encode("utf-8") + image.numpy().tobytes()
                                       + current.numpy().tobytes()).hexdigest()
            if signature in seen_inputs and not torch.equal(seen_inputs[signature], target[:, 7]):
                raise ValueError("精确相同夹爪输入对应不同命令序列；不静默删除冲突")
            seen_inputs[signature] = target[:, 7].clone()
            # 图像观测j与前一命令j-1并列，明确索引关系，不声称测量即刻执行命令。
            offsets = {0, dataset.chunk_size}
            offsets.update((4, 8, 12))
            for switch in row["switch_offsets_zero_based"]:
                offsets.update((switch, switch + 1))
            offsets = sorted(o for o in offsets if 0 <= o <= dataset.chunk_size)
            cols = len(offsets)
            sheet = Image.new("RGB", (cols * 240, 365), "white")
            draw = ImageDraw.Draw(sheet)
            draw.text((8, 7), f"TRAIN index={index} | {category} | command 1=open, 0=close", font=font, fill="black")
            draw.text((8, 30), "Observation indices, NOT seconds. Sensor reading is NOT previous command.", font=font, fill="black")
            for col, offset in enumerate(offsets):
                j = start + offset
                frame = Image.open(io.BytesIO(images[j])).convert("RGB")
                frame.thumbnail((224, 224))
                x = col * 240 + 8
                sheet.paste(frame, (x, 62))
                draw.text((x, 290), f"obs[{j}] sensed={state[j,6]:.3f}", font=font, fill="black")
                prior_command = "outside window" if offset == 0 else str(int(row["commands_open_positive"][offset-1]))
                draw.text((x, 312), f"cmd[{j-1}]={prior_command}", font=font, fill="black")
            draw.text((8, 342), "window commands: " + " ".join(str(int(c)) for c in row["commands_open_positive"]), font=font, fill="black")
            image_path = output_dir / f"{category}_index_{index}.png"
            sheet.save(image_path)
            checks.append({"index": index, "category": category, "sample": sample,
                "commands_open_positive": row["commands_open_positive"],
                "switch_offsets_zero_based": row["switch_offsets_zero_based"],
                "raw_commands": native_actions[start:start+dataset.chunk_size, 6].tolist(),
                "observed_openings": state[start:start+dataset.chunk_size+1, 6].tolist(),
                "input_current_gripper": float(current.item()),
                "raw_image_size": list(raw_image.size), "loader_image_matches_raw_preprocessing": True,
                "loader_gripper_matches_command_labels": True, "input_signature": signature,
                "contact_sheet": str(image_path.resolve())})
    return {"checked_candidates": len(checks), "checks": checks,
            "exact_input_conflicts": 0,
            "note": "核对实现/索引一致性，不独立证明原始命令物理含义或控制时序；类别选样非泛化评估"}


def audit_bridge_gripper_coverage(checkpoint_path, output_path, check_candidates=False):
    """训练分区命令覆盖检查；只准备诊断候选，不加载模型、不训练。"""
    import torch
    import tensorflow as tf
    from collections import defaultdict
    from dataset import UnifiedRobotDataset, bridge_reverse_scan_gripper
    from train import dataset_split_identity, bridge_plan_splits

    output = Path(output_path)
    if output.exists():
        raise ValueError("报告已存在，请使用新路径，不覆盖")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["data_config"]
    if not config.get("bridge_episode_selection") or config.get("bridge_gripper_policy") != "reverse_scan_valid_steps_v2":
        raise ValueError("仅支持已固定演示分区及有效步夹爪策略的Bridge诊断")
    keys = ("chunk_size", "stride", "sources", "max_samples", "max_samples_per_schema",
            "max_tfrecord_episodes", "max_tfrecord_episodes_per_schema", "min_trajectory_steps",
            "exclude_path_parts", "exclude_schemas", "tfrecord_splits", "bcz_target", "bcz_current_gripper",
            "bridge_gripper_policy", "bridge_current_gripper", "bridge_episode_selection")
    dataset = UnifiedRobotDataset(data_dir=config["dataset_dir"], **{k: config[k] for k in keys if k in config})
    if dataset_split_identity(dataset) != checkpoint["dataset_identity"]:
        raise ValueError("数据身份不同，禁止复用训练分区")
    splits = bridge_plan_splits(dataset)
    lookup = defaultdict(lambda: defaultdict(list))
    for index in splits["train"]:
        sample = dataset.samples[index]
        lookup[sample["file_path"]][sample["record_index"]].append(index)
    rows, unique_events = [], set()
    for file_path, records in lookup.items():
        for record_index, serialized in enumerate(tf.data.TFRecordDataset(file_path)):
            if record_index > max(records):
                break
            if record_index not in records:
                continue  # 留出演示不解析命令/测量；扫描索引本身不参与选样。
            features = tf.train.Example.FromString(bytes(serialized.numpy())).features.feature
            state = values(features, "steps/observation/state").reshape(-1, 7)
            action = values(features, "steps/action").reshape(-1, 7)
            n = len(state)
            first, last = values(features, "steps/is_first"), values(features, "steps/is_last")
            if len(action) != n or np.flatnonzero(first).tolist() != [0] or np.flatnonzero(last).tolist() != [n-1]:
                raise ValueError("训练演示字段长度或首末标记非法")
            commands = bridge_reverse_scan_gripper(action[:-1, 6])
            if not np.isin(commands, [0., 1.]).all() or not np.isfinite(state[:, 6]).all():
                raise ValueError("命令未解析为二值或测量非有限")
            for index in records[record_index]:
                sample = dataset.samples[index]
                start = sample["start_index"]
                target = commands[start:start + dataset.chunk_size]
                if len(target) != dataset.chunk_size:
                    raise ValueError("窗口不完整")
                category = bridge_gripper_window_category(target)
                changes = (np.flatnonzero(target[1:] != target[:-1]) + 1).tolist()
                for offset in changes:
                    unique_events.add((file_path, record_index, start + offset))
                rows.append({"index": index, "sample": sample, "category": category,
                    "commands_open_positive": target.tolist(), "switch_offsets_zero_based": changes,
                    "observed_opening": float(state[start, 6]), "open_targets": int(target.sum())})
        print(f"[Bridge夹爪覆盖] 已检查训练窗口={len(rows)}；未检查留出目标", flush=True)
    if sorted(r["index"] for r in rows) != sorted(splits["train"]):
        raise ValueError("训练窗口检查有遗漏或重复，不能发布覆盖报告")
    counts = Counter(row["category"] for row in rows)
    categories = ("hold_open", "hold_closed", "open_to_closed", "closed_to_open", "multiple_switches")
    candidates = {}
    for category in categories:
        seen_episodes, chosen = set(), []
        for row in sorted(rows, key=lambda r: r["index"]):
            episode = dataset.group_key(row["index"])
            if row["category"] == category and episode not in seen_episodes:
                chosen.append(row)
                seen_episodes.add(episode)
                if len(chosen) == 4:
                    break
        candidates[category] = chosen
    report = {"purpose": "bridge_training_gripper_coverage_candidates_not_training_manifest",
        "dataset_identity": dataset_split_identity(dataset), "checked_training_windows": len(rows),
        "checked_training_episodes": len({dataset.group_key(r["index"]) for r in rows}),
        "category_window_counts": {c: counts[c] for c in categories},
        "unique_command_switches_within_windows": len(unique_events),
        "open_targets": sum(r["open_targets"] for r in rows),
        "target_count": len(rows)*dataset.chunk_size, "candidates": candidates,
        "selection_rule": "per_category_first_index_per_training_episode_up_to_four",
        "validation_test_targets_inspected": False, "model_loaded": False, "trained": False,
        "limitations": ["按训练标签选样，只用于分类覆盖诊断，不代表自然分布或泛化。",
            "重叠窗口可重复目标；去重命令切换也不代表独立物理开闭事件。",
            "窗口首个命令与前一命令之间的切换不计入窗口内切换。",
            "候选不是训练清单；尚未检查图像/输入冲突或拟合能力。"]}
    if check_candidates:
        report["candidate_input_checks"] = check_bridge_gripper_candidates(
            dataset, candidates, output.parent / (output.stem + "_images"))
        report["limitations"][-1] = "候选输入/标签数值对应已检查，但不是训练清单，尚未训练或验证泛化。"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in {"candidates", "candidate_input_checks"}}, ensure_ascii=False, indent=2))
    if check_candidates:
        print(f"[候选核对] 通过{report['candidate_input_checks']['checked_candidates']}个训练窗口；图像、输入、命令一致，无精确输入冲突。报告：{output.resolve()}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--dataset-dir", default=str(root / "training_cache/oxe_core/bc_z"))
    parser.add_argument("--episodes-per-file", type=int, default=16)
    parser.add_argument("--selection-manifest", help="仅检查固定训练拟合清单涉及的episode，不扩大到留出分区")
    parser.add_argument("--native-sequence-check", action="store_true", help="逐步检查全部10个原生8维目标及连续夹爪观测；不训练、不改标签")
    parser.add_argument("--output", default=str(root / "results/bcz_semantics_audit/summary.json"))
    parser.add_argument("--pool-checkpoint", help="只读完整池任务覆盖/固定探针误差诊断；不训练、不读取test目标")
    parser.add_argument("--rotation-checkpoint", help="只读BC-Z固定探针或Bridge单任务训练演示回归四元数的一致性核对")
    parser.add_argument("--bridge-gripper-checkpoint", help="只读Bridge完整训练分区命令开闭/切换覆盖；不训练")
    parser.add_argument("--bridge-gripper-candidates", action="store_true", help="覆盖检查时额外核对候选图像/输入/命令并输出对照图；不训练")
    args = parser.parse_args()
    if args.episodes_per_file < 1:
        parser.error("--episodes-per-file 必须大于 0")
    if sum(bool(v) for v in (args.pool_checkpoint, args.rotation_checkpoint, args.bridge_gripper_checkpoint)) > 1:
        parser.error("checkpoint审计入口不能同时使用")
    if args.bridge_gripper_candidates and not args.bridge_gripper_checkpoint:
        parser.error("候选核对必须与--bridge-gripper-checkpoint一起使用")
    if args.bridge_gripper_checkpoint:
        audit_bridge_gripper_coverage(args.bridge_gripper_checkpoint, args.output, args.bridge_gripper_candidates)
    elif args.rotation_checkpoint:
        audit_rotation_consistency(args.rotation_checkpoint, args.output)
    elif args.pool_checkpoint:
        audit_pool_generalization(args.pool_checkpoint, args.output)
    else:
        run(args)

"""Schema-driven loaders for a common 8-D robot action representation.

Every action is an end-effector target relative to the pose at the input image:
``[local_dx, local_dy, local_dz, dqx, dqy, dqz, dqw, gripper]``. Translation
is expressed in the initial tool frame and divided by ``POSITION_SCALE_METERS``.
Quaternions use ``xyzw`` ordering; the gripper convention is ``-1 = closed``
and ``+1 = open``. This removes incompatible world-coordinate origins from
different robots while preserving the requested 3+4+1=8 dimensions.

Files are discovered recursively below one root. Their parser is selected from
internal fields and tensor shapes, never from dataset or directory names.
"""

from __future__ import annotations

import glob
import io
import json
import os
import pickle
import random
import struct
import sys
from fnmatch import fnmatch
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


if __name__ == "__main__" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

try:
    import h5py
except ImportError:  # reported only when LIBERO is requested
    h5py = None


ACTION_DIM = 8
QUATERNION_SLICE = slice(3, 7)
GRIPPER_INDEX = 7
ACTION_REPRESENTATION = "tool_relative_pose_open_positive_v2"
POSITION_SCALE_METERS = 0.10
CLIP_IMAGE_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
CLIP_IMAGE_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
_TFRECORD_OFFSETS: Dict[str, List[int]] = {}


def normalise_quaternion(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float32).reshape(-1)
    if len(quaternion) != 4:
        raise ValueError(f"Expected four quaternion values, received {quaternion.shape}")
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-8:
        quaternion = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    else:
        quaternion = quaternion / norm
    # q and -q encode the same rotation. Fixing the hemisphere removes a
    # discontinuity from the regression target.
    if quaternion[3] < 0:
        quaternion = -quaternion
    return quaternion


def quaternion_conjugate(quaternion: np.ndarray) -> np.ndarray:
    quaternion = normalise_quaternion(quaternion)
    return np.asarray(
        [-quaternion[0], -quaternion[1], -quaternion[2], quaternion[3]],
        dtype=np.float32,
    )


def quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Hamilton product for two ``xyzw`` quaternions."""
    x1, y1, z1, w1 = normalise_quaternion(left)
    x2, y2, z2, w2 = normalise_quaternion(right)
    return normalise_quaternion(
        np.asarray(
            [
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            ],
            dtype=np.float32,
        )
    )


def rotate_vector(vector: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
    """Rotate a 3-D vector by an ``xyzw`` quaternion."""
    vector = np.asarray(vector, dtype=np.float32).reshape(3)
    quaternion = normalise_quaternion(quaternion)
    xyz = quaternion[:3]
    scalar = float(quaternion[3])
    return (
        vector
        + 2.0 * scalar * np.cross(xyz, vector)
        + 2.0 * np.cross(xyz, np.cross(xyz, vector))
    ).astype(np.float32)


def relative_pose_action(
    reference_position: np.ndarray,
    reference_quaternion: np.ndarray,
    target_position: np.ndarray,
    target_quaternion: np.ndarray,
    gripper: float,
) -> np.ndarray:
    """Encode a target pose in the input timestep's end-effector frame."""
    reference_position = np.asarray(reference_position, dtype=np.float32).reshape(3)
    target_position = np.asarray(target_position, dtype=np.float32).reshape(3)
    reference_quaternion = normalise_quaternion(reference_quaternion)
    target_quaternion = normalise_quaternion(target_quaternion)
    inverse_reference = quaternion_conjugate(reference_quaternion)
    local_delta = rotate_vector(target_position - reference_position, inverse_reference)
    local_delta = local_delta / POSITION_SCALE_METERS
    rotation_delta = quaternion_multiply(inverse_reference, target_quaternion)
    return np.concatenate([local_delta, rotation_delta, [gripper]]).astype(np.float32)


def bcz_first_command_action(position, axis_angle, xyz_residual, axis_angle_residual, target_close):
    """BC-Z 字段定义：残差分别加到当前位置/轴角，不是四元数相乘。

    只转换第一个原生控制目标，不假定其等于下一帧或具有固定秒数。
    空间输出仍为当前末端坐标系下的3+4+1维。
    """
    position = np.asarray(position, dtype=np.float32).reshape(3)
    axis_angle = np.asarray(axis_angle, dtype=np.float32).reshape(3)
    xyz_residual = np.asarray(xyz_residual, dtype=np.float32).reshape(3)
    angle_residual = np.asarray(axis_angle_residual, dtype=np.float32).reshape(3)
    if not all(np.isfinite(a).all() for a in (position, axis_angle, xyz_residual, angle_residual)):
        raise ValueError("BC-Z command contains nonfinite pose values")
    if float(target_close) not in (0.0, 1.0):
        raise ValueError("BC-Z first-command diagnostic requires binary target_close")
    return relative_pose_action(position, rotation_vector_to_quaternion(axis_angle),
        position + xyz_residual, rotation_vector_to_quaternion(axis_angle + angle_residual),
        1.0 if float(target_close) == 0 else -1.0)


def bcz_native_command_chunk(position, axis_angle, xyz_residuals, angle_residuals, target_closes):
    """诊断用原生10目标：[10, 8]，每一行的位姿与夹爪来自相同waypoint。

    全部残差相对于同一输入时刻，不做累加，不跨观测拼接，不推定每步秒数。
    仅接入显式native_commands拟合诊断，不改变默认训练或闭环的数据语义。
    """
    xyz = np.asarray(xyz_residuals, dtype=np.float32)
    angles = np.asarray(angle_residuals, dtype=np.float32)
    closes = np.asarray(target_closes)
    if xyz.shape != (10, 3) or angles.shape != (10, 3) or closes.shape != (10,):
        raise ValueError("BC-Z native diagnostic requires shapes (10,3), (10,3), (10,)")
    return np.stack([
        bcz_first_command_action(position, axis_angle, xyz[k], angles[k], closes[k])
        for k in range(10)
    ])


def bcz_continuous_gripper_observation(sensed_close):
    """保留连续观测的诊断编码：0闭合程度→+1，1闭合程度→-1。

    不代表实际夹爪宽度，不施加0.5开闭标定；越界拒绝，不静默裁剪。
    仅显式native_commands使用此输入，不替换二值动作target_close。
    """
    close = np.asarray(sensed_close, dtype=np.float32)
    if not np.isfinite(close).all() or np.any((close < 0) | (close > 1)):
        raise ValueError("BC-Z sensed_close must be finite and within [0,1]")
    return 1.0 - 2.0 * close


def bridge_reverse_scan_gripper(commands):
    """Bridge官方向后扫描：中间值跟随后续端点，末尾中间值原样保留。

    返回0..1原生值，不能将未解决的末尾中间值谎称官方二值标签。
    """
    values = np.asarray(commands, dtype=np.float32)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("Bridge gripper commands must be a finite nonempty vector in [0,1]")
    result = np.empty_like(values)
    carry = values[-1]
    for index in reversed(range(len(values))):
        if values[index] > .95:
            carry = 1.0
        elif values[index] < .05:
            carry = 0.0
        result[index] = carry
    return result


def decode_relative_pose(
    reference_position: np.ndarray,
    reference_quaternion: np.ndarray,
    action: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert one normalized relative action back to an absolute pose."""
    action = np.asarray(action, dtype=np.float32).reshape(ACTION_DIM)
    reference_position = np.asarray(reference_position, dtype=np.float32).reshape(3)
    reference_quaternion = normalise_quaternion(reference_quaternion)
    world_delta = rotate_vector(
        action[:3] * POSITION_SCALE_METERS, reference_quaternion
    )
    target_position = reference_position + world_delta
    target_quaternion = quaternion_multiply(
        reference_quaternion, normalise_quaternion(action[QUATERNION_SLICE])
    )
    return target_position, target_quaternion


def rotation_vector_to_quaternion(rotation_vector: np.ndarray) -> np.ndarray:
    """Convert an axis-angle rotation vector to an ``xyzw`` quaternion."""
    rotation_vector = np.asarray(rotation_vector, dtype=np.float32).reshape(-1)
    angle = float(np.linalg.norm(rotation_vector))
    if angle < 1e-8:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    axis = rotation_vector / angle
    half_angle = 0.5 * angle
    return normalise_quaternion(
        np.concatenate([axis * np.sin(half_angle), [np.cos(half_angle)]])
    )


def euler_xyz_to_quaternion(euler: np.ndarray) -> np.ndarray:
    """Convert XYZ roll-pitch-yaw angles to an ``xyzw`` quaternion."""
    roll, pitch, yaw = np.asarray(euler, dtype=np.float32).reshape(3)
    cr, sr = np.cos(roll / 2.0), np.sin(roll / 2.0)
    cp, sp = np.cos(pitch / 2.0), np.sin(pitch / 2.0)
    cy, sy = np.cos(yaw / 2.0), np.sin(yaw / 2.0)
    return normalise_quaternion(
        np.asarray(
            [
                sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy,
                cr * cp * sy - sr * sp * cy,
                cr * cp * cy + sr * sp * sy,
            ],
            dtype=np.float32,
        )
    )


def prepare_image(image: np.ndarray) -> torch.Tensor:
    """Letterbox an HWC RGB/RGBA image and CLIP-normalise it to 224x224."""
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] not in (3, 4):
        raise ValueError(f"Expected an HWC RGB/RGBA image, received {image.shape}")
    rgb = np.array(image[..., :3], copy=True, order="C")
    if not rgb.size or not np.isfinite(rgb).all():
        raise ValueError("Image must be nonempty and finite")
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float()
    # uint8 的 1 是 1/255，不是浮点图中的满亮度1；不能只看最大值猜类型。
    if rgb.dtype == np.uint8:
        tensor = tensor / 255.0
    elif np.issubdtype(rgb.dtype, np.floating):
        if tensor.min().item() < 0 or tensor.max().item() > 255:
            raise ValueError("Floating image values must lie in [0,1] or [0,255]")
        if tensor.max().item() > 1.0:
            tensor = tensor / 255.0
    else:
        raise ValueError(f"Unsupported image dtype: {rgb.dtype}; convert explicitly to uint8 or float")
    height, width = tensor.shape[-2:]
    scale = min(224.0 / height, 224.0 / width)
    resized_height = max(1, min(224, round(height * scale)))
    resized_width = max(1, min(224, round(width * scale)))
    resized = F.interpolate(
        tensor.unsqueeze(0),
        size=(resized_height, resized_width),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    # Padding with CLIP's mean becomes zero after normalisation and preserves
    # the original aspect ratio instead of stretching rectangular images.
    canvas = CLIP_IMAGE_MEAN.expand(3, 224, 224).clone()
    top = (224 - resized_height) // 2
    left = (224 - resized_width) // 2
    canvas[:, top : top + resized_height, left : left + resized_width] = resized
    return (canvas - CLIP_IMAGE_MEAN) / CLIP_IMAGE_STD


def pad_action_chunk(actions: Sequence[np.ndarray], chunk_size: int) -> torch.Tensor:
    """Pad a short target sequence by holding its final valid target."""
    if not actions:
        raise ValueError("Cannot create an action chunk from an empty sequence")
    array = np.asarray(actions, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != ACTION_DIM:
        raise ValueError(f"Expected [T, {ACTION_DIM}] actions, received {array.shape}")
    if len(array) < chunk_size:
        array = np.concatenate(
            [array, np.repeat(array[-1:], chunk_size - len(array), axis=0)], axis=0
        )
    array = array[:chunk_size]
    for index in range(len(array)):
        array[index, QUATERNION_SLICE] = normalise_quaternion(
            array[index, QUATERNION_SLICE]
        )
    if not np.isfinite(array).all():
        raise ValueError("Action chunk contains NaN or infinite values")
    return torch.from_numpy(array.copy()).float()


class UnifiedRobotDataset(
    Dataset[Tuple[str, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]
):
    """Load selected sources into one checked 8-D action contract.

    Files are discovered recursively from one root. Parsers are selected from
    their internal schema rather than dataset or directory names. Unknown
    schemas are reported and skipped instead of being interpreted incorrectly.
    """

    VALID_SOURCES = {"auto", "libero", "cliport", "tfrecord"}

    def __init__(
        self,
        data_dir: str | Path,
        chunk_size: int = 16,
        stride: int = 4,
        sources: Sequence[str] = ("auto",),
        max_samples: int | None = None,
        max_samples_per_schema: int | None = None,
        max_tfrecord_episodes: int | None = None,
        max_tfrecord_episodes_per_schema: int | None = None,
        min_trajectory_steps: int = 10,
        exclude_path_parts: Sequence[str] = (),
        exclude_schemas: Sequence[str] = (),
        tfrecord_splits: Sequence[str] = (),
        bcz_target: str = "reached",
        bcz_current_gripper: str = "binary",
        bridge_gripper_policy: str = "threshold_v1",
        bridge_current_gripper: str = "binary",
        bridge_episode_selection: Sequence[dict] | None = None,
        bridge_window_horizon: int | None = None,
    ) -> None:
        if chunk_size <= 0 or stride <= 0 or min_trajectory_steps < 2:
            raise ValueError(
                "chunk_size/stride must be positive and min_trajectory_steps >= 2"
            )
        self.data_dir = Path(data_dir)
        self.chunk_size = chunk_size
        self.stride = stride
        if bcz_target not in {"reached", "first_command", "native_commands"}:
            raise ValueError(f"Unknown BC-Z target: {bcz_target}")
        if bcz_target == "first_command" and chunk_size != 1:
            raise ValueError("BC-Z first_command is a one-target diagnostic; set chunk_size=1")
        self.bcz_target = bcz_target
        if bcz_current_gripper not in {"binary", "continuous"}:
            raise ValueError("Unknown BC-Z current gripper encoding")
        if bcz_target == "native_commands" and (chunk_size != 10 or bcz_current_gripper != "continuous"):
            raise ValueError("原生序列拟合诊断必须使用10目标与连续夹爪观测")
        if bcz_current_gripper == "continuous" and bcz_target != "native_commands":
            raise ValueError("连续观测当前仅开放给隔离的native_commands诊断")
        self.bcz_current_gripper = bcz_current_gripper
        if bridge_gripper_policy not in {"threshold_v1", "reverse_scan_v1", "reverse_scan_valid_steps_v2"}:
            raise ValueError(f"Unknown Bridge gripper policy: {bridge_gripper_policy}")
        self.bridge_gripper_policy = bridge_gripper_policy
        if bridge_current_gripper not in {"binary", "continuous"}:
            raise ValueError("Unknown Bridge measured gripper encoding")
        self.bridge_current_gripper = bridge_current_gripper
        self.bridge_episode_selection = list(bridge_episode_selection or [])
        if bridge_window_horizon is not None and (not self.bridge_episode_selection
                or type(bridge_window_horizon) is not int or bridge_window_horizon < chunk_size):
            raise ValueError("bridge_window_horizon requires a fixed Bridge plan and must cover the output chunk")
        # Preserve the original window starts when testing a shorter prediction horizon.
        self.bridge_window_horizon = bridge_window_horizon
        self._bridge_selected = {}
        origin_ids = set()
        for row in self.bridge_episode_selection:
            key = (row["shard"], row["record_index"])
            origin = (row["origin_file_path"], row["episode_id"])
            if (Path(row["shard"]).name != row["shard"] or type(row["record_index"]) is not int
                    or row["record_index"] < 0 or row["partition"] not in {"train", "validation", "test"}
                    or key in self._bridge_selected or origin in origin_ids):
                raise ValueError("Bridge固定计划包含非法记录、重复来源或分区")
            self._bridge_selected[key] = row
            origin_ids.add(origin)
        if self.bridge_episode_selection and any(x is not None for x in
                (max_samples, max_samples_per_schema, max_tfrecord_episodes, max_tfrecord_episodes_per_schema)):
            raise ValueError("固定Bridge计划不允许截断或重采样，否则分区可能不完整")
        self.sources = tuple(source.lower() for source in sources)
        unknown = set(self.sources) - self.VALID_SOURCES
        if unknown:
            raise ValueError(f"Unknown dataset sources: {sorted(unknown)}")
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Dataset directory does not exist: {self.data_dir}")

        self.max_samples = max_samples
        self.max_samples_per_schema = max_samples_per_schema
        self.max_tfrecord_episodes = max_tfrecord_episodes
        self.max_tfrecord_episodes_per_schema = max_tfrecord_episodes_per_schema
        self.min_trajectory_steps = min_trajectory_steps
        self.exclude_path_parts = tuple(
            part.casefold() for part in exclude_path_parts if str(part).strip()
        )
        self.exclude_schemas = frozenset(
            str(schema).strip() for schema in exclude_schemas if str(schema).strip()
        )
        self.tfrecord_splits = frozenset(
            str(split).strip().casefold()
            for split in tfrecord_splits
            if str(split).strip()
        )
        self.samples: List[Dict[str, Any]] = []
        self.source_counts: Dict[str, int] = defaultdict(int)
        self.discovered_counts: Dict[str, int] = defaultdict(int)
        self._schema_sample_positions: Dict[str, List[int]] = defaultdict(list)
        self._reservoir_rng = random.Random(0)
        self.skipped_files: List[Tuple[str, str]] = []
        self.incompatible_counts: Dict[str, int] = defaultdict(int)
        self.excluded_file_count = 0
        self.excluded_tfrecord_split_count = 0
        self.tfrecord_episode_counts: Dict[str, int] = defaultdict(int)
        print(
            f"[数据] 开始扫描根目录：{self.data_dir}\n"
            f"[数据] 模式={','.join(self.sources)}，动作窗口={chunk_size}，"
            f"采样步长={stride}，最短轨迹={min_trajectory_steps}"
        )
        self._scan()
        if bridge_current_gripper == "continuous" and set(self.source_counts) != {"tfrecord_bridge_state_action"}:
            raise ValueError("连续Bridge测量输入只允许单独Bridge来源，不能混评其他数据集")
        if self._bridge_selected:
            actual = {(Path(s["file_path"]).name, s["record_index"]) for s in self.samples}
            if actual != set(self._bridge_selected):
                raise ValueError("Bridge计划中的演示未完整加载，检查分片/语言/长度/排除规则")
        if bcz_target != "reached" and any(s["source"] != "tfrecord_bc_z_pose" for s in self.samples):
            raise ValueError("first_command诊断只允许BC-Z，不能与其他来源或LIBERO实际状态混评")
        if not self.samples:
            raise RuntimeError(
                f"No valid samples found under {self.data_dir} for {self.sources}"
            )
        summary = ", ".join(f"{key}={value}" for key, value in self.source_counts.items())
        print(f"[数据] 扫描完成：共生成 {len(self.samples)} 个动作片段（{summary}）")
        if self.max_samples_per_schema is not None:
            discovered = ", ".join(
                f"{key}={value}" for key, value in self.discovered_counts.items()
            )
            print(
                f"[数据] 完整发现数量：{discovered}；"
                f"检查模式下每种 schema 仅保留前 {self.max_samples_per_schema} 个片段"
            )
        if self.skipped_files:
            print(f"[数据] 有 {len(self.skipped_files)} 个文件无法安全识别，已跳过：")
            for path, reason in self.skipped_files[:10]:
                print(f"  {path}: {reason}")
        if self.incompatible_counts:
            print(
                "[数据] 已识别但未用于低层相对位姿训练："
                + ", ".join(
                    f"{key}={value}" for key, value in self.incompatible_counts.items()
                )
            )
        if self.excluded_file_count:
            print(
                f"[数据] 按路径角色排除了 {self.excluded_file_count} 个文件；"
                f"规则={list(self.exclude_path_parts)}"
            )
        if self.excluded_tfrecord_split_count:
            print(
                f"[数据] 按 TFRecord split 排除了 "
                f"{self.excluded_tfrecord_split_count} 个分片；"
                f"保留={sorted(self.tfrecord_splits)}"
            )
        if self.tfrecord_episode_counts:
            print(
                "[数据] 实际扫描 TFRecord episodes："
                + ", ".join(
                    f"{schema}={count}"
                    for schema, count in self.tfrecord_episode_counts.items()
                )
            )
        if self.exclude_schemas:
            print(f"[数据] 按监督完整性排除 schema：{sorted(self.exclude_schemas)}")

    def _path_is_excluded(self, path: str | Path) -> bool:
        lowered_parts = {part.casefold() for part in Path(path).parts}
        return any(
            fnmatch(path_part, pattern)
            for pattern in self.exclude_path_parts
            for path_part in lowered_parts
        )

    def _limit_reached(self) -> bool:
        return self.max_samples is not None and len(self.samples) >= self.max_samples

    def _append(self, sample: Dict[str, Any]) -> bool:
        schema = sample["source"]
        self.discovered_counts[schema] += 1
        if schema in self.exclude_schemas:
            self.incompatible_counts[f"excluded_schema:{schema}"] += 1
            return True
        # CLIPort 文件只有高层 pick/place 路点，没有与输入图像同步的当前末端位姿，
        # 因而无法无歧义地转成这里的低层相对动作。继续发现并报告，但不混入训练。
        if schema == "cliport":
            self.incompatible_counts[schema] += 1
            return True
        if self._limit_reached():
            return False
        if (
            self.max_samples_per_schema is not None
            and self.source_counts[schema] >= self.max_samples_per_schema
        ):
            # 用蓄水池采样保留分布在整个目录中的代表性样本，同时继续扫描
            # 后续文件和其他 schema，而不是只保留排序最靠前的轨迹。
            replacement = self._reservoir_rng.randrange(self.discovered_counts[schema])
            if replacement < self.max_samples_per_schema:
                sample_position = self._schema_sample_positions[schema][replacement]
                self.samples[sample_position] = sample
            return True
        self.samples.append(sample)
        self._schema_sample_positions[schema].append(len(self.samples) - 1)
        self.source_counts[schema] += 1
        return True

    def _scan(self) -> None:
        if "auto" in self.sources:
            self._scan_hdf5()
            if not self._limit_reached():
                self._scan_cliport()
            if not self._limit_reached():
                self._scan_tfrecord()
            return
        if "libero" in self.sources:
            self._scan_hdf5()
        if "cliport" in self.sources and not self._limit_reached():
            self._scan_cliport()
        if "tfrecord" in self.sources and not self._limit_reached():
            self._scan_tfrecord()

    def _scan_hdf5(self) -> None:
        if h5py is None:
            raise ImportError("h5py is required to discover HDF5 robot datasets")
        files = sorted(
            {path for pattern in ("**/*.hdf5", "**/*.h5") for path in self.data_dir.glob(pattern)}
        )
        excluded = [path for path in files if self._path_is_excluded(path)]
        self.excluded_file_count += len(excluded)
        files = [path for path in files if not self._path_is_excluded(path)]
        print(f"[数据 1/3] 发现 {len(files)} 个 HDF5 文件，正在识别内部字段……")
        for file_path in files:
            recognised = False
            try:
                with h5py.File(file_path, "r") as file:
                    data_group = file["data"] if "data" in file else file
                    for demo_key in data_group.keys():
                        demo = data_group[demo_key]
                        if "actions" not in demo or "obs" not in demo:
                            continue
                        observation = demo["obs"]
                        if "ee_pos" in observation and "ee_ori" in observation:
                            schema = "hdf5_pose_observation"
                            length = min(
                                len(demo["actions"]),
                                len(observation["ee_pos"]),
                                len(observation["ee_ori"]),
                            )
                        elif (
                            "auto" in self.sources
                            and "robot_states" in demo
                            and demo["robot_states"].ndim == 2
                            and demo["robot_states"].shape[1] >= 9
                            and demo["actions"].ndim == 2
                            and demo["actions"].shape[1] >= 4
                            and any(key in observation for key in ("agentview_rgb", "image", "rgb"))
                        ):
                            schema = "hdf5_state9_action4"
                            length = min(len(demo["actions"]), len(demo["robot_states"]))
                        elif (
                            "auto" in self.sources
                            and "ee_states" in observation
                            and observation["ee_states"].ndim == 2
                            and observation["ee_states"].shape[1] >= 3
                            and demo["actions"].ndim == 2
                            and demo["actions"].shape[1] >= 4
                            and any(key in observation for key in ("agentview_rgb", "image", "rgb"))
                        ):
                            schema = "hdf5_position3_action4"
                            length = min(
                                len(demo["actions"]), len(observation["ee_states"])
                            )
                        else:
                            continue
                        recognised = True
                        if length < self.min_trajectory_steps:
                            self.incompatible_counts[
                                f"{schema}_trajectory_too_short"
                            ] += 1
                            continue
                        instruction = self._hdf5_instruction(file_path, file, data_group, demo)
                        for start_index in range(0, max(0, length - 1), self.stride):
                            if not self._append(
                                {
                                    "source": schema,
                                    "file_path": str(file_path),
                                    "demo_key": str(demo_key),
                                    "start_index": start_index,
                                    "instruction": instruction,
                                    "trajectory_steps": length,
                                }
                            ):
                                return
                if not recognised:
                    self.skipped_files.append((str(file_path), "unknown HDF5 schema"))
            except (OSError, KeyError, ValueError) as error:
                self.skipped_files.append((str(file_path), str(error)))

    @staticmethod
    def _hdf5_instruction(
        file_path: Path, file: Any, data_group: Any, demo: Any
    ) -> str:
        # 一些 HDF5/RoboMimic 文件把真正的自然语言放在 data 组的
        # problem_info JSON 中；它必须优先于通用 env_name，否则所有任务
        # 都会退化成同一句 "Libero Tabletop Manipulation"。
        problem_info = data_group.attrs.get("problem_info")
        if problem_info:
            try:
                if isinstance(problem_info, bytes):
                    problem_info = problem_info.decode("utf-8")
                metadata = json.loads(str(problem_info))
                instruction = metadata.get("language_instruction")
                if instruction and str(instruction).strip():
                    return str(instruction).strip()
            except (TypeError, ValueError, json.JSONDecodeError):
                pass
        candidates: Iterable[Any] = (
            demo.attrs.get("language_instruction"),
            demo.attrs.get("instruction"),
            file.attrs.get("language_instruction"),
            data_group.attrs.get("language_instruction"),
            file["problem_info"].attrs.get("language_instruction")
            if "problem_info" in file
            else None,
        )
        for candidate in candidates:
            if candidate is None:
                continue
            if isinstance(candidate, bytes):
                candidate = candidate.decode("utf-8")
            if str(candidate).strip():
                return str(candidate).strip()
        env_args = data_group.attrs.get("env_args")
        if env_args:
            try:
                if isinstance(env_args, bytes):
                    env_args = env_args.decode("utf-8")
                metadata = json.loads(str(env_args))
                task = metadata.get("task_name") or metadata.get("domain_name")
                if task and str(task).lower() not in {"normal", "default"}:
                    return " ".join(str(task).replace("-", " ").replace("_", " ").split())
                domain = metadata.get("domain_name") or metadata.get("env_name")
                if domain:
                    return " ".join(str(domain).replace("-", " ").replace("_", " ").split())
            except (TypeError, ValueError, json.JSONDecodeError):
                pass
        parent_label = file_path.parent.name.replace("-", " ").replace("_", " ")
        if parent_label and parent_label.lower() not in {"data", "dataset", "datasets"}:
            return " ".join(parent_label.split())
        return " ".join(file_path.stem.removesuffix("_demo").replace("_", " ").split())

    def _scan_cliport(self) -> None:
        pattern = os.path.join(str(self.data_dir), "**", "action", "*.pkl")
        action_files = sorted(glob.glob(pattern, recursive=True))
        excluded = [path for path in action_files if self._path_is_excluded(path)]
        self.excluded_file_count += len(excluded)
        action_files = [path for path in action_files if not self._path_is_excluded(path)]
        print(f"[数据 2/3] 发现 {len(action_files)} 组候选 CLIPort 动作文件……")
        for action_path in action_files:
            task_dir = Path(action_path).parent.parent
            filename = Path(action_path).name
            image_path = task_dir / "color" / filename
            info_path = task_dir / "info" / filename
            if not image_path.exists() or not info_path.exists():
                continue
            try:
                with open(action_path, "rb") as stream:
                    actions = pickle.load(stream)
                valid_steps = sum(isinstance(action, dict) for action in actions)
            except (OSError, pickle.UnpicklingError, TypeError):
                continue
            if valid_steps < self.min_trajectory_steps:
                self.incompatible_counts["cliport_trajectory_too_short"] += 1
                continue
            cliport_stride = max(1, self.stride // 4)
            for start_index in range(0, valid_steps, cliport_stride):
                if not self._append(
                    {
                        "source": "cliport",
                        "action_path": action_path,
                        "image_path": str(image_path),
                        "info_path": str(info_path),
                        "start_index": start_index,
                    }
                ):
                    return

    def _scan_tfrecord(self) -> None:
        try:
            import tensorflow as tf
        except ImportError as error:
            raise ImportError("TensorFlow is required for the 'tfrecord' source") from error
        episode_count = 0
        schema_by_parent: Dict[str, str] = {}
        pattern = os.path.join(str(self.data_dir), "**", "*.tfrecord*")
        record_files = sorted(glob.glob(pattern, recursive=True))
        if self._bridge_selected:
            wanted_names = {key[0] for key in self._bridge_selected}
            record_files = [path for path in record_files if Path(path).name in wanted_names]
            if len(record_files) != len(wanted_names):
                raise ValueError("Bridge计划分片缺失或存在同名副本，无法唯一定位")
        excluded = [path for path in record_files if self._path_is_excluded(path)]
        self.excluded_file_count += len(excluded)
        record_files = [path for path in record_files if not self._path_is_excluded(path)]
        if self.tfrecord_splits:
            split_filtered: List[str] = []
            for path in record_files:
                filename = Path(path).name.casefold()
                known_split = next(
                    (
                        split
                        for split in ("train", "validation", "eval", "test")
                        if f"-{split}." in filename
                    ),
                    None,
                )
                # Unknown/legacy filenames are retained. Only an explicitly
                # labelled non-training split is removed, so old local data do
                # not disappear merely because their filenames predate TFDS.
                if known_split is not None and known_split not in self.tfrecord_splits:
                    self.excluded_tfrecord_split_count += 1
                    continue
                split_filtered.append(path)
            record_files = split_filtered
        print(f"[数据 3/3] 发现 {len(record_files)} 个 TFRecord 分片，正在识别内部字段……")
        for file_path in record_files:
            parent_key = str(Path(file_path).parent.resolve()).casefold()
            known_schema = schema_by_parent.get(parent_key)
            if (
                known_schema is not None
                and self.max_tfrecord_episodes_per_schema is not None
                and self.tfrecord_episode_counts[known_schema]
                >= self.max_tfrecord_episodes_per_schema
            ):
                continue
            file_recognised = False
            for record_index, raw_record in enumerate(tf.data.TFRecordDataset([file_path])):
                if self._bridge_selected:
                    selected_indices = [key[1] for key in self._bridge_selected if key[0] == Path(file_path).name]
                    if record_index > max(selected_indices):
                        break
                    if record_index not in selected_indices:
                        continue
                example = tf.train.Example()
                example.ParseFromString(bytes(raw_record.numpy()))
                feature = example.features.feature
                keys = set(feature.keys())
                if self._bridge_selected:
                    row = self._bridge_selected[Path(file_path).name, record_index]
                    origin = feature["episode_metadata/file_path"].bytes_list.value[0].decode("utf-8", errors="replace")
                    eid = feature["episode_metadata/episode_id"].int64_list.value[0]
                    texts = feature["steps/language_instruction"].bytes_list.value
                    normalized = {" ".join(t.decode("utf-8", errors="replace").lower().split()) for t in texts}
                    if (origin != row["origin_file_path"] or eid != row["episode_id"]
                            or len(texts) != row["steps"] or normalized != {row["instruction"]}):
                        raise ValueError("Bridge原始演示身份/长度/指令与固定计划不一致")
                if {
                    "steps/language_instruction",
                    "steps/observation/image",
                    "steps/observation/end_effector_state",
                    "steps/observation/state",
                    "steps/action",
                }.issubset(keys):
                    schema = "tfrecord_end_effector_state"
                    language_key = "steps/language_instruction"
                elif {
                    "steps/observation/instruction",
                    "steps/observation/rgb",
                    "steps/observation/effector_translation",
                    "steps/observation/effector_target_translation",
                    "steps/action",
                }.issubset(keys):
                    # Language Table 的真实机器人数据使用平面内的棍状末端执行器：
                    # 仅有 x/y 位移，没有 z、姿态或夹爪。缺失维度稍后由监督掩码
                    # 屏蔽，绝不能把占位值当作真实的 8 维标签。
                    schema = "tfrecord_language_table_xy"
                    language_key = "steps/observation/instruction"
                elif {
                    "steps/observation/natural_language_instruction",
                    "steps/observation/image",
                    "steps/observation/state",
                    "steps/action/world_vector",
                    "steps/action/rotation_delta",
                    "steps/action/open_gripper",
                }.issubset(keys):
                    schema = "tfrecord_split_delta"
                    language_key = "steps/observation/natural_language_instruction"
                elif {
                    "steps/language_instruction",
                    "steps/observation/image_0",
                    "steps/observation/state",
                    "steps/action",
                }.issubset(keys):
                    # BridgeData V2 的原生 TFDS/RLDS 导出格式。它把末端状态和
                    # 动作分别合并为 7 维向量，并提供至多四个相机视角。
                    schema = "tfrecord_bridge_state_action"
                    language_key = "steps/language_instruction"
                elif {
                    "steps/observation/natural_language_instruction",
                    "steps/observation/image",
                    "steps/observation/present/xyz",
                    "steps/observation/present/axis_angle",
                    "steps/observation/present/sensed_close",
                }.issubset(keys):
                    # BC-Z stores the reached Cartesian pose in observation
                    # fields and packages ten future residual commands into
                    # every step. The reached pose is the unambiguous source
                    # for the common relative-pose supervision contract.
                    schema = "tfrecord_bc_z_pose"
                    language_key = "steps/observation/natural_language_instruction"
                elif {
                    "steps/observation/natural_language_instruction",
                    "steps/observation/image",
                    "steps/observation/base_pose_tool_reached",
                    "steps/observation/gripper_closed",
                    "steps/action/gripper_closedness_action",
                }.issubset(keys):
                    # RT-1 / fractal20220817_data stores reached poses as
                    # [xyz, qx, qy, qz, qw] and closedness commands in [0, 1].
                    schema = "tfrecord_rt1_pose"
                    language_key = "steps/observation/natural_language_instruction"
                else:
                    if not file_recognised:
                        preview = ", ".join(sorted(keys)[:12])
                        self.skipped_files.append(
                            (file_path, f"unknown TFRecord schema; keys include: {preview}")
                        )
                    break
                file_recognised = True
                schema_by_parent[parent_key] = schema
                if (
                    self.max_tfrecord_episodes_per_schema is not None
                    and self.tfrecord_episode_counts[schema]
                    >= self.max_tfrecord_episodes_per_schema
                ):
                    break
                language_feature = feature.get(language_key)
                if schema == "tfrecord_language_table_xy":
                    step_count = len(
                        feature["steps/observation/rgb"].bytes_list.value
                    )
                    flat_tokens = np.asarray(
                        language_feature.int64_list.value if language_feature else (),
                        dtype=np.int64,
                    )
                    if step_count and flat_tokens.size % step_count == 0:
                        token_rows = flat_tokens.reshape(step_count, -1)
                        language_values = tuple(
                            self._decode_language_table_instruction(row).encode("utf-8")
                            for row in token_rows
                        )
                    else:
                        language_values = ()
                else:
                    step_count = (
                        len(language_feature.bytes_list.value)
                        if language_feature
                        else 0
                    )
                    language_values = (
                        language_feature.bytes_list.value if language_feature else ()
                    )
                if step_count < self.min_trajectory_steps:
                    self.incompatible_counts[
                        f"{schema}_trajectory_too_short"
                    ] += 1
                    episode_count += 1
                    self.tfrecord_episode_counts[schema] += 1
                    if (
                        self.max_tfrecord_episodes is not None
                        and episode_count >= self.max_tfrecord_episodes
                    ):
                        return
                    continue
                if not any(value.strip() for value in language_values):
                    # 当前模型以语言为条件；空指令既无法通过检查，也会削弱
                    # 文本分支。所有 schema 都使用同一条完整性规则。
                    self.incompatible_counts[f"{schema}_without_language"] += 1
                for start_index in range(0, max(0, step_count - 1), self.stride):
                    if self._bridge_selected and start_index + (self.bridge_window_horizon or self.chunk_size) >= step_count:
                        # 固定单任务基线只用完整有效窗口，不把重复末观测当额外监督。
                        continue
                    if not language_values[start_index].strip():
                        continue
                    if not self._append(
                        {
                            "source": schema,
                            "file_path": file_path,
                            "record_index": record_index,
                            "start_index": start_index,
                            "trajectory_steps": step_count,
                        }
                    ):
                        return
                episode_count += 1
                self.tfrecord_episode_counts[schema] += 1
                if (
                    self.max_tfrecord_episodes is not None
                    and episode_count >= self.max_tfrecord_episodes
                ):
                    return

    def __len__(self) -> int:
        return len(self.samples)

    def group_key(self, index: int) -> str:
        """Trajectory key used for leakage-free train/validation/test splits."""
        sample = self.samples[index]
        if sample["source"].startswith("hdf5_"):
            return f"{sample['source']}:{sample['file_path']}:{sample['demo_key']}"
        if sample["source"] == "cliport":
            return f"cliport:{sample['action_path']}"
        return f"{sample['source']}:{sample['file_path']}:{sample['record_index']}"

    def __getitem__(
        self, index: int
    ) -> Tuple[str, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        result: Tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]
        if sample["source"] == "hdf5_pose_observation":
            result = self._get_libero(sample)
        elif sample["source"] == "hdf5_state9_action4":
            result = self._get_hdf5_state9_action4(sample)
        elif sample["source"] == "hdf5_position3_action4":
            result = self._get_hdf5_position3_action4(sample)
        elif sample["source"] == "cliport":
            result = self._get_cliport(sample)
        elif sample["source"] == "tfrecord_end_effector_state":
            result = self._get_tfrecord_end_effector_state(sample)
        elif sample["source"] == "tfrecord_split_delta":
            result = self._get_tfrecord_split_delta(sample)
        elif sample["source"] == "tfrecord_bridge_state_action":
            result = self._get_tfrecord_bridge_state_action(sample)
        elif sample["source"] == "tfrecord_bc_z_pose":
            result = self._get_tfrecord_bc_z_pose(sample)
        elif sample["source"] == "tfrecord_rt1_pose":
            result = self._get_tfrecord_rt1_pose(sample)
        elif sample["source"] == "tfrecord_language_table_xy":
            result = self._get_tfrecord_language_table_xy(sample)
        else:
            raise RuntimeError(f"No parser registered for schema {sample['source']!r}")

        supervision_mask = torch.ones(
            (self.chunk_size, ACTION_DIM), dtype=torch.float32
        )
        if sample["source"] == "tfrecord_language_table_xy":
            supervision_mask[:, 2:] = 0.0
        elif sample["source"] == "hdf5_position3_action4":
            supervision_mask[:, 3:7] = 0.0
        return (*result, supervision_mask)

    def _get_libero(
        self, sample: Dict[str, Any]
    ) -> Tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert h5py is not None
        with h5py.File(sample["file_path"], "r") as file:
            data_group = file["data"] if "data" in file else file
            demo = data_group[sample["demo_key"]]
            observation = demo["obs"]
            image_key = next(
                (key for key in ("agentview_rgb", "agentview_image") if key in observation),
                None,
            )
            if image_key is None:
                image_key = next(
                    (key for key in observation.keys() if "image" in key or "rgb" in key),
                    None,
                )
            if image_key is None:
                raise KeyError(f"No RGB image found in {sample['file_path']}")

            start = int(sample["start_index"])
            image = np.asarray(observation[image_key][start])
            positions = np.asarray(observation["ee_pos"])
            orientations = np.asarray(observation["ee_ori"])
            recorded_actions = np.asarray(demo["actions"])
            current_command_index = min(max(start - 1, 0), len(recorded_actions) - 1)
            # RoboSuite/LIBERO 的夹爪控制约定为 -1=张开、+1=闭合；统一模型
            # 合同则规定 +1=张开、-1=闭合，因此这里必须翻转符号。
            current_gripper = (
                1.0 if float(recorded_actions[current_command_index, -1]) < 0.0 else -1.0
            )
            reference_orientation = orientations[start]
            reference_quaternion = (
                rotation_vector_to_quaternion(reference_orientation)
                if reference_orientation.shape[-1] == 3
                else normalise_quaternion(reference_orientation[:4])
            )
            reference_position = positions[start, :3]
            stop = min(len(positions), start + self.chunk_size + 1)
            actions: List[np.ndarray] = []
            for target_index in range(start + 1, stop):
                orientation = orientations[target_index]
                quaternion = (
                    rotation_vector_to_quaternion(orientation)
                    if orientation.shape[-1] == 3
                    else normalise_quaternion(orientation[:4])
                )
                source_index = min(target_index - 1, len(recorded_actions) - 1)
                gripper = (
                    1.0 if float(recorded_actions[source_index, -1]) < 0.0 else -1.0
                )
                actions.append(
                    relative_pose_action(
                        reference_position,
                        reference_quaternion,
                        positions[target_index, :3],
                        quaternion,
                        gripper,
                    )
                )
        return (
            sample["instruction"],
            prepare_image(image),
            torch.tensor([current_gripper], dtype=torch.float32),
            pad_action_chunk(actions, self.chunk_size),
        )

    def _get_hdf5_state9_action4(
        self, sample: Dict[str, Any]
    ) -> Tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Read HDF5 trajectories with [gripper2, xyz3, quaternion4] state."""
        assert h5py is not None
        with h5py.File(sample["file_path"], "r") as file:
            data_group = file["data"] if "data" in file else file
            demo = data_group[sample["demo_key"]]
            observation = demo["obs"]
            image_key = next(
                (key for key in ("agentview_rgb", "image", "rgb") if key in observation),
                None,
            )
            if image_key is None:
                raise KeyError(f"No RGB image found in {sample['file_path']}")
            start = int(sample["start_index"])
            image = np.asarray(observation[image_key][start])
            states = np.asarray(demo["robot_states"], dtype=np.float32)
            recorded_actions = np.asarray(demo["actions"], dtype=np.float32)
            current_command_index = min(max(start - 1, 0), len(recorded_actions) - 1)
            # 该 RoboSuite 数据同样使用 -1=张开、+1=闭合，转换到统一约定。
            current_gripper = (
                1.0 if float(recorded_actions[current_command_index, -1]) < 0.0 else -1.0
            )
            reference_position = states[start, 2:5]
            reference_wxyz = states[start, 5:9]
            reference_quaternion = normalise_quaternion(
                reference_wxyz[[1, 2, 3, 0]]
            )
            stop = min(len(states), start + self.chunk_size + 1)
            actions: List[np.ndarray] = []
            for target_index in range(start + 1, stop):
                position = states[target_index, 2:5]
                quaternion_wxyz = states[target_index, 5:9]
                quaternion = normalise_quaternion(quaternion_wxyz[[1, 2, 3, 0]])
                command = float(recorded_actions[target_index - 1, -1])
                gripper = 1.0 if command < 0.0 else -1.0
                actions.append(
                    relative_pose_action(
                        reference_position,
                        reference_quaternion,
                        position,
                        quaternion,
                        gripper,
                    )
                )
        return (
            sample["instruction"],
            prepare_image(image),
            torch.tensor([current_gripper], dtype=torch.float32),
            pad_action_chunk(actions, self.chunk_size),
        )

    def _get_hdf5_position3_action4(
        self, sample: Dict[str, Any]
    ) -> Tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Read position-only trajectories as relative translation targets.

        These files expose xyz and a 4-D ``delta xyz + gripper`` command but no
        orientation observation. Because their controller does not command
        rotation, the relative rotation target is the identity quaternion.
        """
        assert h5py is not None
        with h5py.File(sample["file_path"], "r") as file:
            data_group = file["data"] if "data" in file else file
            demo = data_group[sample["demo_key"]]
            observation = demo["obs"]
            image_key = next(
                (key for key in ("agentview_rgb", "image", "rgb") if key in observation),
                None,
            )
            if image_key is None:
                raise KeyError(f"No RGB image found in {sample['file_path']}")
            start = int(sample["start_index"])
            image = np.asarray(observation[image_key][start])
            positions = np.asarray(observation["ee_states"], dtype=np.float32)
            recorded_actions = np.asarray(demo["actions"], dtype=np.float32)
            current_command_index = min(max(start - 1, 0), len(recorded_actions) - 1)
            current_gripper = (
                1.0 if float(recorded_actions[current_command_index, -1]) < 0.0 else -1.0
            )
            reference_position = positions[start, :3]
            stop = min(len(positions), start + self.chunk_size + 1)
            identity_quaternion = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            actions: List[np.ndarray] = []
            for target_index in range(start + 1, stop):
                command = float(recorded_actions[target_index - 1, -1])
                gripper = 1.0 if command < 0.0 else -1.0
                actions.append(
                    relative_pose_action(
                        reference_position,
                        identity_quaternion,
                        positions[target_index, :3],
                        identity_quaternion,
                        gripper,
                    )
                )
        return (
            sample["instruction"],
            prepare_image(image),
            torch.tensor([current_gripper], dtype=torch.float32),
            pad_action_chunk(actions, self.chunk_size),
        )

    @staticmethod
    def _cliport_pose_action(pose: Any, gripper: float) -> np.ndarray:
        if not isinstance(pose, (tuple, list)) or len(pose) != 2:
            raise ValueError(f"Invalid CLIPort pose: {pose!r}")
        position = np.asarray(pose[0], dtype=np.float32).reshape(-1)
        quaternion = normalise_quaternion(np.asarray(pose[1], dtype=np.float32))
        if len(position) < 3:
            raise ValueError("CLIPort pose does not contain xyz position")
        return np.concatenate([position[:3], quaternion, [gripper]]).astype(np.float32)

    def _get_cliport(
        self, sample: Dict[str, Any]
    ) -> Tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]:
        with open(sample["action_path"], "rb") as stream:
            raw_actions = pickle.load(stream)
        with open(sample["image_path"], "rb") as stream:
            color = np.asarray(pickle.load(stream))
        with open(sample["info_path"], "rb") as stream:
            information = pickle.load(stream)

        high_level_actions = [action for action in raw_actions if isinstance(action, dict)]
        start = min(int(sample["start_index"]), max(0, len(high_level_actions) - 1))
        waypoints: List[np.ndarray] = []
        for action in high_level_actions[start:]:
            if "pose0" not in action or "pose1" not in action:
                continue
            waypoints.extend(
                [
                    self._cliport_pose_action(action["pose0"], +1.0),
                    self._cliport_pose_action(action["pose0"], -1.0),
                    self._cliport_pose_action(action["pose1"], -1.0),
                    self._cliport_pose_action(action["pose1"], +1.0),
                ]
            )
            if len(waypoints) >= self.chunk_size:
                break

        image = color
        if image.ndim == 5:  # [time, camera, height, width, channel]
            image = image[min(start, image.shape[0] - 1), 0]
        elif image.ndim == 4:
            image = image[min(start, image.shape[0] - 1)]

        instruction = "perform the manipulation task"
        if isinstance(information, list) and information:
            entry = information[min(start, len(information) - 1)]
            if isinstance(entry, dict):
                instruction = str(entry.get("lang_goal", instruction))
        elif isinstance(information, dict):
            instruction = str(information.get("lang_goal", instruction))
        return (
            instruction,
            prepare_image(image),
            torch.tensor([1.0], dtype=torch.float32),
            pad_action_chunk(waypoints, self.chunk_size),
        )

    @staticmethod
    def _load_tfrecord_example(file_path: str, record_index: int) -> Any:
        import tensorflow as tf

        # TFRecord 是可随机寻址的未压缩容器。旧实现每取第 N 条记录都从头
        # 迭代 N 次，在 DataLoader 随机抽样时会产生巨量重复 I/O。这里首次
        # 访问分片时只读取记录头并缓存字节偏移，以后直接 seek 到目标 episode。
        offsets = _TFRECORD_OFFSETS.get(file_path)
        if offsets is None:
            offsets = []
            with open(file_path, "rb") as stream:
                while True:
                    offset = stream.tell()
                    length_bytes = stream.read(8)
                    if not length_bytes:
                        break
                    if len(length_bytes) != 8:
                        raise ValueError(f"Truncated TFRecord length header: {file_path}")
                    record_length = struct.unpack("<Q", length_bytes)[0]
                    if len(stream.read(4)) != 4:
                        raise ValueError(f"Truncated TFRecord length CRC: {file_path}")
                    offsets.append(offset)
                    stream.seek(record_length + 4, io.SEEK_CUR)
            _TFRECORD_OFFSETS[file_path] = offsets
        if not 0 <= record_index < len(offsets):
            raise IndexError(f"TFRecord index {record_index} not found in {file_path}")
        with open(file_path, "rb") as stream:
            stream.seek(offsets[record_index])
            record_length = struct.unpack("<Q", stream.read(8))[0]
            stream.read(4)  # masked CRC of the length
            payload = stream.read(record_length)
            if len(payload) != record_length:
                raise ValueError(f"Truncated TFRecord payload: {file_path}")
        example = tf.train.Example()
        example.ParseFromString(payload)
        return example

    @staticmethod
    def _decode_language_table_instruction(tokens: np.ndarray) -> str:
        """Decode Language Table's zero-padded UTF-8 byte vector."""
        values = np.asarray(tokens, dtype=np.int64).reshape(-1)
        payload = bytes(int(value) for value in values if 0 < int(value) < 256)
        return payload.decode("utf-8", errors="replace").strip()

    def _get_tfrecord_language_table_xy(
        self, sample: Dict[str, Any]
    ) -> Tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Read real Language Table trajectories with truthful partial labels.

        The robot moves a stick end effector in a plane. The source exposes only
        x/y translation, so z, quaternion and gripper are interface placeholders;
        ``__getitem__`` returns an action mask that excludes those six dimensions
        from training and evaluation.
        """
        from PIL import Image

        example = self._load_tfrecord_example(sample["file_path"], sample["record_index"])
        feature = example.features.feature
        required = (
            "steps/observation/instruction",
            "steps/observation/rgb",
            "steps/observation/effector_translation",
            "steps/observation/effector_target_translation",
            "steps/action",
        )
        missing = [key for key in required if key not in feature]
        if missing:
            raise KeyError(f"Unsupported Language Table TFRecord; missing {missing}")

        images = feature["steps/observation/rgb"].bytes_list.value
        step_count = len(images)
        flat_instruction = np.asarray(
            feature["steps/observation/instruction"].int64_list.value,
            dtype=np.int64,
        )
        positions = np.asarray(
            feature["steps/observation/effector_translation"].float_list.value,
            dtype=np.float32,
        )
        if (
            step_count == 0
            or flat_instruction.size % step_count
            or positions.size != step_count * 2
        ):
            raise ValueError("Language Table TFRecord fields have inconsistent lengths")
        instructions = flat_instruction.reshape(step_count, -1)
        positions = positions.reshape(step_count, 2)

        start = int(sample["start_index"])
        reference = np.asarray([positions[start, 0], positions[start, 1], 0.0])
        identity = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        stop = min(step_count, start + self.chunk_size + 1)
        actions: List[np.ndarray] = []
        for target_index in range(start + 1, stop):
            target = np.asarray(
                [positions[target_index, 0], positions[target_index, 1], 0.0],
                dtype=np.float32,
            )
            actions.append(
                relative_pose_action(reference, identity, target, identity, +1.0)
            )

        image = np.asarray(Image.open(io.BytesIO(images[start])).convert("RGB"))
        instruction = self._decode_language_table_instruction(instructions[start])
        return (
            instruction,
            prepare_image(image),
            torch.tensor([1.0], dtype=torch.float32),
            pad_action_chunk(actions, self.chunk_size),
        )

    def _get_tfrecord_end_effector_state(
        self, sample: Dict[str, Any]
    ) -> Tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]:
        from PIL import Image

        example = self._load_tfrecord_example(sample["file_path"], sample["record_index"])
        feature = example.features.feature
        required = (
            "steps/language_instruction",
            "steps/observation/image",
            "steps/observation/end_effector_state",
            "steps/observation/state",
            "steps/action",
        )
        missing = [key for key in required if key not in feature]
        if missing:
            raise KeyError(f"Unsupported TFRecord schema; missing {missing}")

        languages = feature["steps/language_instruction"].bytes_list.value
        images = feature["steps/observation/image"].bytes_list.value
        step_count = len(languages)
        end_effector = np.asarray(
            feature["steps/observation/end_effector_state"].float_list.value,
            dtype=np.float32,
        )
        robot_state = np.asarray(
            feature["steps/observation/state"].float_list.value,
            dtype=np.float32,
        )
        raw_actions = np.asarray(feature["steps/action"].float_list.value, dtype=np.float32)
        if (
            step_count == 0
            or len(images) != step_count
            or end_effector.size % step_count
            or robot_state.size % step_count
            or raw_actions.size % step_count
        ):
            raise ValueError("TFRecord tensors cannot be reshaped with the recorded step count")
        end_effector = end_effector.reshape(step_count, -1)
        robot_state = robot_state.reshape(step_count, -1)
        raw_actions = raw_actions.reshape(step_count, -1)
        if end_effector.shape[1] < 7:
            raise ValueError(f"Expected xyz+quaternion, got {end_effector.shape}")
        if robot_state.shape[1] < 7:
            raise ValueError(
                "Expected Fanuc state=[6 joint angles, gripper, 6 joint velocities], "
                f"got {robot_state.shape}"
            )
        if raw_actions.shape[1] != 6:
            raise ValueError(
                "Expected Fanuc Cartesian action=[dx,dy,dz,droll,dpitch,dyaw], "
                f"got {raw_actions.shape}"
            )

        start = int(sample["start_index"])
        current_gripper = 1.0 if float(robot_state[start, 6]) >= 0.5 else -1.0
        stop = min(step_count, start + self.chunk_size + 1)
        reference_position = end_effector[start, :3]
        reference_wxyz = end_effector[start, 3:7]
        reference_quaternion = normalise_quaternion(
            reference_wxyz[[1, 2, 3, 0]]
        )
        actions: List[np.ndarray] = []
        for target_index in range(start + 1, stop):
            position = end_effector[target_index, :3]
            # Fanuc stores quaternion as [qw, qx, qy, qz]; the common model
            # contract and PyBullet use [qx, qy, qz, qw].
            quaternion_wxyz = end_effector[target_index, 3:7]
            quaternion = normalise_quaternion(quaternion_wxyz[[1, 2, 3, 0]])
            gripper_open = float(robot_state[target_index, 6])
            gripper = 1.0 if gripper_open >= 0.5 else -1.0
            actions.append(
                relative_pose_action(
                    reference_position,
                    reference_quaternion,
                    position,
                    quaternion,
                    gripper,
                )
            )
        image = np.asarray(Image.open(io.BytesIO(images[start])).convert("RGB"))
        instruction = languages[start].decode("utf-8", errors="replace")
        return (
            instruction,
            prepare_image(image),
            torch.tensor([current_gripper], dtype=torch.float32),
            pad_action_chunk(actions, self.chunk_size),
        )

    def _get_tfrecord_split_delta(
        self, sample: Dict[str, Any]
    ) -> Tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Read an RLDS Example with split xyz/rotation/gripper action fields.

        Its observation state is interpreted as absolute
        ``[x, y, z, roll, pitch, yaw, gripper]`` and converted to the same
        tool-relative 8-D target as all other compatible schemas.
        """
        from PIL import Image

        example = self._load_tfrecord_example(sample["file_path"], sample["record_index"])
        feature = example.features.feature
        language_key = "steps/observation/natural_language_instruction"
        languages = feature[language_key].bytes_list.value
        images = feature["steps/observation/image"].bytes_list.value
        step_count = len(languages)
        state = np.asarray(
            feature["steps/observation/state"].float_list.value, dtype=np.float32
        )
        world_vector = np.asarray(
            feature["steps/action/world_vector"].float_list.value, dtype=np.float32
        )
        rotation_delta = np.asarray(
            feature["steps/action/rotation_delta"].float_list.value, dtype=np.float32
        )
        open_gripper = np.asarray(
            feature["steps/action/open_gripper"].int64_list.value, dtype=np.int64
        )
        if (
            step_count == 0
            or len(images) != step_count
            or state.size % step_count
            or world_vector.size != step_count * 3
            or rotation_delta.size != step_count * 3
            or open_gripper.size != step_count
        ):
            raise ValueError("Split-action TFRecord fields have inconsistent lengths")
        state = state.reshape(step_count, -1)
        if state.shape[1] < 7:
            raise ValueError(
                "Expected state=[xyz, roll, pitch, yaw, gripper], "
                f"got {state.shape}"
            )

        start = int(sample["start_index"])
        current_command_index = min(max(start - 1, 0), len(open_gripper) - 1)
        current_gripper = 1.0 if int(open_gripper[current_command_index]) > 0 else -1.0
        stop = min(step_count, start + self.chunk_size + 1)
        reference_position = state[start, :3]
        reference_quaternion = euler_xyz_to_quaternion(state[start, 3:6])
        actions: List[np.ndarray] = []
        for target_index in range(start + 1, stop):
            position = state[target_index, :3]
            quaternion = euler_xyz_to_quaternion(state[target_index, 3:6])
            # 使用真正下发给机器人的动作命令，而不是执行后观测状态。
            # target_index 对应由前一时刻 action 驱动得到的目标状态。
            command_index = min(target_index - 1, len(open_gripper) - 1)
            gripper = 1.0 if int(open_gripper[command_index]) > 0 else -1.0
            actions.append(
                relative_pose_action(
                    reference_position,
                    reference_quaternion,
                    position,
                    quaternion,
                    gripper,
                )
            )
        image = np.asarray(Image.open(io.BytesIO(images[start])).convert("RGB"))
        instruction = languages[start].decode("utf-8", errors="replace")
        return (
            instruction,
            prepare_image(image),
            torch.tensor([current_gripper], dtype=torch.float32),
            pad_action_chunk(actions, self.chunk_size),
        )

    def _get_tfrecord_bridge_state_action(
        self, sample: Dict[str, Any]
    ) -> Tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Read native BridgeData V2 TFDS records.

        Bridge state has ``[x, y, z, roll, pitch, yaw, gripper]`` shape;
        action's first six dimensions are motion commands, NOT absolute poses. The
        observed state supplies the reached Cartesian trajectory, while the
        action supplies the intended gripper command. Its gripper value is a
        continuous opening fraction in ``[0, 1]``, so it is binarised at 0.5
        rather than merely testing whether it is non-zero. This is a project
        legacy convention. Explicit bridge_gripper_policy=reverse_scan_v1 uses
        the official reverse scan, rejecting unresolved nonbinary tail values.
        See single-source audit before mixing datasets. Reached poses are
        converted to the same 8-D tool-relative xyz/quaternion/gripper target
        used by every other compatible source.
        """
        from PIL import Image

        example = self._load_tfrecord_example(sample["file_path"], sample["record_index"])
        feature = example.features.feature
        language_key = "steps/language_instruction"
        image_key = "steps/observation/image_0"
        required = (
            language_key,
            image_key,
            "steps/observation/state",
            "steps/action",
        )
        missing = [key for key in required if key not in feature]
        if missing:
            raise KeyError(f"Unsupported Bridge TFRecord schema; missing {missing}")

        languages = feature[language_key].bytes_list.value
        images = feature[image_key].bytes_list.value
        step_count = len(languages)
        state = np.asarray(
            feature["steps/observation/state"].float_list.value, dtype=np.float32
        )
        raw_actions = np.asarray(
            feature["steps/action"].float_list.value, dtype=np.float32
        )
        if (
            step_count == 0
            or len(images) != step_count
            or state.size % step_count
            or raw_actions.size % step_count
        ):
            raise ValueError("Bridge TFRecord fields have inconsistent lengths")
        state = state.reshape(step_count, -1)
        raw_actions = raw_actions.reshape(step_count, -1)
        if state.shape[1] != 7 or raw_actions.shape[1] != 7:
            raise ValueError(
                "Expected Bridge state/action=[xyz, roll, pitch, yaw, gripper], "
                f"got state={state.shape}, action={raw_actions.shape}"
            )
        gripper_commands = raw_actions[:, 6]
        policy = getattr(self, "bridge_gripper_policy", "threshold_v1")
        if policy in {"reverse_scan_v1", "reverse_scan_valid_steps_v2"}:
            if policy == "reverse_scan_valid_steps_v2":
                first = np.asarray(feature["steps/is_first"].int64_list.value)
                last = np.asarray(feature["steps/is_last"].int64_list.value)
                if (first.shape != (step_count,) or last.shape != (step_count,)
                        or np.flatnonzero(first).tolist() != [0]
                        or np.flatnonzero(last).tolist() != [step_count - 1]):
                    raise ValueError("Bridge有效步策略必须有正确RLDS首末标记")
                # 末步action无效，不能用于向前传播端点、改变有效步的夹爪标签。
                gripper_commands = gripper_commands[:-1]
            gripper_commands = bridge_reverse_scan_gripper(gripper_commands)
            if not np.isin(gripper_commands, [0.0, 1.0]).all():
                raise ValueError("Bridge末尾中间夹爪值无法按官方扫描解析为二值；先审计，不能静默阈值化")

        start = int(sample["start_index"])
        observed_opening = float(state[start, 6])
        if getattr(self, "bridge_current_gripper", "binary") == "continuous":
            if not np.isfinite(observed_opening):
                raise ValueError("Bridge测量必须有限；保持原始数值，不自动裁剪")
            # 连续观测保留抓住物体时的中间开度，不代表上一步控制命令。
            # 实测传感器值可略超过1；仿射变换不是概率化或物理标定。
            current_gripper = 2.0 * observed_opening - 1.0
        else:
            current_gripper = 1.0 if observed_opening >= 0.5 else -1.0
        reference_position = state[start, :3]
        reference_quaternion = euler_xyz_to_quaternion(state[start, 3:6])
        stop = min(step_count, start + self.chunk_size + 1)
        actions: List[np.ndarray] = []
        for target_index in range(start + 1, stop):
            target_position = state[target_index, :3]
            target_quaternion = euler_xyz_to_quaternion(state[target_index, 3:6])
            # target_index 是上一时刻动作执行后的观测，因此夹爪监督取
            # target_index - 1 处真正下发的命令，而不是可能滞后的测量值。
            command_index = min(target_index - 1, len(raw_actions) - 1)
            gripper = (
                1.0 if float(gripper_commands[command_index]) >= 0.5 else -1.0
            )
            actions.append(
                relative_pose_action(
                    reference_position,
                    reference_quaternion,
                    target_position,
                    target_quaternion,
                    gripper,
                )
            )

        image = np.asarray(Image.open(io.BytesIO(images[start])).convert("RGB"))
        instruction = languages[start].decode("utf-8", errors="replace")
        return (
            instruction,
            prepare_image(image),
            torch.tensor([current_gripper], dtype=torch.float32),
            pad_action_chunk(actions, self.chunk_size),
        )

    def _get_tfrecord_bc_z_pose(
        self, sample: Dict[str, Any]
    ) -> Tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Read BC-Z reached poses and convert them to the common 8-D target.

        BC-Z observations expose absolute tool position, an absolute axis-angle
        orientation and sensed gripper closedness.  The dataset also contains
        ten future residual targets. Default mode predicts reached observations;
        first_command is a diagnostic using only native target zero, NOT the
        next observed state. The sensed closedness threshold of 0.5 remains an
        explicit project convention, not an officially verified calibration.
        native_commands keeps all ten matching native targets and continuous
        sensed closure; it is enabled only for isolated training-fit diagnosis.
        """
        from PIL import Image

        example = self._load_tfrecord_example(sample["file_path"], sample["record_index"])
        feature = example.features.feature
        language_key = "steps/observation/natural_language_instruction"
        image_key = "steps/observation/image"
        languages = feature[language_key].bytes_list.value
        images = feature[image_key].bytes_list.value
        step_count = len(languages)
        xyz = np.asarray(
            feature["steps/observation/present/xyz"].float_list.value,
            dtype=np.float32,
        )
        axis_angle = np.asarray(
            feature["steps/observation/present/axis_angle"].float_list.value,
            dtype=np.float32,
        )
        sensed_close = np.asarray(
            feature["steps/observation/present/sensed_close"].float_list.value,
            dtype=np.float32,
        )
        if (
            step_count == 0
            or len(images) != step_count
            or xyz.size != step_count * 3
            or axis_angle.size != step_count * 3
            or sensed_close.size != step_count
        ):
            raise ValueError("BC-Z TFRecord fields have inconsistent lengths")
        xyz = xyz.reshape(step_count, 3)
        axis_angle = axis_angle.reshape(step_count, 3)
        sensed_close = sensed_close.reshape(step_count)

        start = int(sample["start_index"])
        reference_position = xyz[start]
        reference_quaternion = rotation_vector_to_quaternion(axis_angle[start])
        current_gripper = 1.0 if float(sensed_close[start]) < 0.5 else -1.0
        if self.bcz_target in {"first_command", "native_commands"}:
            def command_values(key):
                field = feature[key]
                kind = field.WhichOneof("kind")
                return np.asarray(getattr(field, kind).value, dtype=np.float32) if kind else np.asarray([])
            residual = command_values("steps/action/future/xyz_residual")
            angular = command_values("steps/action/future/axis_angle_residual")
            close = command_values("steps/action/future/target_close")
            if residual.size != step_count * 30 or angular.size != step_count * 30 or close.size != step_count * 10:
                raise ValueError("BC-Z first-command fields must contain 10 targets per observation")
            residual = residual.reshape(step_count, 10, 3)[start]
            angular = angular.reshape(step_count, 10, 3)[start]
            close = close.reshape(step_count, 10)[start]
            if self.bcz_target == "native_commands":
                actions = torch.from_numpy(bcz_native_command_chunk(xyz[start], axis_angle[start], residual, angular, close))
                current_gripper = float(bcz_continuous_gripper_observation(sensed_close[start]))
            else:
                actions = torch.from_numpy(bcz_first_command_action(xyz[start], axis_angle[start], residual[0], angular[0], close[0])).unsqueeze(0)
            image = np.asarray(Image.open(io.BytesIO(images[start])).convert("RGB"))
            return (languages[start].decode("utf-8", errors="replace"), prepare_image(image),
                    torch.tensor([current_gripper], dtype=torch.float32), actions)
        stop = min(step_count, start + self.chunk_size + 1)
        actions: List[np.ndarray] = []
        for target_index in range(start + 1, stop):
            target_quaternion = rotation_vector_to_quaternion(axis_angle[target_index])
            gripper = 1.0 if float(sensed_close[target_index]) < 0.5 else -1.0
            actions.append(
                relative_pose_action(
                    reference_position,
                    reference_quaternion,
                    xyz[target_index],
                    target_quaternion,
                    gripper,
                )
            )

        image = np.asarray(Image.open(io.BytesIO(images[start])).convert("RGB"))
        instruction = languages[start].decode("utf-8", errors="replace")
        return (
            instruction,
            prepare_image(image),
            torch.tensor([current_gripper], dtype=torch.float32),
            pad_action_chunk(actions, self.chunk_size),
        )

    def _get_tfrecord_rt1_pose(
        self, sample: Dict[str, Any]
    ) -> Tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Read RT-1 / fractal20220817_data reached tool poses.

        ``base_pose_tool_reached`` is stored as ``[xyz, qx, qy, qz, qw]``.
        The gripper action is continuous closedness, so values below 0.5 map
        to the project's ``+1 = open`` convention.
        """
        from PIL import Image

        example = self._load_tfrecord_example(sample["file_path"], sample["record_index"])
        feature = example.features.feature
        language_key = "steps/observation/natural_language_instruction"
        image_key = "steps/observation/image"
        languages = feature[language_key].bytes_list.value
        images = feature[image_key].bytes_list.value
        step_count = len(languages)
        poses = np.asarray(
            feature["steps/observation/base_pose_tool_reached"].float_list.value,
            dtype=np.float32,
        )
        observed_closed = np.asarray(
            feature["steps/observation/gripper_closed"].float_list.value,
            dtype=np.float32,
        )
        commanded_closed = np.asarray(
            feature["steps/action/gripper_closedness_action"].float_list.value,
            dtype=np.float32,
        )
        if (
            step_count == 0
            or len(images) != step_count
            or poses.size != step_count * 7
            or observed_closed.size != step_count
            or commanded_closed.size != step_count
        ):
            raise ValueError("RT-1 TFRecord fields have inconsistent lengths")
        poses = poses.reshape(step_count, 7)
        observed_closed = observed_closed.reshape(step_count)
        commanded_closed = commanded_closed.reshape(step_count)

        start = int(sample["start_index"])
        reference_position = poses[start, :3]
        reference_quaternion = normalise_quaternion(poses[start, 3:7])
        current_gripper = 1.0 if float(observed_closed[start]) < 0.5 else -1.0
        stop = min(step_count, start + self.chunk_size + 1)
        actions: List[np.ndarray] = []
        for target_index in range(start + 1, stop):
            command_index = min(target_index - 1, len(commanded_closed) - 1)
            gripper = (
                1.0 if float(commanded_closed[command_index]) < 0.5 else -1.0
            )
            actions.append(
                relative_pose_action(
                    reference_position,
                    reference_quaternion,
                    poses[target_index, :3],
                    normalise_quaternion(poses[target_index, 3:7]),
                    gripper,
                )
            )

        image = np.asarray(Image.open(io.BytesIO(images[start])).convert("RGB"))
        instruction = languages[start].decode("utf-8", errors="replace")
        return (
            instruction,
            prepare_image(image),
            torch.tensor([current_gripper], dtype=torch.float32),
            pad_action_chunk(actions, self.chunk_size),
        )


if __name__ == "__main__":
    dataset = UnifiedRobotDataset(
        data_dir=r"D:\ntu_related\dissertation\dataset",
        sources=("auto",),
        max_samples=32,
        max_tfrecord_episodes=4,
    )
    text, image_tensor, current_gripper_tensor, action_tensor, action_mask = dataset[0]
    print(f"Instruction: {text}")
    print(f"Image: {tuple(image_tensor.shape)}")
    print(f"Actions: {tuple(action_tensor.shape)}")
    print(f"Current gripper: {current_gripper_tensor.item():.1f}")
    print(f"First target: {action_tensor[0].tolist()}")
    print(f"Supervised dimensions: {torch.where(action_mask[0] > 0.5)[0].tolist()}")

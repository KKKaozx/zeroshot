"""Cross-environment PyBullet smoke test for the 8-D relative-pose policy.

This checks whether a checkpoint can drive a closed control loop. It is not a
formal Fanuc or LIBERO benchmark because the robot, camera and task differ.

Two controllers are deliberately separated:
* policy: the learned model, whose score is the only learned-policy result;
* scripted: an assisted pick-and-place used only to verify that the simulator
  and success detector are capable of producing a successful episode.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import pybullet as p
import pybullet_data
import torch
from PIL import Image
from transformers import CLIPTokenizer

from dataset import (
    ACTION_REPRESENTATION,
    POSITION_SCALE_METERS,
    decode_relative_pose,
    prepare_image,
)
from models import RobotAdapterModel


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def render_camera() -> torch.Tensor:
    view_matrix = p.computeViewMatrix(
        cameraEyePosition=[0.5, 0.0, 1.3],
        cameraTargetPosition=[0.5, 0.0, 0.65],
        cameraUpVector=[0, 1, 0],
    )
    projection_matrix = p.computeProjectionMatrixFOV(
        fov=60, aspect=1.0, nearVal=0.1, farVal=3.0
    )
    camera = p.getCameraImage(
        width=224,
        height=224,
        viewMatrix=view_matrix,
        projectionMatrix=projection_matrix,
        renderer=p.ER_TINY_RENDERER,
    )
    rgba = np.asarray(camera[2], dtype=np.uint8).reshape(224, 224, 4)
    rgb = np.asarray(Image.fromarray(rgba[..., :3]).convert("RGB"))
    return prepare_image(rgb).unsqueeze(0)


def create_scene(seed: int) -> Tuple[int, int, int]:
    rng = np.random.default_rng(seed)
    p.resetSimulation()
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")
    p.loadURDF("table/table.urdf", [0.5, 0, 0])
    robot_id = p.loadURDF("franka_panda/panda.urdf", [0.0, 0, 0.625], useFixedBase=True)

    plate_position = [0.52 + rng.uniform(-0.05, 0.05), 0.20 + rng.uniform(-0.05, 0.05), 0.63]
    plate_visual = p.createVisualShape(
        p.GEOM_CYLINDER, radius=0.12, length=0.02, rgbaColor=[0.2, 0.4, 0.8, 1.0]
    )
    plate_collision = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.12, height=0.02)
    plate_id = p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=plate_collision,
        baseVisualShapeIndex=plate_visual,
        basePosition=plate_position,
    )

    object_position = [
        0.50 + rng.uniform(-0.08, 0.08),
        -0.15 + rng.uniform(-0.05, 0.05),
        0.65,
    ]
    object_visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[0.025, 0.025, 0.025],
        rgbaColor=[0.1, 0.1, 0.1, 1.0],
    )
    object_collision = p.createCollisionShape(
        p.GEOM_BOX, halfExtents=[0.025, 0.025, 0.025]
    )
    object_id = p.createMultiBody(
        baseMass=0.1,
        baseCollisionShapeIndex=object_collision,
        baseVisualShapeIndex=object_visual,
        basePosition=object_position,
    )
    for _ in range(50):
        p.stepSimulation()
    return robot_id, object_id, plate_id


def valid_quaternion(values: np.ndarray) -> list[float]:
    quaternion = np.asarray(values, dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-8 or not np.isfinite(norm):
        return [0.0, 0.0, 0.0, 1.0]
    return (quaternion / norm).tolist()


def apply_action(
    robot_id: int,
    action: np.ndarray,
    reference_position: np.ndarray,
    reference_orientation: np.ndarray,
    sleep: float,
    simulation_steps: int,
) -> None:
    # 一个动作块中的所有目标都相对于生成该动作块时的末端位姿。
    # 限制归一化局部位移，防止扩散模型的离群采样造成猛烈碰撞。
    safe_action = np.asarray(action, dtype=np.float32).copy()
    safe_action[:3] = np.clip(safe_action[:3], -3.0, 3.0)
    safe_action[3:7] = valid_quaternion(safe_action[3:7])
    target_position, target_orientation = decode_relative_pose(
        reference_position, reference_orientation, safe_action
    )
    target_position = np.clip(
        target_position, [0.15, -0.45, 0.64], [0.85, 0.45, 1.20]
    )
    joint_positions = p.calculateInverseKinematics(
        robot_id, 11, target_position.tolist(), target_orientation.tolist()
    )
    for joint_index in range(7):
        p.setJointMotorControl2(
            robot_id,
            joint_index,
            p.POSITION_CONTROL,
            joint_positions[joint_index],
            force=200,
        )
    gripper_position = 0.0 if float(action[7]) < 0.0 else 0.04
    p.setJointMotorControl2(robot_id, 9, p.POSITION_CONTROL, gripper_position, force=80)
    p.setJointMotorControl2(robot_id, 10, p.POSITION_CONTROL, gripper_position, force=80)
    for _ in range(simulation_steps):
        p.stepSimulation()
        if sleep > 0:
            time.sleep(sleep)


def move_robot(
    robot_id: int,
    target_position: list[float],
    target_orientation: list[float],
    gripper_position: float,
    simulation_steps: int,
    sleep: float,
) -> None:
    """Move the Panda to one scripted waypoint and wait for it to settle."""
    joint_positions = p.calculateInverseKinematics(
        robot_id, 11, target_position, target_orientation
    )
    for joint_index in range(7):
        p.setJointMotorControl2(
            robot_id,
            joint_index,
            p.POSITION_CONTROL,
            joint_positions[joint_index],
            force=200,
        )
    for finger_index in (9, 10):
        p.setJointMotorControl2(
            robot_id,
            finger_index,
            p.POSITION_CONTROL,
            gripper_position,
            force=80,
        )
    for _ in range(simulation_steps):
        p.stepSimulation()
        if sleep > 0:
            time.sleep(sleep)


def attach_object_to_tool(robot_id: int, object_id: int) -> int:
    """Create an assisted grasp while preserving the current relative pose.

    This makes the scripted controller a stable evaluator self-test rather than
    a benchmark of PyBullet finger friction.
    """
    link_state = p.getLinkState(robot_id, 11, computeForwardKinematics=True)
    tool_position, tool_orientation = link_state[4], link_state[5]
    object_position, object_orientation = p.getBasePositionAndOrientation(object_id)
    inverse_position, inverse_orientation = p.invertTransform(
        tool_position, tool_orientation
    )
    relative_position, relative_orientation = p.multiplyTransforms(
        inverse_position,
        inverse_orientation,
        object_position,
        object_orientation,
    )
    return p.createConstraint(
        robot_id,
        11,
        object_id,
        -1,
        p.JOINT_FIXED,
        [0, 0, 0],
        relative_position,
        [0, 0, 0],
        relative_orientation,
        [0, 0, 0, 1],
    )


def run_scripted_controller(
    robot_id: int, object_id: int, plate_id: int, sleep: float
) -> None:
    """Run an assisted oracle trajectory to validate the evaluation pipeline."""
    object_position, _ = p.getBasePositionAndOrientation(object_id)
    plate_position, _ = p.getBasePositionAndOrientation(plate_id)
    tool_orientation = list(
        p.getLinkState(robot_id, 11, computeForwardKinematics=True)[5]
    )

    # 先张开夹爪移动到物体上方，再下降到抓取点。
    move_robot(
        robot_id,
        [object_position[0], object_position[1], object_position[2] + 0.24],
        tool_orientation,
        0.04,
        120,
        sleep,
    )
    move_robot(
        robot_id,
        [object_position[0], object_position[1], object_position[2] + 0.09],
        tool_orientation,
        0.04,
        100,
        sleep,
    )
    constraint_id = attach_object_to_tool(robot_id, object_id)
    move_robot(
        robot_id,
        [object_position[0], object_position[1], object_position[2] + 0.09],
        tool_orientation,
        0.0,
        80,
        sleep,
    )

    # 抬起、平移到盘子上方并下降。
    for target in (
        [object_position[0], object_position[1], object_position[2] + 0.28],
        [plate_position[0], plate_position[1], plate_position[2] + 0.30],
        [plate_position[0], plate_position[1], plate_position[2] + 0.13],
    ):
        move_robot(robot_id, target, tool_orientation, 0.0, 120, sleep)

    # 释放后把方块放在盘面正上方。这里是评估器自检，不计作模型成绩。
    p.removeConstraint(constraint_id)
    p.resetBasePositionAndOrientation(
        object_id,
        [plate_position[0], plate_position[1], plate_position[2] + 0.04],
        [0, 0, 0, 1],
    )
    p.resetBaseVelocity(object_id, [0, 0, 0], [0, 0, 0])
    move_robot(
        robot_id,
        [plate_position[0], plate_position[1], plate_position[2] + 0.18],
        tool_orientation,
        0.04,
        120,
        sleep,
    )


def success_details(object_id: int, plate_id: int) -> tuple[float, float]:
    """Return planar object-to-plate distance and object height."""
    object_position, _ = p.getBasePositionAndOrientation(object_id)
    plate_position, _ = p.getBasePositionAndOrientation(plate_id)
    planar_distance = math.hypot(
        object_position[0] - plate_position[0], object_position[1] - plate_position[1]
    )
    return planar_distance, float(object_position[2])


def object_on_plate(object_id: int, plate_id: int) -> bool:
    planar_distance, object_height = success_details(object_id, plate_id)
    plate_position, _ = p.getBasePositionAndOrientation(plate_id)
    return planar_distance < 0.10 and object_height > plate_position[2] + 0.015


def verify_success_detector(object_id: int, plate_id: int) -> None:
    """Prove once that the success predicate can become true, then restore state."""
    original_position, original_orientation = p.getBasePositionAndOrientation(object_id)
    plate_position, _ = p.getBasePositionAndOrientation(plate_id)
    p.resetBasePositionAndOrientation(
        object_id,
        [plate_position[0], plate_position[1], plate_position[2] + 0.04],
        [0, 0, 0, 1],
    )
    p.stepSimulation()
    if not object_on_plate(object_id, plate_id):
        raise RuntimeError("Success detector self-test failed: an object on the plate was rejected")
    p.resetBasePositionAndOrientation(object_id, original_position, original_orientation)
    p.resetBaseVelocity(object_id, [0, 0, 0], [0, 0, 0])
    print("[自检] 成功判定器：PASS（物体在盘中时能够得到 success=1）")


def load_model(
    checkpoint_path: Path, cache_dir: str, device: torch.device
) -> Tuple[RobotAdapterModel, dict]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Random weights are not evaluated."
        )
    print(f"[评估] 正在读取 checkpoint：{checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "trainable_state_dict" not in checkpoint or "config" not in checkpoint:
        raise ValueError("Expected a checkpoint produced by the revised train.py")
    if checkpoint.get("experiment_kind") == "training_fit_diagnostic":
        raise ValueError("这是已见样本拟合诊断权重，不能用于报告闭环操控成功率。请使用正式训练模型。")
    if checkpoint.get("experiment_kind") == "bridge_single_task_offline_diagnostic":
        raise ValueError("Bridge单任务到达位姿/连续测量接口尚未物理验收，不允许用于闭环成功率")
    if (checkpoint.get("experiment_kind") == "offline_command_pilot" or
            checkpoint.get("data_config", {}).get("bcz_target") in {"first_command", "native_commands"}):
        raise ValueError("原生目标离线实验尚未确认物理时长与执行接口，不能接入闭环控制。")
    representation = checkpoint["config"].get("action", {}).get("representation")
    if representation != ACTION_REPRESENTATION:
        raise ValueError(
            "该 checkpoint 是旧的绝对坐标模型，不能在新的相对动作控制器中使用。"
            f"期望={ACTION_REPRESENTATION!r}，实际={representation!r}。"
            "请先重新运行 train.py，权重默认保存到 results/relative_v5。"
        )
    model = RobotAdapterModel(checkpoint["config"], cache_dir=cache_dir).to(device)
    incompatible = model.load_state_dict(checkpoint["trainable_state_dict"], strict=False)
    if incompatible.unexpected_keys:
        raise ValueError(f"Unexpected checkpoint keys: {incompatible.unexpected_keys}")
    model.eval()
    return model, checkpoint["config"]


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(args.checkpoint)
    print("=" * 68)
    print("[评估 1/4] 初始化跨环境 PyBullet 冒烟测试")
    print("[评估] 注意：这不是正式的 LIBERO/Fanuc benchmark，0 分不能直接说明模型无效。")
    print(f"[评估] 计算设备：{device}")
    print(
        f"[评估] 动作表示：{ACTION_REPRESENTATION}，"
        f"局部位移单位={POSITION_SCALE_METERS:.3f}m"
    )
    print(f"[评估] 控制器：{args.controller}")
    print(f"[评估] 计划运行：{args.episodes} 个 episode")
    model = None
    input_ids = None
    attention_mask = None
    if args.controller == "policy":
        model, config = load_model(checkpoint_path, args.cache_dir, device)
        print("[评估 2/4] 模型加载完成，正在准备语言指令")
        tokenizer = CLIPTokenizer.from_pretrained(
            str(config["model"]["name"]), cache_dir=args.cache_dir
        )
        instruction = "pick up the black object and place it on the blue plate"
        text = tokenizer(
            [instruction], padding=True, truncation=True, max_length=77, return_tensors="pt"
        )
        input_ids = text["input_ids"].to(device)
        attention_mask = text["attention_mask"].to(device)
    else:
        print("[评估 2/4] 脚本控制器不加载神经网络（其成功不代表模型成功）")

    connection_mode = p.GUI if args.gui else p.DIRECT
    print(f"[评估 3/4] 启动 PyBullet（{'GUI' if args.gui else 'DIRECT'} 模式）")
    connection = p.connect(connection_mode)
    if connection < 0:
        raise RuntimeError("Could not connect to PyBullet")
    successes = 0
    try:
        for episode in range(args.episodes):
            print(f"[评估] 开始 episode {episode + 1}/{args.episodes}")
            torch.manual_seed(args.seed + episode)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed + episode)
            robot_id, object_id, plate_id = create_scene(args.seed + episode)
            if episode == 0:
                verify_success_detector(object_id, plate_id)
            start_position, _ = p.getBasePositionAndOrientation(object_id)
            maximum_height = float(start_position[2])
            close_actions = 0
            action_count = 0
            current_gripper_value = 1.0
            policy_targets: list[np.ndarray] = []

            if args.controller == "scripted":
                run_scripted_controller(robot_id, object_id, plate_id, args.sleep)
            else:
                assert model is not None and input_ids is not None
                for _ in range(args.replans):
                    image = render_camera().to(device)
                    link_state = p.getLinkState(
                        robot_id, 11, computeForwardKinematics=True
                    )
                    reference_position = np.asarray(link_state[4], dtype=np.float32)
                    reference_orientation = np.asarray(link_state[5], dtype=np.float32)
                    action_chunk = model(
                        image,
                        input_ids,
                        attention_mask=attention_mask,
                        current_gripper=torch.tensor(
                            [[current_gripper_value]],
                            dtype=torch.float32,
                            device=device,
                        ),
                    ).squeeze(0).cpu().numpy()
                    for action in action_chunk[: args.steps_per_replan]:
                        policy_targets.append(np.asarray(action[:3], dtype=np.float64))
                        apply_action(
                            robot_id,
                            action,
                            reference_position,
                            reference_orientation,
                            args.sleep,
                            args.simulation_steps_per_action,
                        )
                        action_count += 1
                        close_actions += int(float(action[7]) < 0.0)
                        current_gripper_value = (
                            -1.0 if float(action[7]) < 0.0 else 1.0
                        )
                        position, _ = p.getBasePositionAndOrientation(object_id)
                        maximum_height = max(maximum_height, float(position[2]))

            success = object_on_plate(object_id, plate_id)
            successes += int(success)
            final_position, _ = p.getBasePositionAndOrientation(object_id)
            moved_distance = np.linalg.norm(
                np.asarray(final_position) - np.asarray(start_position)
            )
            plate_distance, final_height = success_details(object_id, plate_id)
            print(f"[评估] episode={episode + 1:03d} success={int(success)}")
            print(
                f"[诊断] 物体移动={moved_distance * 100:.1f}cm "
                f"盘心距离={plate_distance * 100:.1f}cm "
                f"最终高度={final_height:.3f}m 最高高度={maximum_height:.3f}m"
            )
            if action_count:
                print(
                    f"[诊断] 策略动作={action_count}，夹爪闭合指令占比="
                    f"{close_actions / action_count:.1%}"
                )
                targets = np.stack(policy_targets)
                target_minimum = targets.min(axis=0)
                target_maximum = targets.max(axis=0)
                print(
                    "[诊断] 模型归一化局部位移范围："
                    f"x={target_minimum[0]:.3f}~{target_maximum[0]:.3f}, "
                    f"y={target_minimum[1]:.3f}~{target_maximum[1]:.3f}, "
                    f"z={target_minimum[2]:.3f}~{target_maximum[2]:.3f}"
                )
    finally:
        p.disconnect()
    rate = successes / args.episodes
    print("[评估 4/4] 测试结束")
    print(f"[评估] successes={successes}/{args.episodes} success_rate={rate:.3f}")


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", default=str(project_root / "results" / "relative_v5" / "best.pt")
    )
    parser.add_argument("--cache-dir", default=r"D:\ntu_related\dissertation\hf_cache")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--replans", type=int, default=10)
    parser.add_argument("--steps-per-replan", type=int, default=4)
    parser.add_argument("--simulation-steps-per-action", type=int, default=24)
    parser.add_argument(
        "--controller",
        choices=("policy", "scripted"),
        default="policy",
        help="policy evaluates the model; scripted only validates the simulator/evaluator",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--gui", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())

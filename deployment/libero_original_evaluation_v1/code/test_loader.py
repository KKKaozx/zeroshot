"""Fast structural checks for the unified 8-D robot dataset."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import torch

from dataset import (
    ACTION_DIM,
    ACTION_REPRESENTATION,
    POSITION_SCALE_METERS,
    UnifiedRobotDataset,
)


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def validate_sample(
    instruction: str,
    image: torch.Tensor,
    current_gripper: torch.Tensor,
    actions: torch.Tensor,
    supervision_mask: torch.Tensor,
    chunk_size: int,
) -> None:
    assert isinstance(instruction, str) and instruction.strip(), "Instruction is empty"
    assert image.shape == (3, 224, 224), f"Unexpected image shape: {tuple(image.shape)}"
    assert actions.shape == (
        chunk_size,
        ACTION_DIM,
    ), f"Unexpected action shape: {tuple(actions.shape)}"
    assert torch.isfinite(image).all(), "Image contains NaN/Inf"
    assert torch.isfinite(actions).all(), "Actions contain NaN/Inf"
    assert supervision_mask.shape == actions.shape
    assert torch.all((supervision_mask == 0.0) | (supervision_mask == 1.0))
    assert supervision_mask.any(), "Sample has no supervised action dimension"
    assert current_gripper.shape == (1,), (
        f"Unexpected current gripper shape: {tuple(current_gripper.shape)}"
    )
    assert current_gripper.item() in (-1.0, 1.0), "Current gripper must be -1 or +1"
    quaternion_norms = actions[:, 3:7].norm(dim=-1)
    assert torch.allclose(
        quaternion_norms, torch.ones_like(quaternion_norms), atol=1e-4
    ), f"Non-unit quaternion detected: {quaternion_norms}"
    assert torch.all((actions[:, 7] == -1.0) | (actions[:, 7] == 1.0)), (
        "Gripper targets must be binary values in {-1, +1}"
    )


def check_model_math() -> None:
    """不用下载 CLIP，验证扩散终点、旧表兼容和理想去噪器的采样还原。"""
    from train import build_learning_rate_scheduler
    from train import select_bridge_gripper_candidates
    from unittest.mock import patch as candidate_patch
    from types import SimpleNamespace as CandidateFixture
    import json as candidate_json
    candidate_dataset = CandidateFixture(samples=[{"start_index": i} for i in range(4)])
    candidate_rows = [{"index": i, "sample": candidate_dataset.samples[i], "category": category,
                       "loader_image_matches_raw_preprocessing": True,
                       "loader_gripper_matches_command_labels": True}
                      for i, category in enumerate(("hold_open", "hold_closed", "open_to_closed", "closed_to_open"))]
    candidate_report = {"purpose": "bridge_training_gripper_coverage_candidates_not_training_manifest",
                        "dataset_identity": "fixture", "validation_test_targets_inspected": False,
                        "candidate_input_checks": {"exact_input_conflicts": 0, "checked_candidates": 4, "checks": candidate_rows}}
    with candidate_patch("train.dataset_split_identity", return_value="fixture"), candidate_patch(
            "pathlib.Path.read_text", return_value=candidate_json.dumps(candidate_report)):
        assert select_bridge_gripper_candidates(candidate_dataset, list(range(4)), "fixture.json")[0] == list(range(4))
        try:
            select_bridge_gripper_candidates(candidate_dataset, [0, 1, 2], "fixture.json")
            raise AssertionError("留出候选应被拒绝")
        except ValueError:
            pass
    from audit_bcz import bridge_gripper_window_category
    assert bridge_gripper_window_category([1, 1, 1]) == "hold_open"
    assert bridge_gripper_window_category([0, 0, 0]) == "hold_closed"
    assert bridge_gripper_window_category([1, 1, 0]) == "open_to_closed"
    assert bridge_gripper_window_category([0, 0, 1]) == "closed_to_open"
    assert bridge_gripper_window_category([0, 1, 0]) == "multiple_switches"
    try:
        bridge_gripper_window_category([0, .5, 1])
        raise AssertionError("非二值命令应被拒绝")
    except ValueError:
        pass
    for schedule in ("constant", "cosine"):
        scheduler_parameter = torch.nn.Parameter(torch.zeros(1))
        scheduler_optimizer = torch.optim.SGD([
            {"params": [scheduler_parameter], "lr": 3e-5},
            {"params": [], "lr": 3e-4},
        ])
        scheduler = build_learning_rate_scheduler(scheduler_optimizer, schedule, 50)
        for _ in range(50):
            scheduler_optimizer.step()
            scheduler.step()
        expected_rates = (3e-5, 3e-4) if schedule == "constant" else (0., 0.)
        for group, expected in zip(scheduler_optimizer.param_groups, expected_rates):
            assert abs(group["lr"] - expected) < 1e-12
    from types import SimpleNamespace
    from unittest.mock import patch
    import numpy as np
    from models import RobotAdapterModel, diffusion_betas
    from dataset import relative_pose_action, decode_relative_pose, rotation_vector_to_quaternion
    from dataset import prepare_image, CLIP_IMAGE_MEAN, CLIP_IMAGE_STD
    # 暗uint8图必须与显式除255的浮点图一致，不能被放大255倍。
    dark = np.ones((224, 224, 3), dtype=np.uint8)
    assert torch.allclose(prepare_image(dark), prepare_image(dark.astype(np.float32) / 255), atol=1e-7)
    restored = prepare_image(dark) * CLIP_IMAGE_STD + CLIP_IMAGE_MEAN
    assert torch.allclose(restored, torch.full_like(restored, 1 / 255), atol=1e-7)
    for invalid in (np.full((2, 2, 3), np.nan), np.full((2, 2, 3), -1.0), np.ones((2, 2, 3), dtype=np.uint16)):
        try:
            prepare_image(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid image was accepted")
    print("[图像检查] 通过：暗uint8缩放正确；非有限、负值和未声明整数格式拒绝")
    from dataset import bridge_reverse_scan_gripper
    assert np.array_equal(bridge_reverse_scan_gripper([1, .8, .2, 0]), [1, 0, 0, 0])
    assert np.array_equal(bridge_reverse_scan_gripper([0, .2, .8, 1]), [0, 1, 1, 1])
    assert np.allclose(bridge_reverse_scan_gripper([1, .95, .05, .4]), [1, .4, .4, .4])
    for invalid in ([], [np.nan], [-.1], [1.1], [[0, 1]]):
        try:
            bridge_reverse_scan_gripper(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid Bridge command was accepted")
    print("[Bridge夹爪检查] 通过：双向过渡跟随后续端点，尾部中间值保留，非法输入拒绝")
    from train import dataset_split_identity
    identity_fixture = SimpleNamespace(samples=[{"source": "tfrecord_bridge_state_action", "file_path": __file__}],
        chunk_size=16, stride=4, bcz_target="reached", bcz_current_gripper="binary")
    legacy_identity = dataset_split_identity(identity_fixture)
    identity_fixture.bridge_gripper_policy = "threshold_v1"
    assert dataset_split_identity(identity_fixture) == legacy_identity
    identity_fixture.bridge_gripper_policy = "reverse_scan_v1"
    assert dataset_split_identity(identity_fixture) != legacy_identity
    scan_identity = dataset_split_identity(identity_fixture)
    identity_fixture.bridge_current_gripper = "continuous"
    assert dataset_split_identity(identity_fixture) != scan_identity
    print("[Bridge版本检查] 通过：旧默认指纹兼容，新监督策略具有不同数据身份")
    from train import select_bridge_fit_windows
    fit_fixture = SimpleNamespace(group_key=lambda index: f"episode_{index // 4}")
    assert select_bridge_fit_windows(fit_fixture, list(range(12)), 3) == [2, 6, 10]
    assert select_bridge_fit_windows(fit_fixture, list(range(12)), 2) == [2, 10]
    for count in (1, 4):
        try:
            select_bridge_fit_windows(fit_fixture, list(range(12)), count)
        except ValueError:
            pass
        else:
            raise AssertionError("Bridge拟合选择未拒绝非法数量")
    print("[Bridge拟合选择检查] 通过：仅训练演示均匀抽样、每演示中间1窗口、不读取目标选样")
    from audit_dataset import inspect_bridge_timing
    timing_state = np.zeros((3, 7)); timing_state[:, 0] = [0, .1, .3]
    timing_command = np.zeros((3, 7)); timing_command[:, 0] = [.1, .2, 0]
    timing = inspect_bridge_timing(timing_state, timing_command, [1, 0, 0], [0, 0, 1])
    assert timing["is_first_valid"] and timing["is_last_valid"] and timing["last_action_all_zero"]
    assert timing["next_delta_vs_action_t_xyz_rmse_native_units"] < 1e-12
    assert timing["next_delta_vs_action_t_plus_1_xyz_rmse_native_units"] > .01
    assert not inspect_bridge_timing(timing_state, timing_command, [1, 1, 0])["is_first_valid"]
    print("[Bridge边界检查] 通过：空末帧/首末标记/时序候选统计可定位；不自动移帧")
    from adapter import CrossAttentionAdapter
    torch.manual_seed(42)
    old_adapter = CrossAttentionAdapter(num_layers=2, embed_dim=16, text_dim=12,
        attention_dim=8, num_heads=2, dropout=0, pooling="cls").eval()
    new_adapter = CrossAttentionAdapter(num_layers=2, embed_dim=16, text_dim=12,
        attention_dim=8, num_heads=2, dropout=0, pooling="cls_patch_mean").eval()
    new_adapter.load_state_dict(old_adapter.state_dict())
    visual = torch.randn(2, 5, 16, requires_grad=True)
    text = torch.randn(2, 4, 12)
    altered = visual.detach().clone()
    altered[:, 1:] = torch.randn_like(altered[:, 1:]) * 10
    assert torch.equal(old_adapter(visual, text), old_adapter(altered, text))
    fused = visual
    for layer in old_adapter.layers:
        fused = layer(fused, text)
    assert torch.allclose(old_adapter(visual, text), old_adapter.output_norm(fused)[:, 0], atol=1e-7)
    output = new_adapter(visual, text)
    patch_delta = float((output - new_adapter(altered, text)).abs().max().detach())
    assert patch_delta > 1e-4
    (output * torch.arange(16)).sum().backward()
    patch_gradient = float(visual.grad[:, 1:].abs().max())
    assert patch_gradient > 1e-6
    print(f"[视觉通路检查] 通过：旧CLS输出兼容；新patch影响输出={patch_delta:.6f}，patch梯度={patch_gradient:.6f}")

    linear = diffusion_betas(100, "linear")
    cosine = diffusion_betas(100, "squaredcos_cap_v2")
    assert torch.equal(linear, torch.linspace(1e-4, .02, 100))
    old_terminal = float((1 - linear).cumprod(0)[-1])
    new_terminal = float((1 - cosine).cumprod(0)[-1])
    assert old_terminal > .3 and new_terminal < 1e-6

    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(hidden_size=8)

    config = {"model": {"name": "test-only", "chunk_size": 16,
        "decoder_hidden_dim": 8, "num_adapter_layers": 1,
        "attention_dim": 8, "num_attention_heads": 2,
        "beta_schedule": "squaredcos_cap_v2", "clip_denoised": True},
        "action": {"max_normalized_position": 3}}
    with patch("models.CLIPModel.from_pretrained", return_value=SimpleNamespace(vision_model=Encoder(), text_model=Encoder())):
        model = RobotAdapterModel(config)
        legacy = RobotAdapterModel({"model": {**config["model"], "beta_schedule": "linear", "clip_denoised": False}})
        assert torch.equal(legacy.betas, linear)
        no_schedule = dict(config["model"])
        no_schedule.pop("beta_schedule")
        no_schedule.pop("clip_denoised")
        assert RobotAdapterModel({"model": no_schedule}).beta_schedule == "linear"
        regression = RobotAdapterModel({"model": {**config["model"],
            "decoder_type": "regression", "separate_gripper_head": True,
            "trajectory_conditioned_gripper": True, "condition_on_current_gripper": True,
            "gripper_target_mode": "state"}})

    target = torch.zeros(2, 16, 8)
    target[..., :3] = .2
    target[..., 6:] = 1
    class Oracle(torch.nn.Module):
        def forward(self, noisy, timestep, context):
            abar = model.alpha_bars[timestep].reshape(-1, 1, 1)
            return (noisy - abar.sqrt() * target) / (1 - abar).sqrt()
    model.diffusion_decoder = Oracle()
    predicted = model.sample(torch.zeros(2, 8))
    assert torch.isfinite(predicted).all()
    assert torch.allclose(predicted, target, atol=1e-3)
    class CleanOracle(torch.nn.Module):
        def forward(self, noisy, timestep, context):
            return target
    model.diffusion_prediction_type = "sample"
    model.diffusion_decoder = CleanOracle()
    assert torch.allclose(model.sample(torch.zeros(2, 8)), target, atol=1e-3)
    from train import policy_loss
    context = torch.zeros(2, 8)
    current = torch.ones(2, 1)
    first = regression.sample(context, current)
    assert first.shape == (2, 16, 8)
    assert torch.equal(first, regression.sample(context, current))
    assert torch.allclose(first[..., 3:7].norm(dim=-1), torch.ones(2, 16), atol=1e-5)
    output = regression.diffusion_loss(target, context, current)
    loss, pose_loss, _ = policy_loss(regression, output, target, torch.nn.MSELoss(), current, torch.ones_like(target))
    negative = target.clone(); negative[..., 3:7] *= -1
    negative_output = (output[0], negative[..., :7], output[2])
    _, negative_pose_loss, _ = policy_loss(regression, negative_output, negative, torch.nn.MSELoss(), current, torch.ones_like(target))
    assert torch.allclose(pose_loss, negative_pose_loss)
    rotation_only = target[..., :7].clone()
    rotation_only[..., 3] = .1
    rotation_only[..., 6] = (1 - .1 ** 2) ** .5
    rotation_output = (rotation_only, target[..., :7], output[2])
    _, rotation1, grip1 = policy_loss(regression, rotation_output, target, torch.nn.MSELoss(), current, torch.ones_like(target))
    regression.regression_rotation_weight = 4.0
    _, rotation4, grip4 = policy_loss(regression, rotation_output, target, torch.nn.MSELoss(), current, torch.ones_like(target))
    assert torch.allclose(rotation4, 4 * rotation1) and torch.equal(grip1, grip4)
    regression.regression_rotation_weight = 1.0
    loss.backward()
    assert torch.isfinite(regression.regression_head[-1].weight.grad).all()

    reference = np.array([.2, -.1, .3], dtype=np.float32)
    orientation = rotation_vector_to_quaternion(np.array([.1, .2, .3]))
    goal = reference + np.array([.04, -.02, .01])
    goal_q = rotation_vector_to_quaternion(np.array([-.2, .1, .4]))
    encoded = relative_pose_action(reference, orientation, goal, goal_q, 1)
    decoded_position, decoded_q = decode_relative_pose(reference, orientation, encoded)
    assert np.allclose(decoded_position, goal, atol=1e-6)
    assert abs(float(np.dot(decoded_q, goal_q))) > 1 - 1e-6
    from dataset import bcz_first_command_action
    command_delta = np.array([.04, -.02, .01], dtype=np.float32)
    angular_delta = np.array([-.15, .22, .03], dtype=np.float32)
    current_angle = np.array([.2, -.4, .3], dtype=np.float32)
    for close in (0, 1):
        command = bcz_first_command_action(reference, current_angle, command_delta, angular_delta, close)
        restored_p, restored_q = decode_relative_pose(reference, rotation_vector_to_quaternion(current_angle), command)
        assert command.shape == (8,)
        assert np.allclose(restored_p, reference + command_delta, atol=1e-6)
        assert abs(float(np.dot(restored_q, rotation_vector_to_quaternion(current_angle + angular_delta)))) > 1 - 1e-6
        assert command[7] == (1 if close == 0 else -1)
    try:
        bcz_first_command_action(reference, current_angle, command_delta, angular_delta, .3)
    except ValueError:
        pass
    else:
        raise AssertionError("非二值控制目标没有被拒绝")
    print("[BC-Z控制目标检查] 通过：原生残差相加、8维转换往返、绝对夹爪命令；不假定物理时长")
    from dataset import bcz_native_command_chunk, bcz_continuous_gripper_observation
    deltas = np.stack([command_delta * (k + 1) for k in range(10)])
    angle_deltas = np.stack([angular_delta * (k + 1) for k in range(10)])
    closes = np.arange(10) % 2
    chunk = bcz_native_command_chunk(reference, current_angle, deltas, angle_deltas, closes)
    assert chunk.shape == (10, 8) and chunk.dtype == np.float32
    for k, action in enumerate(chunk):
        p, q = decode_relative_pose(reference, rotation_vector_to_quaternion(current_angle), action)
        assert np.allclose(p, reference + deltas[k], atol=1e-6)  # 不累加相邻残差
        assert abs(float(np.dot(q, rotation_vector_to_quaternion(current_angle + angle_deltas[k])))) > 1 - 1e-6
        assert action[7] == (1 if closes[k] == 0 else -1)  # 不错配夹爪索引
    encoded_close = bcz_continuous_gripper_observation(np.array([0., .2, .6, .9, 1.]))
    assert np.allclose(encoded_close, [1., .6, -.2, -.8, -1.])
    assert encoded_close[2] != encoded_close[3]  # 同二值桶仍保留闭合程度差别
    for invalid in (np.array([-.1]), np.array([1.1]), np.array([np.nan])):
        try:
            bcz_continuous_gripper_observation(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("连续观测的越界/非有限值未被拒绝")
    try:
        bcz_native_command_chunk(reference, current_angle, deltas[:9], angle_deltas, closes)
    except ValueError:
        pass
    else:
        raise AssertionError("原生10目标的形状错误未被拒绝")
    print("[BC-Z原生序列检查] 通过：同索引10×8目标、不累加残差、连续观测保真；默认训练未改变")
    # 真实加载方法的单观测fixture：原生10目标无需伪造后续观测或尾部填充。
    import io
    from types import SimpleNamespace
    from PIL import Image
    encoded_image = io.BytesIO()
    Image.fromarray(np.zeros((16, 20, 3), dtype=np.uint8)).save(encoded_image, format="JPEG")
    def feature_fixture(values, kind):
        return SimpleNamespace(WhichOneof=lambda _: kind, **{kind: SimpleNamespace(value=values)})
    fields = {
        "steps/observation/natural_language_instruction": feature_fixture([b"fixture"], "bytes_list"),
        "steps/observation/image": feature_fixture([encoded_image.getvalue()], "bytes_list"),
        "steps/observation/present/xyz": feature_fixture(reference.tolist(), "float_list"),
        "steps/observation/present/axis_angle": feature_fixture(current_angle.tolist(), "float_list"),
        "steps/observation/present/sensed_close": feature_fixture([.6], "float_list"),
        "steps/action/future/xyz_residual": feature_fixture(deltas.ravel().tolist(), "float_list"),
        "steps/action/future/axis_angle_residual": feature_fixture(angle_deltas.ravel().tolist(), "float_list"),
        "steps/action/future/target_close": feature_fixture(closes.tolist(), "int64_list"),
    }
    native_fixture = UnifiedRobotDataset.__new__(UnifiedRobotDataset)
    native_fixture.samples = [{"source": "tfrecord_bc_z_pose", "file_path": "fixture", "record_index": 0, "start_index": 0}]
    native_fixture.chunk_size = 10
    native_fixture.bcz_target = "native_commands"
    native_fixture.bcz_current_gripper = "continuous"
    native_fixture._load_tfrecord_example = lambda *_: SimpleNamespace(features=SimpleNamespace(feature=fields))
    loaded = native_fixture[0]
    assert loaded[3].shape == (10, 8) and bool(loaded[4].all())
    assert abs(loaded[2].item() + .2) < 1e-6 and np.allclose(loaded[3].numpy(), chunk)
    native_fixture.bcz_target, native_fixture.chunk_size = "first_command", 1
    old = native_fixture[0]
    assert old[2].item() == -1 and np.allclose(old[3][0].numpy(), chunk[0])
    print("[加载兼容检查] 通过：原生10步不补齐/混帧，连续输入有效，第一目标旧二值输入不变")
    bridge_state = np.zeros((3, 7)); bridge_state[:, 6] = [.58, .6, 1.]
    bridge_commands = np.zeros((3, 7)); bridge_commands[:, 6] = [0., 1., .4]
    bridge_fields = {
        "steps/language_instruction": feature_fixture([b"fixture"] * 3, "bytes_list"),
        "steps/observation/image_0": feature_fixture([encoded_image.getvalue()] * 3, "bytes_list"),
        "steps/observation/state": feature_fixture(bridge_state.ravel().tolist(), "float_list"),
        "steps/action": feature_fixture(bridge_commands.ravel().tolist(), "float_list"),
        "steps/is_first": feature_fixture([1, 0, 0], "int64_list"),
        "steps/is_last": feature_fixture([0, 0, 1], "int64_list"),
    }
    bridge_fixture = UnifiedRobotDataset.__new__(UnifiedRobotDataset)
    bridge_fixture.samples = [{"source": "tfrecord_bridge_state_action", "file_path": "fixture", "record_index": 0, "start_index": 0}]
    bridge_fixture.chunk_size = 2
    bridge_fixture.bridge_gripper_policy = "reverse_scan_valid_steps_v2"
    bridge_fixture.bridge_current_gripper = "continuous"
    bridge_fixture._load_tfrecord_example = lambda *_: SimpleNamespace(features=SimpleNamespace(feature=bridge_fields))
    continuous = bridge_fixture[0]
    assert abs(continuous[2].item() - .16) < 1e-6
    assert continuous[3][:, 7].tolist() == [-1., 1.]  # 不用测量或无效末步当命令
    bridge_fixture.bridge_current_gripper = "binary"
    binary = bridge_fixture[0]
    assert binary[2].item() == 1. and torch.equal(binary[3], continuous[3])
    bridge_fixture.bridge_current_gripper = "continuous"
    bridge_fields["steps/action"].float_list.value[-1] = float("nan")
    assert torch.equal(bridge_fixture[0][3], continuous[3])  # 无效末动作不污染扫描
    bridge_fields["steps/observation/state"].float_list.value[6] = 1.01
    assert abs(bridge_fixture[0][2].item() - 1.02) < 1e-6  # 保留测量超调，不裁剪成1
    for invalid in (float("inf"), -float("inf"), float("nan")):
        bridge_fields["steps/observation/state"].float_list.value[6] = invalid
        try:
            bridge_fixture[0]
        except ValueError:
            pass
        else:
            raise AssertionError("Bridge连续测量未拒绝非法值")
    with patch("models.CLIPModel.from_pretrained", return_value=SimpleNamespace(vision_model=Encoder(), text_model=Encoder())):
        try:
            RobotAdapterModel({"model": {**config["model"], "gripper_target_mode": "transition",
                "current_gripper_encoding": "bridge_measured_affine_unbounded_v1"}})
        except ValueError:
            pass
        else:
            raise AssertionError("测量观测被误当成transition初始命令")
    print("[Bridge输入/末步检查] 通过：连续测量保真、目标仍8维命令，末步不传播，非法测量/transition混用拒绝")
    from audit_bcz import inspect_command_neighborhood
    neighborhood = inspect_command_neighborhood(fields, 0)
    assert len(neighborhood["neighboring_observations"]) == 1
    assert len(neighborhood["waypoint_pose_matches"]) == 10
    assert neighborhood["neighboring_observations"][0]["native_target_closes"] == closes.tolist()
    for invalid_start in (-1, 1):
        try:
            inspect_command_neighborhood(fields, invalid_start)
        except ValueError:
            pass
        else:
            raise AssertionError("原始邻域越界输入未被拒绝")
    print("[原始邻域检查] 通过：短轨迹不越界，10个字段原样保留；近邻不当作时间标定")
    from train import frozen_policy_digest, GRIPPER_PARAMETER_PREFIXES
    from train import load_gripper_fit_weights, trainable_state_dict
    from models import build_gripper_readout
    readout_fixture = torch.nn.Module()
    readout_fixture.simple_fusion = torch.nn.Linear(2, 2)
    readout_fixture.gripper_head = build_gripper_readout(16, "legacy")
    source_weights = {key: value.clone() for key, value in trainable_state_dict(readout_fixture).items()}
    assert set(readout_fixture.gripper_head.state_dict()) == {"0.weight", "0.bias", "2.weight", "2.bias"}
    for head_type in ("legacy", "mlp2"):
        readout_fixture.gripper_head = build_gripper_readout(16, head_type)
        load_gripper_fit_weights(readout_fixture, source_weights, reset_readout=True, seed=42)
        first_reset = {key: value.clone() for key, value in readout_fixture.gripper_head.state_dict().items()}
        load_gripper_fit_weights(readout_fixture, source_weights, reset_readout=True, seed=42)
        assert all(torch.equal(value, first_reset[key]) for key, value in readout_fixture.gripper_head.state_dict().items())
        assert torch.equal(readout_fixture.simple_fusion.weight, source_weights["simple_fusion.weight"])
        logits = readout_fixture.gripper_head(torch.randn(4, 10, 16))
        assert logits.shape == (4, 10, 1)
        logits.square().mean().backward()
        assert all(parameter.grad is not None for parameter in readout_fixture.gripper_head.parameters())
    try:
        load_gripper_fit_weights(readout_fixture, source_weights, reset_readout=False)
    except ValueError:
        pass
    else:
        raise AssertionError("不同结构未显式重置却被允许载入")
    print("[夹爪结构检查] 通过：旧头兼容、两种头前向反向、确定性重置及异构权重拒绝")
    digest_fixture = torch.nn.Module()
    digest_fixture.simple_fusion = torch.nn.Linear(2, 2)
    digest_fixture.regression_head = torch.nn.Linear(2, 2)
    digest_fixture.gripper_head = torch.nn.Linear(2, 1)
    for name, parameter in digest_fixture.named_parameters():
        parameter.requires_grad_(name.startswith(GRIPPER_PARAMETER_PREFIXES))
    before_digest = frozen_policy_digest(digest_fixture)
    head_before = digest_fixture.gripper_head.weight.detach().clone()
    head_optimizer = torch.optim.SGD([p for p in digest_fixture.parameters() if p.requires_grad], lr=.01)
    head_optimizer.zero_grad()
    digest_fixture.gripper_head(torch.ones(1, 2)).square().sum().backward()
    head_optimizer.step()
    assert frozen_policy_digest(digest_fixture) == before_digest
    assert not torch.equal(head_before, digest_fixture.gripper_head.weight)
    assert digest_fixture.simple_fusion.weight.grad is None and digest_fixture.regression_head.weight.grad is None
    with torch.no_grad():
        digest_fixture.regression_head.weight.add_(.1)
    assert frozen_policy_digest(digest_fixture) != before_digest
    print("[仅夹爪隔离检查] 通过：头部可更新，图文/位姿无梯度且不变，意外位姿变更能检测")
    from train import validate_split_indices
    from train import parse_args as parse_train_args, validate_native_experiment
    pool_args = parse_train_args(["@experiments/bcz_native_pool_smoke.args"])
    validate_native_experiment(pool_args)
    assert pool_args.offline_command_experiment and not pool_args.overfit_samples
    for field, value in (("overfit_samples", 32), ("gripper_only_fit_from", "old.pt"),
                         ("init_from", "old.pt"), ("resume", "old.pt"),
                         ("chunk_size", 16), ("balanced_gripper_loss", True),
                         ("fusion_type", "cross_attention"), ("split_manifest", None)):
        invalid_args = parse_train_args(["@experiments/bcz_native_pool_smoke.args"])
        setattr(invalid_args, field, value)
        try:
            validate_native_experiment(invalid_args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"原生完整池危险混用未被拒绝：{field}")
    validate_native_experiment(parse_train_args(["@experiments/bcz_native_sequence_fit.args"]))
    probe_fit_args = parse_train_args(["@experiments/bcz_pool_seen_probe_fit.args"])
    validate_native_experiment(probe_fit_args)
    assert probe_fit_args.overfit_samples == 52 and probe_fit_args.epochs * probe_fit_args.max_steps_per_epoch == 400
    rotation_args = parse_train_args(["@experiments/bcz_pool_seen_probe_rotation4.args"])
    validate_native_experiment(rotation_args)
    rotation_changed = {key for key, value in vars(probe_fit_args).items() if value != vars(rotation_args)[key]}
    assert rotation_changed == {"output_dir", "regression_rotation_weight"}, rotation_changed
    for value in (0, -1, float("nan"), float("inf")):
        invalid_weight = parse_train_args(["@experiments/bcz_pool_seen_probe_rotation4.args"])
        invalid_weight.regression_rotation_weight = value
        try:
            validate_native_experiment(invalid_weight)
        except ValueError:
            pass
        else:
            raise AssertionError("接受了非法旋转权重")
    for field, value in (("offline_command_experiment", True), ("init_from", "old.pt"),
                         ("resume", "old.pt"), ("overfit_manifest", "other.json"),
                         ("balanced_sampling", True), ("overfit_samples", 0)):
        invalid_probe = parse_train_args(["@experiments/bcz_pool_seen_probe_fit.args"])
        setattr(invalid_probe, field, value)
        try:
            validate_native_experiment(invalid_probe)
        except ValueError:
            pass
        else:
            raise AssertionError(f"完整池已见拟合入口接受危险混用：{field}")
    adapter_fit_args = parse_train_args(["@experiments/bcz_native_adapter_fit.args"])
    validate_native_experiment(adapter_fit_args)
    assert adapter_fit_args.overfit_samples == 32 and adapter_fit_args.audit_first_update
    assert adapter_fit_args.dropout == 0 and adapter_fit_args.epochs * adapter_fit_args.max_steps_per_epoch == 400
    assert adapter_fit_args.fusion_type == "cross_attention" and adapter_fit_args.adapter_pooling == "cls_patch_mean"
    low_lr_args = parse_train_args(["@experiments/bcz_native_adapter_fit_lr_low.args"])
    validate_native_experiment(low_lr_args)
    changed_fields = {key for key, value in vars(adapter_fit_args).items() if value != vars(low_lr_args)[key]}
    assert changed_fields == {"output_dir", "learning_rate"}, changed_fields
    assert abs(low_lr_args.learning_rate - adapter_fit_args.learning_rate / 10) < 1e-12
    split_lr_args = parse_train_args(["@experiments/bcz_native_adapter_fit_lr_split.args"])
    validate_native_experiment(split_lr_args)
    split_changed_fields = {key for key, value in vars(low_lr_args).items() if value != vars(split_lr_args)[key]}
    assert split_changed_fields == {"output_dir", "gripper_learning_rate"}, split_changed_fields
    assert split_lr_args.gripper_learning_rate == adapter_fit_args.learning_rate
    split_pool_args = parse_train_args(["@experiments/bcz_native_pool_adapter_lr_split.args"])
    original_pool_args = parse_train_args(["@experiments/bcz_native_pool_adapter.args"])
    validate_native_experiment(split_pool_args)
    pool_changed = {key for key, value in vars(original_pool_args).items() if value != vars(split_pool_args)[key]}
    assert pool_changed == {"output_dir", "learning_rate", "gripper_learning_rate"}, pool_changed
    assert split_pool_args.gripper_learning_rate == original_pool_args.learning_rate
    assert split_pool_args.epochs == 3 and split_pool_args.max_steps_per_epoch == 0
    adapter_args = parse_train_args(["@experiments/bcz_native_pool_adapter_smoke.args"])
    validate_native_experiment(adapter_args)
    assert adapter_args.adapter_layers == 8 and adapter_args.attention_dim == 512
    for field, value in (("adapter_pooling", "cls"), ("fusion_type", "simple_concat"),
                         ("native_adapter_comparison", False), ("decoder_type", "diffusion"),
                         ("overfit_samples", 32), ("resume", "old.pt")):
        invalid_adapter = parse_train_args(["@experiments/bcz_native_pool_adapter_smoke.args"])
        setattr(invalid_adapter, field, value)
        try:
            validate_native_experiment(invalid_adapter)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Adapter对照接受了非受控配置：{field}")
    print("[Adapter实验隔离] 通过：显式patch读出合法；隐式融合切换/目标变更/旧权重混用被拒绝")
    print("[原生模式隔离] 通过：完整池与旧拟合入口合法；混用诊断、迁移、语义与重采样被拒绝")
    class SplitFixture:
        def __len__(self):
            return 6
        def group_key(self, index):
            return index // 2
    fixture = SplitFixture()
    validate_split_indices(fixture, {"train": [0, 1], "validation": [2, 3], "test": [4, 5]})
    for invalid in (
        {"train": [0, 2], "validation": [1, 3], "test": [4, 5]},
        {"train": [0, 1], "validation": [2, 3], "test": [4]},
        {"train": [0, 1], "validation": [2, 3], "test": [4, 4, 5]},
    ):
        try:
            validate_split_indices(fixture, invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("完整轨迹泄漏/遗漏/重复没有被拒绝")
    print("[划分检查] 通过：完整轨迹隔离；跨集合泄漏、遗漏与重复会被拒绝")
    from evaluate_offline import metric_coverage
    coverage = metric_coverage(fixture, [5, 4, 3, 2, 1, 0], 4, 2, 2)
    assert coverage["sampled_indices"] == [5, 4, 1, 0]
    assert coverage["sampled_windows"] == 4 and coverage["sampled_episodes"] == 2
    assert coverage["available_episodes"] == 3
    assert metric_coverage(fixture, [5, 4, 3, 2, 1, 0], 4, 1, 8)["sampled_windows"] == 4
    from evaluate_offline import select_episode_windows
    from evaluate_offline import diagnose_conditions
    from types import SimpleNamespace
    diagnostic_fixture = SimpleNamespace(bcz_target="native_commands", chunk_size=10)
    diagnostic_args = SimpleNamespace(checkpoint="fixture.pt", seed=42)
    for invalid_fixed in ({"test": [4, 5]}, {"validation": [0, 1]}, {"validation": [2, 2]}, {}):
        try:
            diagnose_conditions(SimpleNamespace(), None, diagnostic_fixture,
                                {"train": [0, 1], "validation": [2, 3], "test": [4, 5]},
                                torch.device("cpu"), diagnostic_args, fixed_partitions=invalid_fixed)
        except ValueError:
            pass
        else:
            raise AssertionError("固定条件诊断接受了测试/跨分区/重复/空探针")
    print("[验证探针隔离] 通过：固定诊断拒绝测试、跨分区、重复或空窗口")
    selected = select_episode_windows(fixture, list(range(6)), 1, 42)
    assert len(selected) == 3 and len({fixture.group_key(i) for i in selected}) == 3
    assert selected == select_episode_windows(fixture, list(range(6)), 1, 42)
    assert select_episode_windows(fixture, list(range(6)), 0, 42) == list(range(6))
    from train import IndexedTrainingSubset, collate_indexed_training
    sample = ("fixture", torch.zeros(3, 224, 224), torch.ones(1), torch.zeros(1, 8), torch.ones(1, 8))
    indexed = IndexedTrainingSubset([sample] * 6, [5, 2, 0])
    tracked = collate_indexed_training([indexed[0], indexed[2]])
    assert tracked[5] == [5, 0] and tracked[1].shape[0] == 2
    from train import select_balanced_overfit_samples
    import tempfile
    class BalancedFixture(SplitFixture):
        bcz_target = "first_command"
        def __getitem__(self, index):
            target = torch.tensor([[0., 0., 0., 0., 0., 0., 1., 1. if index % 2 == 0 else -1.]])
            return (str(index), torch.zeros(3, 2, 2), torch.ones(1), target, torch.ones_like(target))
    with tempfile.TemporaryDirectory(prefix="bcz_balanced_check_") as check_dir:
        from pathlib import Path
        chosen, decoded = select_balanced_overfit_samples(BalancedFixture(), list(range(6)), 4, 2, Path(check_dir))
        assert len(chosen) == 4 and len(set(chosen)) == 4
        assert sum(bool(item[3][0, 7] >= 0) for item in decoded) == 2
        class ConflictingFixture(BalancedFixture):
            def __getitem__(self, index):
                item = super().__getitem__(index)
                return ("same", *item[1:])
        try:
            select_balanced_overfit_samples(ConflictingFixture(), list(range(6)), 4, 2, Path(check_dir))
        except ValueError:
            pass
        else:
            raise AssertionError("精确相同输入的冲突标签未被拒绝")
    print("[平衡拟合检查] 通过：训练样本各类数量固定，冲突输入不会被静默删除")
    print("[覆盖检查] 通过：按实际batch抽样记录窗口与独立演示，不将重叠窗口当独立样本")
    print(f"[数学检查] 通过：线性终点={old_terminal:.6f}，余弦终点={new_terminal:.9f}；理想去噪/位姿往返/回归确定性/四元数符号不变损失/梯度通过")


def run_checks(args: argparse.Namespace) -> None:
    if args.model_math_only:
        check_model_math()
        return
    sources = tuple(source.strip() for source in args.sources.split(",") if source.strip())
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
    print("[检查 1/3] 扫描数据并识别文件结构")
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
    )
    indices_by_schema: dict[str, list[int]] = defaultdict(list)
    for index, sample in enumerate(dataset.samples):
        schema = str(sample["source"])
        indices_by_schema[schema].append(index)

    total_checked = sum(
        min(args.check_samples, len(indices)) for indices in indices_by_schema.values()
    )
    print(
        f"[检查 2/3] 共识别 {len(indices_by_schema)} 种 schema；"
        f"每种最多验证 {args.check_samples} 个样本"
    )
    decoded: dict[
        int, tuple[str, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ] = {}
    for schema, indices in indices_by_schema.items():
        check_indices = indices[: args.check_samples]
        stats_indices = indices[: args.stats_samples]
        needed_indices = list(dict.fromkeys(check_indices + stats_indices))
        print(f"[检查] {schema}：验证 {len(check_indices)} 个样本")
        for index in needed_indices:
            decoded[index] = dataset[index]
        for index in check_indices:
            validate_sample(*decoded[index], chunk_size=args.chunk_size)

    print("[检查 3/3] 数据结构检查通过")
    print(
        f"  动作语义：{ACTION_REPRESENTATION}；"
        f"前三维 × {POSITION_SCALE_METERS:.3f}m = 工具坐标系相对位移"
    )
    print(f"  实际验证样本总数：{total_checked}")
    print(f"  每种 schema 保留的检查样本：{dict(dataset.source_counts)}")
    print(f"  扫描发现的全部片段：{dict(dataset.discovered_counts)}")

    print("\n  相对动作范围抽样：")
    for schema, indices in indices_by_schema.items():
        stats_indices = indices[: args.stats_samples]
        action_values = torch.cat([decoded[index][3] for index in stats_indices], dim=0)
        displacement_cm = action_values[:, :3] * POSITION_SCALE_METERS * 100.0
        xyz_min = displacement_cm.amin(dim=0).tolist()
        xyz_max = displacement_cm.amax(dim=0).tolist()
        xyz_mean = displacement_cm.mean(dim=0).tolist()
        open_ratio = float((action_values[:, 7] > 0).float().mean().item())
        print(
            f"    {schema}: samples={len(stats_indices)}, "
            f"local_delta_cm_min={[round(value, 2) for value in xyz_min]}, "
            f"max={[round(value, 2) for value in xyz_max]}, "
            f"mean={[round(value, 2) for value in xyz_mean]}, "
            f"gripper_open={open_ratio:.1%}"
        )

    # 每种 schema 单独展示一个样本，避免只看到目录排序最靠前的数据。
    for schema, indices in indices_by_schema.items():
        index = indices[0]
        sample = dataset.samples[index]
        if index not in decoded:
            decoded[index] = dataset[index]
        instruction, image, current_gripper, actions, supervision_mask = decoded[index]
        source_path = sample.get("file_path", sample.get("action_path", "unknown"))
        start = int(sample.get("start_index", 0))
        trajectory = sample.get("demo_key", sample.get("record_index", "unknown"))
        print(f"\n  [{schema}] 示例")
        print(f"    文件：{source_path}")
        print(f"    轨迹：{trajectory}，输入图像时间步：{start}")
        print(f"    指令：{instruction!r}")
        print(f"    图像形状：{tuple(image.shape)}")
        print(f"    动作窗口形状：{tuple(actions.shape)}")
        print(f"    输入时刻夹爪：{current_gripper.item():+.0f}")
        supervised = torch.where(supervision_mask[0] > 0.5)[0].tolist()
        print(f"    真实监督维度索引：{supervised}")
        print(f"    第一个动作来源：输入时间步 {start} 之后的目标时间步 {start + 1}")
        print(
            "    第一个目标 [归一化局部位移, 相对xyzw, gripper]："
            f"{actions[0].tolist()}"
        )


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-math-only", action="store_true", help="只验证扩散和位姿数学，不读数据、不下载CLIP。")
    parser.add_argument(
        "--dataset-dir",
        default=str(project_root / "training_cache" / "oxe_core"),
    )
    parser.add_argument("--sources", default="auto")
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument(
        "--exclude-path-parts",
        default=(
            "LIBERO,bridge,bridge_data_msr,old1.0.1,"
            "language_table_sim*,language_table*oracle*,"
            "VIOLA-dataset,fanuc_manipulation,cliport"
        ),
        help="逗号分隔的路径段或通配模式；默认检查导师指定的训练输入。",
    )
    parser.add_argument(
        "--tfrecord-splits",
        default="train",
        help="逗号分隔的 TFRecord split；默认只检查 train。",
    )
    parser.add_argument(
        "--exclude-schemas",
        default="",
        help="逗号分隔的 schema；加载器检查默认不排除，便于检查全部已识别格式。",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="全局样本上限；通常留空，避免在第一种文件格式处提前停止。",
    )
    parser.add_argument(
        "--max-samples-per-schema",
        type=int,
        default=32,
        help="每种 schema 最多保留多少个样本用于检查。",
    )
    parser.add_argument(
        "--check-samples",
        type=int,
        default=4,
        help="每种 schema 实际解码并验证多少个样本。",
    )
    parser.add_argument(
        "--stats-samples",
        type=int,
        default=8,
        help="每种 schema 使用多少个代表样本估计动作范围。",
    )
    parser.add_argument("--max-tfrecord-episodes", type=int)
    parser.add_argument("--max-tfrecord-episodes-per-schema", type=int)
    parser.add_argument("--min-trajectory-steps", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    run_checks(parse_args())

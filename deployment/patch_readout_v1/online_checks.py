"""A3在线接线的本机检查（不启动LIBERO仿真，不训练）。

运行（Windows PowerShell）：
  cd D:\\ntu_related\\dissertation\\Zero_shot\\deployment\\patch_readout_v1
  D:\\ntu_related\\dissertation\\Zero_shot\\training_cache\\libero_native_env_v1\\Scripts\\python.exe online_checks.py

检查内容：
 1. 位姿差公式：已知平移/旋转能被准确还原。
 2. 上一条命令编码：由航点得到的command8与Bridge原始命令经encode_commands得到的完全一致(含裁剪)。
 3. A3 checkpoint 能严格加载(EMA权重、无缺失/多余键)。
 4. 真实冻结DINO/CLIP + A3在CPU上连续调用3次：第1次无历史，执行满4步后第2次有历史；
    输出有限、单位四元数、二值夹爪；并报告历史输入对第1个航点的影响大小。
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import online_policy as op  # noqa: E402

CHECKPOINT = ROOT / 'reports/patch_readout_239393/A3_policy/policy.pt'
CLIP = ROOT.parent / 'hf_cache/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41'
DINO = ROOT / 'baselines/dinov2_large_frozen_assets_20261023'
MAPPING = ROOT / 'configs/libero_bridge_pose_mapping_uncalibrated_engineering_20261030.json'


def test_state_delta():
    p0, q0 = np.array([.3, -.1, .2]), Rotation.from_euler('xyz', [.1, -.2, .3]).as_quat()
    turn = Rotation.from_rotvec([.02, -.01, .03])
    p1, q1 = p0 + [.01, -.02, .005], (turn * Rotation.from_quat(q0)).as_quat()
    d = op.state_delta6(p0, q0, p1, q1)
    assert np.allclose(d[:3], [.1, -.2, .05]) and np.allclose(d[3:], [.02, -.01, .03], atol=1e-9), d
    print('[1] state delta (position/0.1, left rotvec)  OK')


def test_command_encoding():
    from bridge_command8_targets import encode_commands
    rng = np.random.default_rng(0)
    raw = np.c_[rng.uniform(-.08, .08, (200, 3)), rng.uniform(-.4, .4, (200, 3)), rng.integers(0, 2, 200)]
    expected, _, info = encode_commands(raw, op.COMMAND_CLIP, op.POSITION_DIVISOR)
    assert info['motion_components_clipped'] > 0       # 覆盖裁剪分支
    worst = 0.
    for r, e in zip(raw, expected):
        waypoint = np.r_[r[:3] / op.POSITION_DIVISOR, Rotation.from_euler('xyz', r[3:6]).as_quat(), 2 * r[6] - 1]
        worst = max(worst, float(np.abs(op.command8_from_waypoint(waypoint) - e).max()))
    assert worst < 1e-5, worst
    print(f'[2] previous-command encoding matches encode_commands (max diff {worst:.1e})  OK')


def test_load():
    model, saved = op.load_model(CHECKPOINT)
    n = sum(p.numel() for p in model.parameters())
    assert saved['completed_steps'] == 10000 and n == 133103528, (saved['completed_steps'], n)
    print(f'[3] A3 checkpoint strict load, {n} params, steps={saved["completed_steps"]}  OK')


def test_policy_cpu():
    torch.manual_seed(0)
    mapping = json.loads(MAPPING.read_text(encoding='utf-8'))
    policy = op.load_policy(CHECKPOINT, clip_dir=CLIP, dino_dir=DINO, mapping=mapping, device='cpu')
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)
    q = Rotation.from_euler('xyz', [np.pi, 0, 0]).as_quat()
    request = {'rgb_upright': rgb, 'instruction': 'close the top drawer of the cabinet and put the black bowl on top of it',
               'reference_position_m': np.array([-.1, .2, 1.0]), 'reference_quaternion_xyzw': q,
               'current_open_command': None}
    outputs, history = [], []
    for k in range(3):
        out = policy(request)
        assert out.shape == (16, 8) and np.isfinite(out).all()
        assert np.allclose(np.linalg.norm(out[:, 3:7], axis=1), 1, atol=1e-3) and np.isin(out[:, 7], [-1, 1]).all()
        history.append((policy.last_inference['state_delta_available'], policy.last_inference['prev_command_available']))
        outputs.append(out)
        policy.mark_executed(4)
        request = dict(request, current_open_command=float(out[0, 7]),
                       reference_position_m=request['reference_position_m'] + 0.1 * out[0, :3] * .5)
    assert history == [(False, False), (True, True), (True, True)], history
    # 同一输入、同一采样种子，历史开/关对比：看历史是否真的进入了模型
    saved_previous, calls = dict(policy.previous), policy.calls
    with_hist = policy(request)
    policy.previous, policy.calls = dict(saved_previous, native_steps=0), calls
    no_hist = policy(request)
    diff = float(np.linalg.norm(no_hist[0, :3] - with_hist[0, :3]))
    print(f'[4] CPU policy 3 calls OK; history availability {history}; '
          f'first-waypoint xyz with vs without history differs by {diff:.3f} (x0.1 m)')
    print('    call0 first waypoint:', np.round(outputs[0][0], 3).tolist())


if __name__ == '__main__':
    test_state_delta()
    test_command_encoding()
    test_load()
    test_policy_cpu()
    print('ALL ONLINE CHECKS PASSED')

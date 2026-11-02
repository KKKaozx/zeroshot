"""本机CPU检查（不需要GPU、编码器权重或真实数据）。临时文件写在本目录 .tmp_tests 下，不写C盘。

运行（Windows PowerShell）：
  cd D:\\ntu_related\\dissertation\\Zero_shot\\deployment\\patch_readout_v2
  D:\\ntu_related\\dissertation\\Zero_shot\\training_cache\\libero_native_env_v1\\Scripts\\python.exe cpu_tests.py

检查内容：
 1. 旧CrossAttentionAdapter(pooling='cls')对patch token的梯度恒为0；新读出不为0。
 2. 本体输入关闭时，条件向量与夹爪状态/绝对位姿完全无关；B2只受位姿差影响；B1不建本体分支。
 3. 图像增强：尺寸/类型不变、同种子可复现、确实改变了画面。
 4. 旋转向量/四元数 torch 与 numpy 实现一致。
 5. 合成任务(v2多版本缓存格式)：动作只取决于"亮块"位置。开发集上，不用任何本体输入的spatial读出能学会，
    CLS读出学不会；augment=true 的训练路径也能跑通并学会。
    步数1200：400步时连不加任何改动的spatial也随种子不稳定(0.041/0.195/0.100)，加本体分支后更慢，
    1200步时 state_delta 配置可学会(0.047)。走通 cached_data -> train_cached -> fair_metrics -> summarize_runs。
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'code'))
TMP = HERE / '.tmp_tests'
NO_PROPRIO = {'use_pose18': False, 'use_current2': False}


def small_cfg(**extra):
    cfg = {'readout': 'spatial', 'grid': 4, 'num_adapter_layers': 2, 'attention_dim': 16, 'num_heads': 4,
           'dropout': 0.0, 'readout_dim': 16, 'num_queries': 2, 'num_keypoints': 4, 'keypoint_feat_dim': 8,
           'context_dim': 64, 'down_dims': [32, 64, 128], 'history': []}
    cfg.update(extra)
    return cfg


def small_batch(n=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g)
    return {'visual': r(n, 17, 32), 'text': r(n, 5, 16), 'text_mask': torch.ones(n, 5, dtype=torch.bool),
            'current2': r(n, 2), 'pose18': r(n, 18), 'state_delta': r(n, 6), 'state_delta_mask': torch.ones(n),
            'prev_command': r(n, 7), 'prev_command_mask': torch.ones(n)}


def test_old_cls_bottleneck():
    from adapter import CrossAttentionAdapter
    from patch_policy import PatchPolicy
    torch.manual_seed(0)
    old = CrossAttentionAdapter(num_layers=2, embed_dim=32, text_dim=16, attention_dim=16, num_heads=4, dropout=0.0)
    vis = torch.randn(2, 17, 32, requires_grad=True)
    text, mask = torch.randn(2, 5, 16), torch.ones(2, 5, dtype=torch.bool)
    old(vis, text, mask).sum().backward()
    g_old = vis.grad[:, 1:].abs().max().item()
    assert g_old == 0.0, g_old
    g_new = {}
    for readout in ('patch', 'spatial'):
        new = PatchPolicy(small_cfg(readout=readout), n_tokens=17, vis_dim=32, text_dim=16)
        b = small_batch()
        b['visual'] = b['visual'].clone().requires_grad_(True)
        new.context(b).sum().backward()
        g_new[readout] = b['visual'].grad[:, 1:].abs().max().item()
        assert g_new[readout] > 0, (readout, g_new[readout])
    print(f'[1] old CLS adapter patch-gradient max={g_old}; new readout patch-gradient max='
          f'patch {g_new["patch"]:.3e} / spatial {g_new["spatial"]:.3e}  OK')


def test_proprio_switches():
    from patch_policy import PatchPolicy
    torch.manual_seed(0)
    b1 = PatchPolicy(small_cfg(**NO_PROPRIO), n_tokens=17, vis_dim=32, text_dim=16).eval()
    assert b1.proprio is None
    b2 = PatchPolicy(small_cfg(**NO_PROPRIO, history=['state_delta']), n_tokens=17, vis_dim=32, text_dim=16).eval()
    assert b2.proprio[0].in_features == 7
    base = small_batch()
    for model in (b1, b2):
        changed = dict(base, current2=base['current2'] * 9 + 3, pose18=base['pose18'] * 9 + 3,
                       prev_command=base['prev_command'] * 9 + 3)
        assert torch.equal(model.context(base), model.context(changed)), 'gripper/pose/prev-command must be ignored'
    moved = dict(base, state_delta=base['state_delta'] + 1)
    assert not torch.equal(b2.context(base), b2.context(moved)), 'B2 must use state_delta'
    assert torch.equal(b1.context(base), b1.context(moved)), 'B1 must ignore state_delta'
    full = PatchPolicy(small_cfg(history=['state_delta', 'prev_command']), n_tokens=17, vis_dim=32, text_dim=16)
    assert full.proprio[0].in_features == 2 + 18 + 7 + 8, 'v1 (A3) input layout must be unchanged'
    print('[2] B1 has no proprio branch; B1/B2 ignore gripper state, pose18, previous command; B2 uses state_delta; '
          'A3 layout unchanged  OK')


def test_augment():
    from augment import augment_image, view_rng
    rgb = np.random.default_rng(0).integers(0, 256, (256, 256, 3), dtype=np.uint8)
    a = augment_image(rgb, view_rng(1, 5, 1))
    b = augment_image(rgb, view_rng(1, 5, 1))
    c = augment_image(rgb, view_rng(1, 5, 2))
    assert a.shape == rgb.shape and a.dtype == np.uint8
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    assert np.abs(a.astype(int) - rgb).mean() > 5
    print('[3] augmentation keeps shape/dtype, is reproducible per (seed, window, view), changes the image  OK')


def test_geometry():
    from geometry import rotvec_to_quat, torch_rotvec_to_quat
    r = np.random.default_rng(0).normal(size=(500, 3)) * 0.8
    r[:5] *= 1e-9
    a = rotvec_to_quat(r)
    b = torch_rotvec_to_quat(torch.from_numpy(r)).numpy()
    assert np.allclose(a, b, atol=1e-6)
    print('[4] torch/numpy rotvec->quat agree  OK')


def make_cache(path, n_train=512, n_dev=48, views=3, seed=0, noise=0.5):
    """v2格式：直接生成已池化的 CLS+8x8 token；版本k在原图上加额外噪声，模拟增强。"""
    rng = np.random.default_rng(seed)
    n = n_train + n_dev
    path.mkdir(parents=True)
    cells = rng.integers(0, 8, size=(n, 2))
    direction = rng.normal(size=1024).astype(np.float32)
    direction /= np.linalg.norm(direction)
    vis = np.lib.format.open_memmap(path / 'visual_views.npy', mode='w+', dtype=np.float16, shape=(n, views, 65, 1024))
    for lo in range(0, n, 128):
        m = min(n, lo + 128) - lo
        # 与v1等价：v1在16x16上加噪声N(0,noise²)再2x2平均池化，池化后噪声标准差为noise/2
        block = rng.normal(size=(m, 65, 1024)).astype(np.float32) * (noise / 2)
        for i, (r, c) in enumerate(cells[lo:lo + m]):
            block[i, 1 + r * 8 + c] += 4.0 * direction
        for v in range(views):
            # 温和扰动，只检查训练路径；真实增强的强度在本机真实缓存上另行检查
            extra = 0 if v == 0 else rng.normal(size=block.shape).astype(np.float32) * 0.1
            vis[lo:lo + m, v] = (block + extra).astype(np.float16)
    vis.flush()
    del vis
    np.save(path / 'text_tokens.npy', rng.normal(size=(2, 77, 768)).astype(np.float16))
    tm = np.zeros((2, 77), bool)
    tm[:, :6] = True
    np.save(path / 'text_mask.npy', tm)
    tgt = np.zeros((n, 16, 8), np.float32)
    tgt[..., 0] = ((cells[:, 1] - 3.5) / 3.5 * 0.3)[:, None]
    tgt[..., 1] = ((cells[:, 0] - 3.5) / 3.5 * 0.3)[:, None]
    tgt[..., 2] = rng.normal(size=(n, 16)) * 0.02
    tgt[..., 6] = 1.0
    tgt[..., 7] = np.where(cells[:, :1] > 3, 1.0, -1.0)
    mask = np.ones_like(tgt)
    role = np.r_[np.zeros(n_train), np.ones(n_dev)].astype(np.int8)
    eval_flag = np.zeros(n, bool)
    eval_flag[:32] = True
    eval_flag[role == 1] = True
    # 本体输入故意填成与目标强相关的值：关闭后不应被用到
    leak = np.c_[tgt[:, 0, :2], np.zeros((n, 16))].astype(np.float32)
    np.savez(path / 'numeric.npz', command8_targets=tgt, command8_masks=mask, future_targets=tgt, future_masks=mask,
             pose18=leak, current2=tgt[:, 0, :2].copy(),
             # 位姿差用常数：每个样本不同的随机值等于样本编号，模型会去死记编号而不学亮块规则
             state_delta=np.full((n, 6), 0.1, np.float32), state_delta_mask=np.ones(n, np.float32),
             prev_command=tgt[:, 0].copy(), prev_command_mask=np.zeros(n, np.float32),
             role=role, window_index=np.arange(n), eval_flag=eval_flag,
             train_mean_command=tgt[role == 0].mean((0, 1)), text_index=(np.arange(n) % 2).astype(np.int32))
    (path / 'manifest.json').write_text(json.dumps({'format': 'patch-readout-bridge-cache-v2', 'grid': 8}), encoding='utf-8')


def run_config(cache, name, readout, augment=False, history=()):
    cfg = {'name': name, 'target': 'command8', 'augment': augment,
           'policy': {'readout': readout, 'grid': 8, 'num_adapter_layers': 1, 'history': list(history),
                      'down_dims': [64, 128, 256], 'readout_layers': 1, 'dropout': 0.0, **NO_PROPRIO}}
    cpath = TMP / f'{name}.json'
    cpath.write_text(json.dumps(cfg), encoding='utf-8')
    out = TMP / 'runs' / name
    subprocess.run([sys.executable, str(HERE / 'train_cached.py'), '--cache', str(cache), '--config', str(cpath),
                    '--out', str(out), '--steps', '1200', '--batch', '32', '--lr', '3e-4', '--warmup', '50',
                    '--eval-every', '600', '--device', 'cpu', '--time-cap', '1800'], check=True)
    rep = json.loads((out / 'report.json').read_text(encoding='utf-8'))
    return rep['results']['validation']['metrics']['point']['first_position_normalized'], \
        rep['results']['validation']['metrics']['zero']['first_position_normalized']


def test_synthetic_learning():
    if TMP.exists():
        shutil.rmtree(TMP)
    TMP.mkdir()
    cache = TMP / 'cache'
    make_cache(cache)
    spatial_err, zero = run_config(cache, 'spatial', 'spatial')
    aug_err, _ = run_config(cache, 'spatial_aug_delta', 'spatial', augment=True, history=['state_delta'])
    cls_err, _ = run_config(cache, 'cls', 'cls')
    subprocess.run([sys.executable, str(HERE / 'summarize_runs.py'), str(TMP / 'runs')], check=True)
    print(f'[5] synthetic location task (dev point first-pos): spatial={spatial_err:.3f} '
          f'spatial+aug+delta={aug_err:.3f} cls={cls_err:.3f} zero={zero:.3f}')
    assert spatial_err < 0.6 * cls_err, 'spatial readout should exploit spatial information that CLS cannot see'
    assert spatial_err < 0.6 * zero and aug_err < 0.6 * zero, 'spatial readout should learn the location rule'
    print('[5] OK')


if __name__ == '__main__':
    test_old_cls_bottleneck()
    test_proprio_switches()
    test_augment()
    test_geometry()
    test_synthetic_learning()
    shutil.rmtree(TMP, ignore_errors=True)
    print('ALL CPU TESTS PASSED')

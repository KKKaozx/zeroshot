"""本机CPU检查（不需要GPU、编码器权重或真实数据）。临时文件写在本目录 .tmp_tests 下，不写C盘。

运行（Windows PowerShell）：
  cd D:\\ntu_related\\dissertation\\Zero_shot\\deployment\\patch_readout_v1
  D:\\ntu_related\\dissertation\\Zero_shot\\training_cache\\libero_native_env_v1\\Scripts\\python.exe cpu_tests.py

检查内容：
 1. 旧CrossAttentionAdapter(pooling='cls')对patch token的梯度恒为0（旧瓶颈的直接证明）；新读出不为0。
 2. 旋转向量/四元数 torch 与 numpy 实现一致。
 3. 合成任务：动作只取决于“亮patch”的位置。在开发集上，spatial读出能学会，CLS读出学不会。
    (patch注意力读出在此任务多个种子都学不会，见README)
    同时走通 cached_data -> train_cached -> fair_metrics -> summarize_runs 整条链。
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
        cfg = {'readout': readout, 'grid': 4, 'num_adapter_layers': 2, 'attention_dim': 16, 'num_heads': 4,
               'dropout': 0.0, 'readout_dim': 16, 'num_queries': 2, 'num_keypoints': 4, 'keypoint_feat_dim': 8,
               'context_dim': 64, 'down_dims': [32, 64, 128], 'history': []}
        new = PatchPolicy(cfg, n_tokens=17, vis_dim=32, text_dim=16)
        vis2 = vis.detach().clone().requires_grad_(True)
        b = {'visual': vis2, 'text': text, 'text_mask': mask, 'current2': torch.zeros(2, 2), 'pose18': torch.zeros(2, 18)}
        new.context(b).sum().backward()
        g_new[readout] = vis2.grad[:, 1:].abs().max().item()
        assert g_new[readout] > 0, (readout, g_new[readout])
    print(f'[1] old CLS adapter patch-gradient max={g_old}; new readout patch-gradient max='
          f'patch {g_new["patch"]:.3e} / spatial {g_new["spatial"]:.3e}  OK')


def test_geometry():
    from geometry import rotvec_to_quat, torch_rotvec_to_quat
    r = np.random.default_rng(0).normal(size=(500, 3)) * 0.8
    r[:5] *= 1e-9
    a = rotvec_to_quat(r)
    b = torch_rotvec_to_quat(torch.from_numpy(r)).numpy()
    assert np.allclose(a, b, atol=1e-6)
    print('[2] torch/numpy rotvec->quat agree  OK')


def make_cache(path, n_train=512, n_dev=48, seed=0, noise=0.5):
    # 开发集要有意义：192个训练样本分到64个格子太少，换成512。
    rng = np.random.default_rng(seed)
    n = n_train + n_dev
    path.mkdir(parents=True)
    cells = rng.integers(0, 8, size=(n, 2))          # 8x8 网格中的位置
    direction = rng.normal(size=1024).astype(np.float32)
    direction /= np.linalg.norm(direction)
    vis = np.lib.format.open_memmap(path / 'visual_tokens.npy', mode='w+', dtype=np.float16, shape=(n, 257, 1024))
    for lo in range(0, n, 128):                      # 分块生成，控制内存
        block = rng.normal(size=(min(n, lo + 128) - lo, 257, 1024)).astype(np.float32) * noise
        for i, (r, c) in enumerate(cells[lo:lo + len(block)]):
            for dr in (0, 1):
                for dc in (0, 1):
                    block[i, 1 + (2 * r + dr) * 16 + (2 * c + dc)] += 4.0 * direction
        vis[lo:lo + len(block)] = block.astype(np.float16)
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
    np.savez(path / 'numeric.npz', command8_targets=tgt, command8_masks=mask, future_targets=tgt, future_masks=mask,
             pose18=np.zeros((n, 18), np.float32), current2=np.zeros((n, 2), np.float32),
             state_delta=rng.normal(size=(n, 6)).astype(np.float32), state_delta_mask=np.ones(n, np.float32),
             prev_command=tgt[:, 0].copy(), prev_command_mask=np.zeros(n, np.float32),
             role=role, window_index=np.arange(n), eval_flag=eval_flag,
             train_mean_command=tgt[role == 0].mean((0, 1)), text_index=(np.arange(n) % 2).astype(np.int32))
    (path / 'manifest.json').write_text(json.dumps({'format': 'synthetic-test'}), encoding='utf-8')


def run_config(cache, name, readout):
    cfg = {'name': name, 'target': 'command8',
           'policy': {'readout': readout, 'grid': 8, 'num_adapter_layers': 1, 'history': [],
                      'down_dims': [64, 128, 256], 'readout_layers': 1, 'dropout': 0.0}}
    cpath = TMP / f'{name}.json'
    cpath.write_text(json.dumps(cfg), encoding='utf-8')
    out = TMP / 'runs' / name
    subprocess.run([sys.executable, str(HERE / 'train_cached.py'), '--cache', str(cache), '--config', str(cpath),
                    '--out', str(out), '--steps', '400', '--batch', '32', '--lr', '3e-4', '--warmup', '50',
                    '--eval-every', '200', '--device', 'cpu', '--time-cap', '900'], check=True)
    rep = json.loads((out / 'report.json').read_text(encoding='utf-8'))
    # 看开发集：训练集上比较的是谁更会背样本，而不是谁用上了空间信息
    return rep['results']['validation']['metrics']['point']['first_position_normalized'], \
        rep['results']['validation']['metrics']['zero']['first_position_normalized']


def test_synthetic_learning():
    if TMP.exists():
        shutil.rmtree(TMP)
    TMP.mkdir()
    cache = TMP / 'cache'
    make_cache(cache)
    spatial_err, zero = run_config(cache, 'spatial', 'spatial')
    cls_err, _ = run_config(cache, 'cls', 'cls')
    subprocess.run([sys.executable, str(HERE / 'summarize_runs.py'), str(TMP / 'runs')], check=True)
    print(f'[3] synthetic location task (dev point first-pos): spatial={spatial_err:.3f} cls={cls_err:.3f} zero={zero:.3f}')
    assert spatial_err < 0.6 * cls_err, 'spatial readout should exploit spatial information that CLS cannot see'
    assert spatial_err < 0.6 * zero, 'spatial readout should generalize the location rule beyond the zero command'
    print('[3] OK')


if __name__ == '__main__':
    test_old_cls_bottleneck()
    test_geometry()
    test_synthetic_learning()
    shutil.rmtree(TMP, ignore_errors=True)
    print('ALL CPU TESTS PASSED')

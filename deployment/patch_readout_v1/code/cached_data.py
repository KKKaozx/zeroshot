"""读取build_cache.py的缓存；把全部特征一次性放到GPU，批次只是索引操作。"""
import json
from pathlib import Path

import numpy as np
import torch

from geometry import external8_to_motion6

TRAIN, DEV = 0, 1


def pool_grid(patches, grid):
    """[N,256,D] (16x16) -> [N,grid*grid,D]，平均池化保持空间排列。"""
    n, p, d = patches.shape
    side = int(round(p ** 0.5))
    if side * side != p or side % grid:
        raise ValueError('Patch grid must be square and divisible by target grid')
    k = side // grid
    x = patches.reshape(n, grid, k, grid, k, d).mean(axis=(2, 4))
    return x.reshape(n, grid * grid, d)


class CachedBridge:
    def __init__(self, cache_dir, *, target, grid, device, chunk=1024):
        cache_dir = Path(cache_dir)
        self.manifest = json.loads((cache_dir / 'manifest.json').read_text(encoding='utf-8'))
        z = np.load(cache_dir / 'numeric.npz', allow_pickle=False)
        self.numeric = {k: z[k] for k in z.files}
        if target not in ('command8', 'future_pose'):
            raise ValueError('target must be command8 or future_pose')
        key = 'command8' if target == 'command8' else 'future'
        self.target_name = target
        self.targets8 = self.numeric[key + '_targets'].astype(np.float32)
        self.masks8 = self.numeric[key + '_masks'].astype(np.float32)
        self.role = self.numeric['role']
        self.eval_flag = self.numeric['eval_flag']
        self.n = len(self.role)
        self.train_ids = np.flatnonzero(self.role == TRAIN)
        self.eval_ids = {'train': np.flatnonzero((self.role == TRAIN) & self.eval_flag),
                         'validation': np.flatnonzero(self.role == DEV)}

        # 视觉token：分块读取，必要时池化，fp16上GPU
        vis = np.load(cache_dir / 'visual_tokens.npy', mmap_mode='r')
        parts = []
        for lo in range(0, self.n, chunk):
            block = np.asarray(vis[lo:lo + chunk], dtype=np.float32)
            cls, patches = block[:, :1], block[:, 1:]
            if grid != 16:
                patches = pool_grid(patches, grid)
            parts.append(torch.from_numpy(np.concatenate([cls, patches], axis=1)).half())
        self.visual = torch.cat(parts).to(device)
        self.text = torch.from_numpy(np.load(cache_dir / 'text_tokens.npy').astype(np.float16)).to(device)
        self.text_mask = torch.from_numpy(np.load(cache_dir / 'text_mask.npy')).to(device)
        self.text_index = torch.from_numpy(self.numeric['text_index'].astype(np.int64)).to(device)

        # 内部运动：位置3 + 旋转向量3；只用训练窗有效步拟合均值/标准差
        valid = self.masks8[..., :7].all(-1)
        motion = np.zeros(self.targets8.shape[:2] + (6,), np.float32)
        motion[valid] = external8_to_motion6(self.targets8[valid]).astype(np.float32)
        train_valid = valid & (self.role == TRAIN)[:, None]
        m = motion[train_valid]
        self.motion_mean = m.mean(0)
        self.motion_std = np.maximum(m.std(0), 1e-3)
        self.motion_z = (motion - self.motion_mean) / self.motion_std * valid[..., None]

        # 历史/状态特征同样只用训练窗拟合尺度
        def scaled(x, mask):
            sel = (mask > 0) & (self.role == TRAIN)
            if sel.any():
                mu, sd = x[sel].mean(0), np.maximum(x[sel].std(0), 1e-3)
            else:  # 全部不可用时保持0，避免空均值产生NaN
                mu, sd = np.zeros(x.shape[1]), np.ones(x.shape[1])
            out = np.where((mask > 0)[:, None], (x - mu) / sd, 0.0)
            return out.astype(np.float32), {'mean': mu.tolist(), 'std': sd.tolist()}
        sd6 = self.numeric['state_delta']
        self.state_delta, sd_stats = scaled(sd6, self.numeric['state_delta_mask'])
        pc = self.numeric['prev_command']
        pc6 = np.zeros((self.n, 6), np.float32)
        have = self.numeric['prev_command_mask'] > 0
        pc6[have] = external8_to_motion6(pc[have])
        pc6, pc_stats = scaled(pc6, self.numeric['prev_command_mask'])
        self.prev_command = np.concatenate([pc6, pc[:, 7:8] * have[:, None]], axis=1).astype(np.float32)
        self.feature_stats = {'state_delta': sd_stats, 'prev_command': pc_stats,
                              'motion_mean': self.motion_mean.tolist(), 'motion_std': self.motion_std.tolist()}

        t = lambda x: torch.from_numpy(np.ascontiguousarray(x)).to(device)
        self.dev = {
            'motion_z': t(self.motion_z.astype(np.float32)),
            'motion_valid': t(valid.astype(np.float32)),
            'gripper': t(self.targets8[..., 7]),
            'gripper_mask': t(self.masks8[..., 7]),
            'pose18': t(self.numeric['pose18'].astype(np.float32)),
            'current2': t(self.numeric['current2'].astype(np.float32)),
            'state_delta': t(self.state_delta),
            'state_delta_mask': t(self.numeric['state_delta_mask'].astype(np.float32)),
            'prev_command': t(self.prev_command),
            'prev_command_mask': t(self.numeric['prev_command_mask'].astype(np.float32)),
        }

    def batch(self, ids):
        idx = torch.as_tensor(np.asarray(ids), dtype=torch.long, device=self.visual.device)
        ti = self.text_index[idx]
        b = {k: v[idx] for k, v in self.dev.items()}
        b.update(visual=self.visual[idx].float(), text=self.text[ti].float(), text_mask=self.text_mask[ti])
        return b

    def train_mean_command(self):
        if self.target_name == 'command8':
            return self.numeric['train_mean_command'].astype(np.float32)
        # 未来位姿：训练集逐步均值（只用训练有效步），四元数单位化
        valid = self.masks8[..., :7].all(-1) & (self.role == TRAIN)[:, None]
        mean = np.zeros((16, 8), np.float32)
        for h in range(16):
            v = self.targets8[valid[:, h], h]
            mean[h] = v.mean(0)
        q = mean[:, 3:7]
        mean[:, 3:7] = q / np.linalg.norm(q, axis=1, keepdims=True)
        return mean

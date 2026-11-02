"""四元数(xyzw)与旋转向量互转。内部学习用3维旋转向量，外部仍输出8维(含四元数)。

原外部8维里 qw≈1、std≈0.001，标准化后会把数值噪声放大约百倍；改成旋转向量
后旋转只有3个自由度，与位置一起做6维标准化。四元数统一取 w>=0 半球。
"""
import numpy as np

try:
    import torch
except ImportError:  # 仅在构建缓存的纯NumPy路径中允许
    torch = None


def _normalize(q):
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    if not np.isfinite(q).all() or (n < 1e-12).any():
        raise ValueError('Finite nonzero quaternion required')
    q = q / n
    return np.where(q[..., 3:4] < 0, -q, q)


def quat_to_rotvec(q):
    q = _normalize(q)
    v, w = q[..., :3], np.clip(q[..., 3:4], -1.0, 1.0)
    s = np.linalg.norm(v, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(s, w)
    # 小角度极限 angle/s -> 2/w
    scale = np.where(s > 1e-9, angle / np.maximum(s, 1e-12), 2.0 / w)
    return v * scale


def rotvec_to_quat(r):
    r = np.asarray(r, dtype=np.float64)
    angle = np.linalg.norm(r, axis=-1, keepdims=True)
    half = 0.5 * angle
    scale = np.where(angle > 1e-9, np.sin(half) / np.maximum(angle, 1e-12), 0.5 - angle ** 2 / 48.0)
    q = np.concatenate([r * scale, np.cos(half)], axis=-1)
    return _normalize(q)


def quat_mul(a, b):
    """Hamilton积 a⊗b，xyzw。"""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    ax, ay, az, aw = np.moveaxis(a, -1, 0)
    bx, by, bz, bw = np.moveaxis(b, -1, 0)
    return np.stack([aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw,
                     aw * bw - ax * bx - ay * by - az * bz], axis=-1)


def quat_inv(q):
    q = _normalize(q)
    return np.concatenate([-q[..., :3], q[..., 3:]], axis=-1)


def left_delta(q_new, q_old):
    """世界系左增量 q_new ⊗ q_old^-1，与未来到达位姿标签的'左相对四元数'一致。"""
    return _normalize(quat_mul(_normalize(q_new), quat_inv(q_old)))


def external8_to_motion6(x):
    """[N..,8] 外部格式 -> [N..,6] (位置3 + 旋转向量3)。"""
    x = np.asarray(x, np.float64)
    return np.concatenate([x[..., :3], quat_to_rotvec(x[..., 3:7])], axis=-1)


def torch_rotvec_to_quat(r):
    angle = r.norm(dim=-1, keepdim=True)
    half = 0.5 * angle
    small = angle < 1e-6
    scale = torch.where(small, 0.5 - angle ** 2 / 48.0, torch.sin(half) / angle.clamp_min(1e-12))
    q = torch.cat([r * scale, torch.cos(half)], dim=-1)
    q = q / q.norm(dim=-1, keepdim=True)
    return torch.where(q[..., 3:4] < 0, -q, q)

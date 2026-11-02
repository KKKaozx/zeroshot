"""Versioned causal state features invariant to a rigid change of world frame.

Only current and episode-initial observations are inputs. Future poses remain
action targets. This is a NEW training representation, never a way to relabel
an old checkpoint as already adapted. Tool/TCP and temporal assumptions remain.
"""
import numpy as np
from scipy.spatial.transform import Rotation

RELATIVE_FORMAT = 'episode_initial_relative_pose9_v1'
ABSOLUTE_FORMAT = 'original229116_absolute_pose9_v1'


def absolute_pose6_features(observed, stats):
    """Bit-preserving original RPY formula for the paired training control."""
    v = np.asarray(observed, dtype=np.float64)
    if v.ndim != 2 or v.shape[1] != 6 or not np.isfinite(v).all():
        raise ValueError('Original observed XYZ/RPY rows required')
    xyz = (v[:, :3]-np.asarray(stats['xyz_mean']))/np.asarray(stats['xyz_std'])
    roll, pitch, yaw = v[:, 3], v[:, 4], v[:, 5]
    cr, cp, cy = np.cos(roll), np.cos(pitch), np.cos(yaw)
    sr, sp, sy = np.sin(roll), np.sin(pitch), np.sin(yaw)
    columns = np.stack([cy*cp, sy*cp, -sp, cy*sp*sr-sy*cr, sy*sp*sr+cy*cr, cp*sr], axis=1)
    return np.concatenate([xyz, columns], axis=1).astype('<f4')


def _pose(position, quaternion):
    p, q = np.asarray(position, dtype=np.float64), np.asarray(quaternion, dtype=np.float64)
    if (p.shape[-1:] != (3,) or q.shape != p.shape[:-1] + (4,)
            or not np.isfinite(p).all() or not np.isfinite(q).all()
            or not np.allclose(np.linalg.norm(q, axis=-1), 1, rtol=0, atol=1e-3)):
        raise ValueError('Finite matching XYZ and unit xyzw observations required')
    return p, Rotation.from_quat(q.reshape(-1, 4)).as_matrix().reshape(p.shape[:-1] + (3, 3))


def pose_features(position, quaternion, stats, *, mode, initial_position=None, initial_quaternion=None):
    p, r = _pose(position, quaternion)
    scale = np.asarray(stats['xyz_std'], dtype=np.float64)
    if scale.shape != (3,) or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError('Pinned positive source XYZ scales required; no target fitting')
    if mode == RELATIVE_FORMAT:
        if initial_position is None or initial_quaternion is None:
            raise ValueError('Explicit episode-initial observation required')
        p0, r0 = _pose(initial_position, initial_quaternion)
        try:
            rt = np.swapaxes(r0, -1, -2)
            xyz = np.matmul(rt, (p - p0)[..., None])[..., 0] / scale
            r = np.matmul(rt, r)
        except ValueError as e:
            raise ValueError('Initial/current observation shapes differ') from e
    elif mode == ABSOLUTE_FORMAT:
        mean = np.asarray(stats['xyz_mean'], dtype=np.float64)
        if mean.shape != (3,) or not np.isfinite(mean).all():
            raise ValueError('Pinned source XYZ means required')
        xyz = (p - mean) / scale
    else:
        raise ValueError('Unknown state-feature contract; no inferred checkpoint conversion')
    columns = np.concatenate((r[..., :, 0], r[..., :, 1]), axis=-1)
    result = np.concatenate((xyz, columns), axis=-1).astype('<f4')
    if result.shape != p.shape[:-1] + (9,) or not np.isfinite(result).all():
        raise ValueError('Invalid 9D state features')
    return result


class EpisodeStateReference:
    """Explicit begin_episode required; the anchor never moves during a rollout."""
    def __init__(self, stats, mode):
        if mode not in (RELATIVE_FORMAT, ABSOLUTE_FORMAT):
            raise ValueError('Unsupported state contract')
        self.stats, self.mode = stats, mode
        self.anchor = None

    def begin_episode(self, position, quaternion):
        if np.shape(position) != (3,) or np.shape(quaternion) != (4,):
            raise ValueError('One causal initial observation required')
        _pose(position, quaternion)
        self.anchor = (np.array(position, copy=True), np.array(quaternion, copy=True))

    def __call__(self, position, quaternion, stats):
        if stats != self.stats or self.anchor is None:
            raise ValueError('Bound source statistics and explicit episode reset required')
        return pose_features(position, quaternion, stats, mode=self.mode,
            initial_position=self.anchor[0], initial_quaternion=self.anchor[1])

"""Explicit source-Cartesian reached-pose targets, independent of tool axes.

This is a new action contract, not a conversion for old checkpoints. Coordinates
stay in each source's recorded Cartesian frame; physical alignment between robot
bases, camera views, TCPs, units and sample times is NOT inferred here.
"""
import numpy as np

TOOL_REPRESENTATION = 'tool_relative_pose_open_positive_v2'
SOURCE_REPRESENTATION = 'source_cartesian_delta_pose_open_positive_v1'


def representation_for_frame(frame):
    if frame == 'tool': return TOOL_REPRESENTATION
    if frame == 'source_cartesian': return SOURCE_REPRESENTATION
    raise ValueError('Explicit tool or source_cartesian action frame required')


def _vector(value, width):
    value = np.asarray(value, dtype=np.float64)
    if value.shape != (width,) or not np.isfinite(value).all():
        raise ValueError('Finite vector with declared dimensions required')
    return value


def _quaternion(value):
    value = _vector(value, 4)
    norm = np.linalg.norm(value)
    if norm < 1e-8: raise ValueError('Missing orientation cannot become identity')
    value = value / norm
    return value if value[3] >= 0 else -value


def _multiply(a, b):
    a, b = _quaternion(a), _quaternion(b)
    xyz = a[3] * b[:3] + b[3] * a[:3] + np.cross(a[:3], b[:3])
    return _quaternion(np.r_[xyz, a[3] * b[3] - a[:3] @ b[:3]])


def _scale(scale):
    if not np.isfinite(scale) or scale <= 0: raise ValueError('Positive declared metric scale required')
    return scale


def encode_source_pose(reference_position, reference_quaternion, target_position,
                       target_quaternion, gripper, position_scale_meters=.1):
    """[delta p / scale, q_target * inverse(q_reference), open-positive bit]."""
    ref = _vector(reference_position, 3)
    target = _vector(target_position, 3)
    q0 = _quaternion(reference_quaternion)
    if gripper not in (-1, 1): raise ValueError('Binary command required for full-pose source')
    delta_q = _multiply(target_quaternion, np.r_[-q0[:3], q0[3]])
    return np.r_[(target - ref) / _scale(position_scale_meters), delta_q, gripper].astype(np.float32)


def encode_source_planar(reference_xy, target_xy, position_scale_meters=.1):
    """Only measured XY is a label. Other entries are masked placeholders."""
    xy = (_vector(target_xy, 2) - _vector(reference_xy, 2)) / _scale(position_scale_meters)
    action = np.r_[xy, 0., 0., 0., 0., 1., 1.].astype(np.float32)
    mask = np.array([1., 1., 0., 0., 0., 0., 0., 0.], dtype=np.float32)
    return action, mask


def decode_source_pose(action, reference_position, reference_quaternion,
                       position_scale_meters=.1, *, supervision_mask=None):
    """Decode a complete prediction; reject partial labels when a mask is given.

    A bare 8-D array carries no capability mask. Its caller must explicitly
    distinguish a complete model prediction from a source-label placeholder.
    """
    action = _vector(action, 8)
    if supervision_mask is not None and not np.array_equal(_vector(supervision_mask, 8), np.ones(8)):
        raise ValueError('A planar partial label is not an executable 3D pose')
    if action[7] not in (-1, 1): raise ValueError('Binary decoded command required')
    return (_vector(reference_position, 3) + action[:3] * _scale(position_scale_meters),
            _multiply(action[3:7], reference_quaternion))

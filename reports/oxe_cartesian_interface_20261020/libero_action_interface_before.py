"""Explicit geometry boundary; does not establish cross-robot timing or success.

Input is one tool-relative reached-pose waypoint referenced to a fixed observation
pose. Output is the native symmetric OSC_POSE command at a separately measured
current pose. Strict bounds reject unreachable one-step setpoints, never silently
clip the model target. The execution scheduler remains a separate contract.
"""
import numpy as np
from scipy.spatial.transform import Rotation


def _vector(value, width, label):
    value = np.asarray(value, dtype=np.float64)
    if value.shape != (width,) or not np.isfinite(value).all():
        raise ValueError(label + ' must be a finite vector of length ' + str(width))
    return value


def _rotation(quaternion, label):
    quaternion = _vector(quaternion, 4, label)
    if abs(np.linalg.norm(quaternion) - 1.0) > 1e-3:
        raise ValueError(label + ' must be a nondegenerate unit xyzw quaternion')
    return Rotation.from_quat(quaternion)


def waypoint_to_native_osc(waypoint, reference_position, reference_quaternion,
                           current_position, current_quaternion,
                           position_scale_meters=0.1,
                           output_scales=(0.05, 0.05, 0.05, 0.5, 0.5, 0.5)):
    waypoint = _vector(waypoint, 8, 'waypoint')
    reference_position = _vector(reference_position, 3, 'reference position')
    current_position = _vector(current_position, 3, 'current position')
    reference_rotation = _rotation(reference_quaternion, 'reference orientation')
    relative_rotation = _rotation(waypoint[3:7], 'waypoint orientation')
    current_rotation = _rotation(current_quaternion, 'current orientation')
    scales = _vector(output_scales, 6, 'native output scales')
    if not np.isfinite(position_scale_meters) or position_scale_meters <= 0 or (scales <= 0).any():
        raise ValueError('Positive position scale and native output scales required')
    if waypoint[-1] not in (-1.0, 1.0):
        raise ValueError('Explicit binary open-positive gripper command required')
    target_position = reference_position + reference_rotation.apply(waypoint[:3] * position_scale_meters)
    target_rotation = reference_rotation * relative_rotation
    # Robosuite set_goal_orientation pre-multiplies current world orientation.
    world_delta_rotation = target_rotation * current_rotation.inv()
    delta = np.concatenate((target_position - current_position, world_delta_rotation.as_rotvec()))
    normalized = delta / scales
    if (np.abs(normalized) > 1.0 + 1e-7).any():
        raise ValueError('Target exceeds native one-control-step bounds; scheduling must be defined explicitly')
    action = np.concatenate((normalized, [-waypoint[-1]]))
    return {'native_action': action,
            'target_world_position': target_position,
            'target_world_quaternion_xyzw': target_rotation.as_quat(),
            'reference_pose_is_fixed_for_chunk': True,
            'target_was_clipped': False}


def camera_rgb_for_policy(image, image_convention):
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
        raise ValueError('Expected rendered uint8 HWC RGB')
    if image_convention == 'opengl':
        return image[::-1].copy()
    if image_convention == 'opencv':
        return image.copy()
    raise ValueError('Declare the native image convention explicitly')

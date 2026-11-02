"""Candidate command8 semantics, distinct from old reached-pose8 checkpoints.

Implements the public Bridge collector's *unclipped* rigid goal update, with
optional explicit workspace clipping. Public collector provenance has not been
bound to the cached0.0.1 export. This is not a relabeling approval or a policy.
No default previous goal, source/target calibration, clock or gripper command is
invented. Existing data, checkpoints and executors are unchanged.
"""
import numpy as np
from scipy.spatial.transform import Rotation

COMMAND_FORMAT = 'bridge_controller_increment_pose8_candidate_v1'


def vector(value, n):
    value = np.asarray(value, dtype=np.float64)
    if value.shape != (n,) or not np.isfinite(value).all():
        raise ValueError('Finite explicit vector of length '+str(n)+' required')
    return value.copy()


def orientation(q):
    q = vector(q, 4)
    if abs(np.linalg.norm(q)-1) > 1e-6:
        raise ValueError('Unit xyzw quaternion required')
    return Rotation.from_quat(q)


def positive_scale(value):
    if isinstance(value, (bool, np.bool_)) or not np.isfinite(value) or value <= 0:
        raise ValueError('Explicit positive position scale required')
    return float(value)


def encode_bridge_increment(raw_action, *, position_divisor, open_positive_command):
    """Motion6 -> position3 + quaternion4. The gripper projection is explicit.

    Source rotations are XYZ Euler angles of a *world-axis delta rotation*,
    Rz*Ry*Rx, not angles to add to the measured EEF Euler state. No measured
    future pose is used. The caller must bind the source action-clipping rule;
    this codec does not silently apply a public default to the cached export.
    """
    action = vector(raw_action, 7)
    scale = positive_scale(position_divisor)
    if open_positive_command is not None and open_positive_command not in (-1., 1.):
        raise ValueError('Explicit binary open-positive projection or unknown required')
    q = Rotation.from_euler('xyz', action[3:6]).as_quat()
    if q[3] < 0:
        q = -q
    values = np.r_[action[:3]/scale, q, 0. if open_positive_command is None else open_positive_command]
    mask = np.ones(8); mask[7] = open_positive_command is not None
    return {'format':COMMAND_FORMAT, 'values':values, 'supervision_mask':mask,
            'position_divisor':scale,
            'gripper_projection':'Explicit binary command; unknown masked, continuous source grip is not reproduced',
            'formal_training_admitted':False}


def checked_packet(packet):
    if not isinstance(packet, dict) or packet.get('format') != COMMAND_FORMAT:
        raise ValueError('Explicit controller-increment8 packet required; reached-pose8 is a different contract')
    values = vector(packet['values'], 8)
    orientation(values[3:7])
    mask = vector(packet['supervision_mask'], 8)
    if not np.array_equal(mask[:7], np.ones(7)) or mask[7] not in (0., 1.):
        raise ValueError('Motion7 must be complete; only unknown gripper may be masked')
    if mask[7] and values[7] not in (-1., 1.):
        raise ValueError('Known binary gripper command required')
    return values, mask, positive_scale(packet['position_divisor'])


def map_increment(packet, *, source_axes_to_target, target_meters_per_source_unit,
                  target_position_divisor, common_control_point_convention):
    """Declared same-control-point mapping: d_T=s*A*d_S, D_T=A*D_S*A^-1.

    A supplied convention is not measured calibration. No TCP offsets are
    inferred; the caller must supply a corresponding-point definition.
    """
    value, mask, scale = checked_packet(packet)
    axes = np.asarray(source_axes_to_target, dtype=np.float64)
    if (axes.shape != (3,3) or not np.isfinite(axes).all()
            or not np.allclose(axes.T@axes, np.eye(3), atol=1e-8, rtol=0)
            or not np.isclose(np.linalg.det(axes), 1., atol=1e-8, rtol=0)):
        raise ValueError('Explicit proper source-to-target axis rotation required')
    if not isinstance(common_control_point_convention, str) or not common_control_point_convention.strip():
        raise ValueError('An explicit corresponding control-point convention is required')
    units = positive_scale(target_meters_per_source_unit)
    divisor = positive_scale(target_position_divisor)
    A = Rotation.from_matrix(axes)
    out = value.copy()
    out[:3] = units*A.apply(value[:3]*scale)/divisor
    out[3:7] = (A*orientation(value[3:7])*A.inv()).as_quat()
    return dict(packet, values=out, supervision_mask=mask.copy(), position_divisor=divisor,
                common_control_point_convention=common_control_point_convention,
                physical_calibration_verified=False)


def advance_control_goal(packet, *, previous_goal_position, previous_goal_quaternion,
                         measured_pivot_position, workspace_bounds):
    """Advance once per source command, never once per native servo tick.

    p_new = c + d + D*(p_previous_goal-c), R_new = D*R_previous_goal.
    workspace_bounds=None explicitly selects the unclipped algebraic example.
    Current measured pose is NOT substituted for a missing previous goal.
    """
    value, mask, scale = checked_packet(packet)
    prior = vector(previous_goal_position, 3)
    prior_r = orientation(previous_goal_quaternion)
    pivot = vector(measured_pivot_position, 3)
    D = orientation(value[3:7])
    unclipped = pivot+value[:3]*scale+D.apply(prior-pivot)
    position = unclipped.copy()
    if workspace_bounds is not None:
        low, high = (vector(v, 3) for v in workspace_bounds)
        if not (low < high).all():
            raise ValueError('Ordered explicit workspace bounds required')
        position = np.clip(position, low, high)
    return {'position':position, 'quaternion_xyzw':(D*prior_r).as_quat(),
            'open_positive_command':float(value[7]) if mask[7] else None,
            'workspace_clipped':bool(np.any(position!=unclipped)),
            'unclipped_position':unclipped, 'formal_training_admitted':False}


def native_feedback_for_goal(goal, *, measured_position, measured_quaternion,
                             output_scales, saturation_policy):
    """Use an already-updated goal; this function never accumulates increments.

    The caller must use corresponding target-world EEF frames and declare its
    native tick schedule. Reuses the existing geometric conversion unchanged.
    No simulator, expert trajectory or learned target-domain executor is used.
    """
    from cartesian_action_contract import SOURCE_REPRESENTATION
    from libero_action_interface import waypoint_to_native_osc
    if goal['open_positive_command'] not in (-1., 1.):
        raise ValueError('Unknown gripper cannot be executed')
    current = vector(measured_position, 3)
    current_q = vector(measured_quaternion, 4)
    relative = orientation(goal['quaternion_xyzw'])*orientation(current_q).inv()
    waypoint = np.r_[vector(goal['position'],3)-current, relative.as_quat(), goal['open_positive_command']]
    return waypoint_to_native_osc(waypoint, current, current_q, current, current_q,
        position_scale_meters=1., output_scales=output_scales,
        action_representation=SOURCE_REPRESENTATION, saturation_policy=saturation_policy)

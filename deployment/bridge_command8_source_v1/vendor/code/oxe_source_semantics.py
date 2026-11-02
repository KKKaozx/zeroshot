"""Opt-in source semantics primitives; existing loaders/models are unchanged.

Command features use history strictly before the image, not sensed opening or a
future-derived initial condition. Planar supervision requires independently
established real-robot frame calibration; simulator orientation is insufficient.
"""
import numpy as np


def validate_command_feature_tensor(features, batch_size):
    """Check the two-feature interface without inferring missing values."""
    import torch
    if features.shape != (batch_size, 2) or not bool(torch.isfinite(features).all()):
        raise ValueError('Expected finite [B,2] command/availability features')
    value, known = features[:, 0], features[:, 1]
    valid = ((known == 0) & (value == 0)) | ((known == 1) & ((value == 1) | (value == -1)))
    if not bool(valid.all()):
        raise ValueError('Unknown=[0,0], open=[1,1], closed=[-1,1]; no implicit repair')


def last_command_features(source, past_commands):
    """Return [last open-positive binary command, availability], float32.

    Callers must supply only commands before the current observation. Bridge:
    raw absolute opening fraction. BC-Z: target_close[:,0]. Fractal: relative
    gripper_closedness_action. Language Table: empty history, always missing.
    Unknown is [0,0], open is [1,1], closed is [-1,1]. No future scan or sensor.
    """
    values = np.asarray(past_commands, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError('Finite one-dimensional past-command history required')
    if source not in ('bridge', 'bc_z', 'fractal', 'language_table'):
        raise ValueError('Explicit supported source contract required')
    unknown = np.array([0., 0.], dtype=np.float32)
    if source == 'language_table':
        if len(values):
            raise ValueError('Language Table has no gripper commands')
        return unknown
    if not len(values):
        return unknown
    if source == 'fractal':
        # Match established sign/dead-zone semantics, but initialize unknown.
        # Unlike a target-label transform, input features cannot look ahead to
        # infer the pre-first-command state from a future first switch.
        changes = np.flatnonzero(np.abs(values) > .1)
        if not len(changes):
            return unknown
        command = -1. if values[changes[-1]] > .1 else 1.
        return np.array([command, 1.], dtype=np.float32)
    if source == 'bc_z' and not np.all(np.isclose(values, 0, atol=1e-6, rtol=0)
                                     | np.isclose(values, 1, atol=1e-6, rtol=0)):
        raise ValueError('BC-Z target_close must be an absolute binary command')
    latest = values[-1]
    if np.isclose(latest, 0, atol=1e-6, rtol=0):
        command = -1. if source == 'bridge' else 1.
    elif np.isclose(latest, 1, atol=1e-6, rtol=0):
        command = 1. if source == 'bridge' else -1.
    else:
        # Raw Bridge values can include small negative values. Their validity
        # as source labels is a separate audit; do not clip/threshold them into
        # known binary inputs, reverse-fill from future, or retain an older bit.
        return unknown
    return np.array([command, 1.], dtype=np.float32)


def command_features_before_rows(source, commands):
    """Features at row t depend only on commands[:t], for explicit RLDS phase.

    The caller still has to establish that this source's observation is before
    its same-row action. Post-step datasets require a different phase binding.
    """
    values = np.asarray(commands, dtype=np.float64)
    if values.ndim != 1 or source == 'language_table':
        raise ValueError('Use finite one-dimensional command streams from a supported command source')
    # The last same-row action has not happened at any returned observation;
    # it can also be an invalid terminal RLDS action. Never validate or use it.
    last_command_features(source, values[:-1])
    result = np.zeros((len(values), 2), dtype=np.float32)
    current = np.zeros(2, dtype=np.float32)
    for index, command in enumerate(values):
        result[index] = current
        if index == len(values) - 1:
            break
        if source == 'fractal':
            if command > .1:
                current = np.array([-1., 1.], np.float32)
            elif command < -.1:
                current = np.array([1., 1.], np.float32)
        else:
            current = last_command_features(source, [command])
    return result


def calibrated_planar_projection(source_from_tool_rotation, calibration):
    """Map normalized tool-frame xyz predictions to observed source-frame xy.

    y_xy / position_scale = P_xy R_source_from_tool x_tool_normalized. The
    missing source z component remains unsupervised; it is never filled as a
    measured zero. General rotation cannot be represented by a binary xyz mask.
    """
    if (not isinstance(calibration, dict) or calibration.get('domain') != 'real'
            or calibration.get('orientation_status') != 'verified_at_input'
            or not calibration.get('source_frame') or not calibration.get('tool_frame')
            or not calibration.get('evidence')):
        raise ValueError('Real-robot tool orientation at the input must be established; no simulated calibration fallback')
    rotation = np.asarray(source_from_tool_rotation, dtype=np.float64)
    if (rotation.shape != (3, 3) or not np.isfinite(rotation).all()
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0)
            or not np.isclose(np.linalg.det(rotation), 1., atol=1e-6, rtol=0)):
        raise ValueError('Calibrated source-from-tool rotation must be a proper rotation matrix')
    return rotation[:2].copy()


def planar_xyz_loss(predicted_tool_xyz, observed_source_xy_m, projection, position_scale_meters):
    """Differentiable two-component loss kernel, not yet in main policy_loss.

    predicted_tool_xyz is a clean x0 prediction, never a raw epsilon prediction.
    Only source x/y are supervised. No z, orientation or gripper target is
    created. Calibration construction and actual source timing remain required.
    """
    import torch
    if not np.isfinite(position_scale_meters) or position_scale_meters <= 0:
        raise ValueError('Positive explicit position normalization scale required')
    if predicted_tool_xyz.shape[-1] != 3 or observed_source_xy_m.shape != (*predicted_tool_xyz.shape[:-1], 2):
        raise ValueError('Need aligned xyz predictions and measured xy displacements')
    matrix = torch.as_tensor(projection, dtype=predicted_tool_xyz.dtype, device=predicted_tool_xyz.device)
    if (matrix.shape != (2, 3) or not bool(torch.isfinite(matrix).all())
            or not torch.allclose(matrix @ matrix.T, torch.eye(2, dtype=matrix.dtype, device=matrix.device), atol=1e-6, rtol=0)):
        raise ValueError('Need a calibrated orthonormal rank-two projection')
    target = observed_source_xy_m.to(predicted_tool_xyz) / position_scale_meters
    if not bool(torch.isfinite(predicted_tool_xyz).all()) or not bool(torch.isfinite(target).all()):
        raise ValueError('Predictions and observed displacement must be finite')
    return ((predicted_tool_xyz @ matrix.T - target).square()).mean()

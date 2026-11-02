"""Explicit spatial mapping for current-state inputs AND reached-pose outputs.

Parameters are supplied facts, never inferred from target tasks or defaulted to
identity. Mathematical consistency does not establish real calibration.
"""
import numpy as np
from scipy.spatial.transform import Rotation


def _vector(value, size, name):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(name + ' must be a finite vector of length ' + str(size))
    return array.copy()


def _matrix(value, name):
    array = np.asarray(value, dtype=np.float64)
    if (array.shape != (3, 3) or not np.isfinite(array).all()
            or not np.allclose(array.T @ array, np.eye(3), atol=1e-8, rtol=0)
            or not np.isclose(np.linalg.det(array), 1, atol=1e-8, rtol=0)):
        raise ValueError(name + ' must be a proper rotation matrix')
    return Rotation.from_matrix(array)


def _orientation(value):
    q = _vector(value, 4, 'EEF xyzw')
    if abs(np.linalg.norm(q) - 1) > 1e-3:
        raise ValueError('Measured EEF orientation must be unit xyzw')
    return Rotation.from_quat(q)


class SourceTargetPoseMapping:
    """p_G^T=b+s*A*p_G^S, R_E^T=A*R_E^S*C.

    G is a declared corresponding physical point. Source/target offsets from
    each recorded EEF to G are expressed in that EEF's local axes.
    """
    required = ('target_meters_per_source_unit', 'source_axes_to_target',
                'source_origin_in_target_m', 'right_tool_alignment',
                'source_eef_to_point_source_units', 'target_eef_to_point_m', 'evidence')

    def __init__(self, spec):
        if not isinstance(spec, dict):
            raise ValueError('Explicit source/target mapping required; calibration is unavailable')
        missing = [name for name in self.required if spec.get(name) is None]
        if missing:
            raise ValueError('Missing spatial mapping facts: ' + ', '.join(missing))
        if spec.get('format') != 'source-target-pose-mapping-v1' or spec.get('source') != 'Bridge':
            raise ValueError('This checkpoint interface requires an explicit Bridge mapping')
        if not isinstance(spec['evidence'], str) or not spec['evidence'].strip():
            raise ValueError('Mapping provenance required; a supplied value is not automatically calibrated')
        scale = spec['target_meters_per_source_unit']
        if type(scale) not in (int, float) or not np.isfinite(scale) or scale <= 0:
            raise ValueError('Positive source-unit to target-meter scale required')
        self.scale = float(scale)
        self.axes = _matrix(spec['source_axes_to_target'], 'source axes to target')
        self.tool = _matrix(spec['right_tool_alignment'], 'right tool alignment')
        self.origin = _vector(spec['source_origin_in_target_m'], 3, 'source origin')
        self.source_offset = _vector(spec['source_eef_to_point_source_units'], 3, 'source TCP offset')
        self.target_offset = _vector(spec['target_eef_to_point_m'], 3, 'target TCP offset')
        self.evidence = spec['evidence']

    def source_reference(self, target_position_m, target_quaternion_xyzw):
        target_p = _vector(target_position_m, 3, 'target EEF position')
        target_r = _orientation(target_quaternion_xyzw)
        source_r = self.axes.inv() * target_r * self.tool.inv()
        point_target = target_p + target_r.apply(self.target_offset)
        point_source = self.axes.inv().apply(point_target - self.origin) / self.scale
        source_p = point_source - source_r.apply(self.source_offset)
        return source_p, source_r.as_quat()

    def target_waypoints(self, source_waypoints, target_position_m, target_quaternion_xyzw,
                         source_position_scale, target_position_scale):
        values = np.asarray(source_waypoints, dtype=np.float64)
        if (values.ndim != 2 or values.shape[1] != 8 or not np.isfinite(values).all()
                or not np.allclose(np.linalg.norm(values[:, 3:7], axis=1), 1, atol=1e-3, rtol=0)
                or not np.isin(values[:, 7], [-1, 1]).all()):
            raise ValueError('Complete predicted source [H,8] poses and binary gripper commands required')
        if (not np.isfinite(source_position_scale) or source_position_scale <= 0
                or not np.isfinite(target_position_scale) or target_position_scale <= 0):
            raise ValueError('Explicit positive saved-source and target normalization scales required')
        target_p = _vector(target_position_m, 3, 'target EEF position')
        target_r = _orientation(target_quaternion_xyzw)
        source_p, source_q = self.source_reference(target_p, target_quaternion_xyzw)
        source_r = Rotation.from_quat(source_q)
        out = values.copy()
        for index, value in enumerate(values):
            future_source_p = source_p + source_position_scale * value[:3]
            future_source_r = _orientation(value[3:7]) * source_r
            future_target_r = self.axes * future_source_r * self.tool
            future_point_source = future_source_p + future_source_r.apply(self.source_offset)
            future_target_p = (self.origin + self.scale * self.axes.apply(future_point_source)
                               - future_target_r.apply(self.target_offset))
            out[index, :3] = (future_target_p - target_p) / target_position_scale
            out[index, 3:7] = (future_target_r * target_r.inv()).as_quat()
        return out

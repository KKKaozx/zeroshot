"""One explicit-time chunk of policy waypoints in an existing native environment.

No reset, checkpoint selection, task scoring, clipping or inferred source rate.
The callback supplies decoded [H,8] targets. Evidence strings document assumptions;
they cannot by themselves prove timing or cross-robot TCP calibration.
"""
from dataclasses import dataclass

import numpy as np

from libero_action_interface import camera_rgb_for_policy, waypoint_to_native_osc
from cartesian_action_contract import TOOL_REPRESENTATION, SOURCE_REPRESENTATION


@dataclass(frozen=True)
class WaypointExecutionContract:
    target_times_seconds: tuple
    execute_waypoints: int
    position_scale_meters: float
    timing_evidence: str
    tcp_mapping_evidence: str
    native_control_hz: float = 20.0
    image_convention: str = 'opengl'
    camera_key: str = 'agentview_image'
    output_scales: tuple = (.05, .05, .05, .5, .5, .5)
    action_representation: str = TOOL_REPRESENTATION
    saturation_policy: str = 'reject'

    def ticks(self):
        if self.action_representation not in (TOOL_REPRESENTATION, SOURCE_REPRESENTATION):
            raise ValueError('Execution requires an explicit supported action representation')
        times = np.asarray(self.target_times_seconds, dtype=np.float64)
        if (times.ndim != 1 or not len(times) or not np.isfinite(times).all()
                or times[0] <= 0 or not np.all(np.diff(times) > 0)
                or type(self.execute_waypoints) is not int
                or not 1 <= self.execute_waypoints <= len(times)
                or not np.isfinite(self.native_control_hz) or self.native_control_hz <= 0
                or not np.isfinite(self.position_scale_meters) or self.position_scale_meters <= 0
                or not isinstance(self.timing_evidence, str) or not self.timing_evidence.strip()
                or not isinstance(self.tcp_mapping_evidence, str) or not self.tcp_mapping_evidence.strip()
                or self.saturation_policy not in ('reject','bounded_feedback')
                or self.image_convention not in ('opengl', 'opencv')):
            raise ValueError('Explicit ordered target times, prefix, scale and evidence are required')
        fractional_ticks = times * self.native_control_hz
        if not np.allclose(fractional_ticks, np.rint(fractional_ticks), rtol=0, atol=1e-8):
            raise ValueError('Target times must coincide with native control ticks; no implicit resampling')
        ticks = np.rint(fractional_ticks).astype(np.int64)
        if ticks[0] < 1 or not np.all(np.diff(ticks) > 0):
            raise ValueError('At least one native control tick is required per waypoint')
        return ticks


def _pose(observation):
    position = np.array(observation['robot0_eef_pos'], dtype=np.float64, copy=True)
    quaternion = np.array(observation['robot0_eef_quat'], dtype=np.float64, copy=True)
    if (position.shape != (3,) or quaternion.shape != (4,)
            or not np.isfinite(position).all() or not np.isfinite(quaternion).all()
            or abs(np.linalg.norm(quaternion) - 1) > 1e-3):
        raise ValueError('Native observation must supply a finite measured EEF pose and unit xyzw')
    return position, quaternion


def execute_policy_chunk(env, policy, observation, instruction, current_open_command,
                         contract, *, native_step_budget):
    """Call policy once, execute a fixed-reference prefix with measured feedback.

    policy receives an upright RGB image, language, reference EEF pose and the
    last explicit open-positive command. That command is not a measured opening.
    None represents unknown history and requires a policy that supports missing
    command input; a command is known after the first actual env.step call.
    For each declared interval, send its target and gripper command from interval
    start, recomputing OSC deltas each tick. No intermediate poses are invented.
    Stops on native done; leaves scoring/replanning to the caller.
    """
    ticks = contract.ticks()
    required = int(ticks[contract.execute_waypoints - 1])
    if type(native_step_budget) is not int or native_step_budget < required:
        raise ValueError('Budget must cover the declared execution prefix before inference')
    if current_open_command is not None and (type(current_open_command) not in (int, float)
                                           or current_open_command not in (-1, 1)):
        raise ValueError('Last explicit binary gripper command required; no sensed-opening threshold')
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError('Nonempty instruction required')
    native = env.env
    controller = env.robots[0].controller
    scales = np.asarray(contract.output_scales, dtype=np.float64)
    if (scales.shape != (6,) or not np.isfinite(scales).all() or (scales <= 0).any()
            or native.control_freq != contract.native_control_hz
            or controller.control_dim != 6 or not controller.use_delta
            or not np.array_equal(controller.output_max, scales)
            or not np.array_equal(controller.output_min, -scales)
            or not np.array_equal(np.broadcast_to(controller.input_max, (6,)), np.ones(6))
            or not np.array_equal(np.broadcast_to(controller.input_min, (6,)), -np.ones(6))):
        raise ValueError('Actual native timing/OSC bounds differ from execution contract')
    reference_position, reference_quaternion = _pose(observation)
    request = {
        'rgb_upright': camera_rgb_for_policy(observation[contract.camera_key], contract.image_convention),
        'instruction': instruction, 'reference_position_m': reference_position.copy(),
        'reference_quaternion_xyzw': reference_quaternion.copy(),
        'current_open_command': None if current_open_command is None else float(current_open_command),
        'action_representation': contract.action_representation,
    }
    waypoints = np.array(policy(request), dtype=np.float64, copy=True)
    if (waypoints.shape != (len(ticks), 8) or not np.isfinite(waypoints).all()
            or not np.allclose(np.linalg.norm(waypoints[:, 3:7], axis=1), 1, rtol=0, atol=1e-3)
            or not np.isin(waypoints[:, 7], [-1., 1.]).all()):
        raise ValueError('Policy must decode finite [H,8] targets with unit xyzw and binary commands')
    trace = []
    obs, done = observation, False
    last_command = None if current_open_command is None else float(current_open_command)
    previous_tick = 0
    for index in range(contract.execute_waypoints):
        interval_ticks = int(ticks[index]) - previous_tick
        for _ in range(interval_ticks):
            current_position, current_quaternion = _pose(obs)
            converted = waypoint_to_native_osc(
                waypoints[index], reference_position, reference_quaternion,
                current_position, current_quaternion, contract.position_scale_meters, scales,
                action_representation=contract.action_representation,
                saturation_policy=contract.saturation_policy)
            action = converted['native_action']
            obs, _, done, _ = env.step(action.copy())
            last_command = float(waypoints[index, 7])
            measured_position, _ = _pose(obs)
            trace.append({'native_tick': len(trace) + 1, 'waypoint_index': index,
                          'simulated_seconds': (len(trace) + 1) / contract.native_control_hz,
                          'native_action': action.tolist(),
                          'native_command_saturated':converted['native_command_saturated'],
                          'saturated_components':converted['saturated_components'],
                          'target_world_position_m': converted['target_world_position'].tolist(),
                          'target_world_quaternion_xyzw': converted['target_world_quaternion_xyzw'].tolist(),
                          'measured_position_m': measured_position.tolist(),
                          'target_position_error_m': float(np.linalg.norm(
                              measured_position - converted['target_world_position']))})
            if done:
                break
        previous_tick = int(ticks[index])
        if done:
            break
    return {'observation': obs, 'last_open_command': last_command, 'native_done': bool(done),
            'native_steps': len(trace), 'trace': trace, 'policy_calls': 1,
            'reference_position_m': reference_position.tolist(),
            'reference_quaternion_xyzw': reference_quaternion.tolist(),
            'predicted_waypoints': waypoints.tolist(),
            'target_times_seconds': list(contract.target_times_seconds),
            'timing_evidence': contract.timing_evidence, 'tcp_mapping_evidence': contract.tcp_mapping_evidence,
            'execution_rule': 'Fixed world target with measured feedback during each declared interval',
            'saturation_policy':contract.saturation_policy,
            'source_timing_or_tcp_calibration_automatically_proven': False,
            'task_success_scored': False}

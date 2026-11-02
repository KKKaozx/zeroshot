"""New geometry boundary vs the actual installed robosuite OSC set_goal method."""
import json
import os
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
os.environ['MUJOCO_GL'] = 'glfw'
from libero_action_interface import waypoint_to_native_osc, camera_rgb_for_policy
from robosuite.controllers.osc import OperationalSpaceController


def controller(position, rotation):
    obj = OperationalSpaceController.__new__(OperationalSpaceController)
    obj.update = lambda: None
    obj.ee_pos = position.copy()
    obj.ee_ori_mat = rotation.as_matrix()
    obj.goal_ori = obj.ee_ori_mat.copy()
    obj.impedance_mode, obj.use_delta, obj.use_ori = 'fixed', True, True
    obj.action_scale = None
    obj.input_min, obj.input_max = -np.ones(6), np.ones(6)
    obj.output_max = np.asarray([.05, .05, .05, .5, .5, .5])
    obj.output_min = -obj.output_max
    obj.orientation_limits = obj.position_limits = None
    obj.interpolator_pos = obj.interpolator_ori = None
    return obj


def rejected(fn):
    try:
        fn()
    except ValueError:
        return True
    raise AssertionError('Invalid control target accepted')


def main():
    rng = np.random.default_rng(20261020)
    max_position, max_rotation = 0., 0.
    for _ in range(128):
        reference = rng.normal(0, .1, 3)
        reference_rotation = Rotation.random(random_state=rng)
        current = reference + rng.normal(0, .002, 3)
        current_rotation = reference_rotation * Rotation.from_rotvec(rng.normal(0, .01, 3))
        local = rng.uniform(-.1, .1, 3)
        relative = Rotation.from_rotvec(rng.uniform(-.04, .04, 3))
        target = reference + reference_rotation.apply(local * .1)
        target_rotation = reference_rotation * relative
        waypoint = np.r_[local, relative.as_quat(), rng.choice([-1., 1.])]
        converted = waypoint_to_native_osc(waypoint, reference, reference_rotation.as_quat(),
                                           current, current_rotation.as_quat())
        actual_controller = controller(current, current_rotation)
        actual_controller.set_goal(converted['native_action'][:6])
        position_error = np.max(np.abs(actual_controller.goal_pos - target))
        rotation_error = (Rotation.from_matrix(actual_controller.goal_ori).inv() * target_rotation).magnitude()
        max_position = max(max_position, float(position_error))
        max_rotation = max(max_rotation, float(rotation_error))
        assert position_error < 1e-10 and rotation_error < 1e-6
        assert converted['native_action'][-1] == -waypoint[-1]
    base = np.r_[np.zeros(3), [0, 0, 0, 1], 1.]
    invoke = lambda w: waypoint_to_native_osc(w, [0, 0, 0], [0, 0, 0, 1], [0, 0, 0], [0, 0, 0, 1])
    assert rejected(lambda: invoke(np.r_[np.ones(3) * 5, base[3:]]))
    assert rejected(lambda: invoke(np.r_[base[:3], [0, 0, 0, 0], 1]))
    assert rejected(lambda: invoke(np.r_[base[:-1], .3]))
    assert rejected(lambda: invoke(base * np.nan))
    image = np.arange(4 * 3 * 3, dtype=np.uint8).reshape(4, 3, 3)
    assert np.array_equal(camera_rgb_for_policy(image, 'opengl'), image[::-1])
    assert np.array_equal(camera_rgb_for_policy(image, 'opencv'), image)
    assert rejected(lambda: camera_rgb_for_policy(image, 'unknown'))
    # Independent of current pose: both targets retain the same chunk reference.
    first = invoke(np.r_[[.1, 0, 0], base[3:]])
    second = waypoint_to_native_osc(np.r_[[.2, 0, 0], base[3:]], [0, 0, 0], [0, 0, 0, 1],
                                    first['target_world_position'], [0, 0, 0, 1])
    assert np.allclose(second['target_world_position'], [.02, 0, 0])
    report = {'geometry_checks_passed': True, 'cases': 128,
              'native_robosuite_set_goal_used': True, 'max_position_difference_m': max_position,
              'max_orientation_difference_rad': max_rotation, 'gripper_signs_checked': True,
              'fixed_chunk_reference_checked': True, 'camera_vertical_convention_checked': True,
              'invalid_targets_rejected_without_clipping': True, 'learned_model_used': False,
              'control_timing_and_cross_robot_calibration_validated': False,
              'benchmark_success_claimed': False}
    output = ROOT / 'reports/libero_action_interface_20261020.json'
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

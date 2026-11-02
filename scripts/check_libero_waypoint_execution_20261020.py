"""Timing/feedback execution with synthetic dynamics and real OSC goal code.

Not a physics rollout, learned policy test, or benchmark success measurement.
"""
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
sys.path.insert(0, str(ROOT / 'scripts'))
os.environ['MUJOCO_GL'] = 'glfw'
from libero_waypoint_execution import WaypointExecutionContract, execute_policy_chunk
from check_libero_action_interface_20261020 import controller


class FeedbackFixture:
    def __init__(self, stop_after=None):
        self.position = np.array([.3, -.2, .1])
        self.rotation = Rotation.from_euler('z', 90, degrees=True)
        self.ctl = controller(self.position, self.rotation)
        self.ctl.control_dim = 6
        self.env = SimpleNamespace(control_freq=20)
        self.robots = [SimpleNamespace(controller=self.ctl)]
        self.actions, self.goals = [], []
        self.stop_after = stop_after

    def obs(self):
        return {'robot0_eef_pos': self.position.copy(), 'robot0_eef_quat': self.rotation.as_quat(),
                'agentview_image': np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)}

    def step(self, action):
        self.ctl.ee_pos, self.ctl.ee_ori_mat = self.position.copy(), self.rotation.as_matrix()
        self.ctl.set_goal(action[:6])
        self.actions.append(action.copy())
        self.goals.append(self.ctl.goal_pos.copy())
        # Deliberate lag. Reusing the original delta on tick 2 would overshoot
        # the fixed target instead of reducing the error again.
        self.position += .5 * (self.ctl.goal_pos - self.position)
        self.rotation = Rotation.from_matrix(self.ctl.goal_ori)
        done = self.stop_after is not None and len(self.actions) >= self.stop_after
        return self.obs(), 0., done, {}

    def reset(self):
        raise AssertionError('Executor must not reset the environment')

    def check_success(self):
        raise AssertionError('Executor must not inspect success labels')


def rejected(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError('Invalid execution contract or prediction accepted')


def main():
    contract = WaypointExecutionContract(
        target_times_seconds=(.1, .2), execute_waypoints=2, position_scale_meters=.1,
        timing_evidence='Synthetic fixture: .1 second targets; not an OXE source-rate claim',
        tcp_mapping_evidence='Synthetic identical reference and native EEF; not cross-robot calibration')
    targets = np.array([[.1, 0, 0, 0, 0, 0, 1, -1.], [.2, 0, 0, 0, 0, 0, 1, 1.]])
    calls = []

    def policy(request):
        calls.append(request)
        assert np.array_equal(request['rgb_upright'], np.arange(48, dtype=np.uint8).reshape(4, 4, 3)[::-1])
        assert request['current_open_command'] == 1
        return targets.copy()

    env = FeedbackFixture()
    start = env.obs()
    result = execute_policy_chunk(env, policy, start, 'synthetic test', 1, contract, native_step_budget=4)
    expected = [np.array([.3, -.19, .1])] * 2 + [np.array([.3, -.18, .1])] * 2
    np.testing.assert_allclose(env.goals, expected, atol=1e-10)
    assert result['native_steps'] == 4 and len(calls) == 1
    assert [r['waypoint_index'] for r in result['trace']] == [0, 0, 1, 1]
    assert [a[-1] for a in env.actions] == [1., 1., -1., -1.]
    assert result['last_open_command'] == 1 and not result['task_success_scored']
    np.testing.assert_allclose(env.actions[1][:3], env.actions[0][:3] * .5, atol=1e-12)
    np.testing.assert_allclose([r['simulated_seconds'] for r in result['trace']], [.05, .1, .15, .2])
    # The next call replans from its new observation. It must not retain or
    # accumulate the previous chunk's reference or execute its unused tail.
    new_reference = env.position.copy()
    expected_next_target = new_reference + env.rotation.apply([.01, 0, 0])
    next_result = execute_policy_chunk(env, policy, result['observation'], 'synthetic test',
                                      result['last_open_command'], replace(contract, execute_waypoints=1),
                                      native_step_budget=2)
    np.testing.assert_allclose(next_result['reference_position_m'], new_reference)
    np.testing.assert_allclose(env.goals[-2:], [expected_next_target] * 2, atol=1e-10)

    fresh = FeedbackFixture()
    before_calls = len(calls)
    rejected(lambda: execute_policy_chunk(fresh, policy, fresh.obs(), 'test', 1, contract, native_step_budget=3))
    assert not fresh.actions and len(calls) == before_calls
    for bad in (replace(contract, target_times_seconds=(.1, .21)),
                replace(contract, target_times_seconds=(.2, .1)),
                replace(contract, execute_waypoints=0), replace(contract, timing_evidence='')):
        rejected(bad.ticks)
    fresh.env.control_freq = 10
    rejected(lambda: execute_policy_chunk(fresh, policy, fresh.obs(), 'test', 1, contract, native_step_budget=4))
    fresh.env.control_freq = 20
    rejected(lambda: execute_policy_chunk(fresh, policy, fresh.obs(), 'test', 0, contract, native_step_budget=4))
    for bad in ('zero_quaternion', 'neutral_gripper', 'nonfinite', 'too_far'):
        changed = targets.copy()
        if bad == 'zero_quaternion':
            changed[1, 3:7] = 0
        elif bad == 'neutral_gripper':
            changed[1, 7] = 0
        elif bad == 'nonfinite':
            changed[1, 0] = np.nan
        else:
            changed[0, 0] = 10
        rejected(lambda: execute_policy_chunk(fresh, lambda _: changed, fresh.obs(), 'test', 1,
                                             contract, native_step_budget=4))
        assert not fresh.actions
    stopped = FeedbackFixture(stop_after=1)
    early = execute_policy_chunk(stopped, policy, stopped.obs(), 'test', 1, contract, native_step_budget=4)
    assert early['native_steps'] == 1 and early['native_done'] and early['last_open_command'] == -1

    report = {'checks_passed': True, 'native_robosuite_set_goal_used': True,
              'synthetic_dynamics_only': True, 'real_physics_steps': 0, 'learned_model_loaded': False,
              'declared_10hz_targets_execute_on_20hz_native_ticks': True,
              'measured_feedback_recomputes_delta_each_tick': True,
              'reference_fixed_within_chunk_and_refreshed_on_replan': True,
              'gripper_command_applies_from_interval_start': True,
              'budget_checked_before_inference': True, 'native_done_stops_execution': True,
              'invalid_targets_and_unknown_timing_rejected': True,
              'environment_reset_or_success_labels_accessed': False,
              'oxe_source_timing_or_cross_robot_tcp_validated': False,
              'benchmark_success_claimed': False}
    path = ROOT / 'reports/libero_waypoint_execution_20261020.json'
    path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

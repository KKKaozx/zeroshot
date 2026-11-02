"""CPU interface tests only: no learned policy, simulator or performance claim."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
from cartesian_action_contract import SOURCE_REPRESENTATION
from libero_execution_binding import snapshot_execution_state, step_bound_native_command
from libero_waypoint_execution import WaypointExecutionContract, execute_policy_chunk


class Environment:
    def __init__(self):
        self.state = np.array([0., .2, -.1, .5])
        self.actions = []
        self.env = SimpleNamespace(control_freq=20)
        self.robots = [SimpleNamespace(controller=SimpleNamespace(
            control_dim=6, use_delta=True, input_min=-1., input_max=1.,
            output_max=np.array([.05]*3+[.5]*3), output_min=-np.array([.05]*3+[.5]*3)))]

    def get_sim_state(self):
        return self.state.copy()

    def observation(self):
        return {'robot0_eef_pos': self.state[1:].copy(),
                'robot0_eef_quat': np.array([0., 0., 0., 1.]),
                'agentview_image': np.zeros((4, 4, 3), np.uint8)}

    def step(self, action):
        self.actions.append(action.copy())
        self.state[0] += .05
        self.state[1:] += action[:3]*.05
        return self.observation(), 0., False, {}


def contract():
    return WaypointExecutionContract((.2,), 1, .1, 'Synthetic5Hz fixture',
        'Synthetic coincident-frame fixture, no calibration claim',
        action_representation=SOURCE_REPRESENTATION)


class BindingTests(unittest.TestCase):
    def test_nontrivial_pose8_still_converts_rotation_and_open_positive_gripper(self):
        env = Environment()
        q = Rotation.from_rotvec([.1, -.05, .025]).as_quat()
        waypoint = np.r_[.2, -.1, .05, q, 1.][None]
        result = execute_policy_chunk(env, lambda request: waypoint, env.observation(),
                                      'synthetic motion', 1., contract(), native_step_budget=4)
        np.testing.assert_allclose(env.actions[0], [.4, -.2, .1, .2, -.1, .05, -1.], atol=1e-12)
        np.testing.assert_allclose(np.array(env.actions)[1:, :3], 0., atol=1e-12)
        self.assertEqual(result['native_steps'], 4)
        self.assertEqual(result['predicted_waypoints'], waypoint.tolist())
        self.assertFalse(result['incoming_observation_freshness_certified'])

    def test_callback_advancing_simulator_is_rejected_before_executor_sends_action(self):
        env = Environment()
        def policy(request):
            env.state[0] += .05
            return np.array([[0., 0., 0., 0., 0., 0., 1., 1.]])
        with self.assertRaisesRegex(RuntimeError, 'Simulator changed'):
            execute_policy_chunk(env, policy, env.observation(), 'synthetic', 1., contract(), native_step_budget=4)
        self.assertEqual(env.actions, [])

    def test_stale_command_and_wrong_dimension_are_rejected_without_environment_step(self):
        env = Environment(); state = snapshot_execution_state(env)
        env.state[0] += .05
        with self.assertRaisesRegex(RuntimeError, 'Simulator changed'):
            step_bound_native_command(env, np.zeros(7), state)
        with self.assertRaisesRegex(ValueError, 'pose8'):
            step_bound_native_command(env, np.zeros(8), state)
        with self.assertRaisesRegex(ValueError, 'bounds'):
            step_bound_native_command(env, np.full(7, 2.), state)
        self.assertEqual(env.actions, [])

    def test_native7_cannot_be_misread_as_pose8_by_waypoint_executor(self):
        env = Environment()
        with self.assertRaisesRegex(ValueError, r'\[H,8\]'):
            execute_policy_chunk(env, lambda request: np.zeros((1, 7)), env.observation(),
                                 'synthetic', 1., contract(), native_step_budget=4)
        self.assertEqual(env.actions, [])


if __name__ == '__main__':
    unittest.main(verbosity=2)

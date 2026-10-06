"""Continuous action semantics and rejection must be tested before motor calls."""
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from cliport_controller import ContinuousTCPController


class EndEffector:
    activated = False

    def __init__(self):
        self.activations = 0
        self.releases = 0

    def activate(self):
        self.activated = True
        self.activations += 1

    def release(self):
        self.activated = False
        self.releases += 1

    def check_grasp(self):
        return self.activated


class Environment:
    def __init__(self, timeout=False):
        self.ee = EndEffector()
        self.calls = []
        self.step_counter = 0
        self.timeout = timeout

    def movep(self, pose, speed):
        self.calls.append(pose)
        self.step_counter += 1
        return self.timeout


class ContinuousTests(unittest.TestCase):
    def actions(self, xs):
        a = np.zeros((len(xs), 8), np.float32)
        a[:, 0], a[:, 6], a[:, 7] = xs, 1., 1.
        return a

    def run_actions(self, env, a):
        return ContinuousTCPController(env).execute(a, [.4, 0, .2], [0, 0, 0, 1])

    def test_fixed_reference_for_entire_block(self):
        env = Environment()
        self.run_actions(env, self.actions([.5, 1.]))
        np.testing.assert_allclose([p[0][0] for p in env.calls], [.45, .5], atol=1e-6)

    def test_close_hold_open_commands(self):
        env = Environment()
        a = self.actions([0, 0, 0])
        a[:, 7] = [-1, -1, 1]
        self.run_actions(env, a)
        self.assertEqual(env.ee.activations, 1)
        self.assertEqual(env.ee.releases, 1)
        self.assertFalse(env.ee.activated)

    def test_suction_only_events_do_not_move_or_advance_physics(self):
        env = Environment(); controller = ContinuousTCPController(env)
        controller.command_suction(False)
        controller.command_suction(False)
        controller.command_suction(True)
        self.assertEqual((env.ee.activations, env.ee.releases), (1, 1))
        self.assertEqual(env.calls, [])
        self.assertEqual(env.step_counter, 0)

    def test_invalid_later_target_prevents_all_motor_calls(self):
        env = Environment()
        a = self.actions([0, 0]); a[1, 2] = -3
        with self.assertRaisesRegex(ValueError, 'workspace'):
            self.run_actions(env, a)
        self.assertEqual(env.calls, [])

    def test_normalized_outlier_is_not_clipped(self):
        env = Environment()
        with self.assertRaisesRegex(ValueError, 'no clipping'):
            self.run_actions(env, self.actions([3.1]))
        self.assertEqual(env.calls, [])

    def test_timeout_stops_before_suction_or_later_targets(self):
        env = Environment(timeout=True)
        a = self.actions([0, 1]); a[:, 7] = -1
        result = self.run_actions(env, a)
        self.assertEqual(len(env.calls), 1)
        self.assertTrue(result[0]['timeout'])
        self.assertEqual(env.ee.activations, 0)

    def test_zero_quaternion_rejected_before_motion(self):
        env = Environment(); a = self.actions([0]); a[:, 3:7] = 0
        with self.assertRaisesRegex(ValueError, 'quaternions'):
            self.run_actions(env, a)
        self.assertEqual(env.calls, [])


if __name__ == '__main__':
    unittest.main()

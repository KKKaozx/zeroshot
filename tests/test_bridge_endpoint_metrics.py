import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from evaluate_bridge_endpoint_metrics import scalar_metrics, oracle_minimum


class EndpointMetricTests(unittest.TestCase):
    def fixture(self):
        target = np.zeros((2, 16, 8), np.float32)
        target[..., 6] = 1
        target[..., 7] = 1
        target[0, 8:, 7] = -1
        target[1, 12:, 7] = -1
        prediction = target.copy()
        prediction[:, :-1, 0] = 1
        prediction[:, -1, 0] = 2
        return prediction, target

    def test_path_endpoint_and_event_are_distinct(self):
        prediction, target = self.fixture()
        result = scalar_metrics(prediction, target)
        self.assertAlmostEqual(result['path_position_cm'], 10.625)
        self.assertAlmostEqual(result['endpoint_position_cm'], 20.0)
        self.assertEqual(result['grasp_events'], 2)
        self.assertAlmostEqual(result['grasp_position_cm_at_true_event'], 10.0)
        self.assertEqual(result['grasp_exact_timing_accuracy'], 1.0)

    def test_quaternion_sign_and_oracle(self):
        prediction, target = self.fixture()
        prediction[..., 3:7] *= -1
        self.assertAlmostEqual(scalar_metrics(prediction, target)['path_rotation_deg'], 0.0)
        worse = prediction.copy(); worse[..., 0] += 3
        oracle = oracle_minimum(np.stack([worse, prediction]), target)
        self.assertAlmostEqual(oracle['endpoint_position_cm'], 20.0)
        self.assertEqual(oracle['draws'], 2)


if __name__ == '__main__':
    unittest.main()

import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from evaluate_bridge_endpoint_metrics import pose_errors
from evaluate_bridge_official_unet_one_step import extended_metrics, groups_for


class OfficialUnetEvaluationTest(unittest.TestCase):
    def test_metrics_include_balanced_gripper_and_events(self):
        target = np.zeros((1, 3, 8), np.float32)
        prediction = np.zeros_like(target)
        target[..., 6] = 1
        prediction[..., 6] = 1
        target[0, :, 7] = [1, -1, 1]
        prediction[0, :, 7] = [1, -1, -1]
        result = extended_metrics(prediction, target, pose_errors)
        self.assertEqual(result["gripper_class_count"], 2)
        self.assertAlmostEqual(result["gripper_accuracy"], 2 / 3)
        self.assertAlmostEqual(result["gripper_balanced_accuracy"], 0.75)
        self.assertEqual(result["grasp_events"], 1)
        self.assertEqual(result["grasp_exact_timing_accuracy"], 1.0)
        self.assertEqual(result["release_events"], 1)
        self.assertEqual(result["release_exact_timing_accuracy"], 0.0)

    def test_groups_keep_episode_boundaries(self):
        rows = [
            {"partition": "train", "task": "a", "shard": "x", "record_index": 1},
            {"partition": "train", "task": "a", "shard": "x", "record_index": 1},
            {"partition": "validation", "task": "b", "shard": "y", "record_index": 2},
        ]
        groups = groups_for(rows)
        self.assertEqual(groups["train/overall"], [0, 1])
        self.assertEqual(groups["validation/overall"], [2])
        self.assertEqual(groups["train/episode/x::1"], [0, 1])


if __name__ == "__main__":
    unittest.main()

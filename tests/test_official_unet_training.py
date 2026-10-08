import sys
from pathlib import Path
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from run_bridge_official_unet_training import common_state, orders


class OfficialUnetTrainingProtocolTest(unittest.TestCase):
    def test_orders_are_complete_and_deterministic(self):
        first = orders([0, 1, 2, 3], epochs=3)
        second = orders([0, 1, 2, 3], epochs=3)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)
        for epoch in first:
            self.assertEqual(sorted(epoch), [0, 1, 2, 3])
        self.assertNotEqual(first[0], first[1])

    def test_common_state_excludes_both_pose_heads(self):
        state = {
            "adapter.weight": torch.tensor([1.0]),
            "gripper_head.weight": torch.tensor([2.0]),
            "regression_head.weight": torch.tensor([3.0]),
            "diffusion_decoder.weight": torch.tensor([4.0]),
        }
        shared = common_state(state)
        self.assertEqual(set(shared), {"adapter.weight", "gripper_head.weight"})
        state["adapter.weight"].zero_()
        self.assertEqual(float(shared["adapter.weight"]), 1.0)


if __name__ == "__main__":
    unittest.main()

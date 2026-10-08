import hashlib
import json
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from run_bridge_epsilon_control import common_state, digest, orders


class EpsilonControlTest(unittest.TestCase):
    def test_orders_cover_each_index_once_and_are_reproducible(self):
        first = orders([2, 5, 9], epochs=4)
        second = orders([2, 5, 9], epochs=4)
        self.assertEqual(first, second)
        self.assertTrue(all(sorted(epoch) == [2, 5, 9] for epoch in first))
        self.assertEqual(
            hashlib.sha256(json.dumps(first).encode()).hexdigest(),
            hashlib.sha256(json.dumps(second).encode()).hexdigest(),
        )

    def test_common_state_excludes_only_alternative_pose_heads(self):
        state = {
            "adapter.weight": torch.tensor([1.0]),
            "gripper_head.weight": torch.tensor([2.0]),
            "regression_head.weight": torch.tensor([3.0]),
            "diffusion_decoder.weight": torch.tensor([4.0]),
        }
        common = common_state(state)
        self.assertEqual(set(common), {"adapter.weight", "gripper_head.weight"})
        self.assertEqual(digest(common), digest(common_state(state)))


if __name__ == "__main__":
    unittest.main()

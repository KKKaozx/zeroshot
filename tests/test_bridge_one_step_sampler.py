import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from evaluate_bridge_one_step_sampler import finish_pose, decode_state_gripper


class FakeModel:
    gripper_target_mode = "state"

    def predict_gripper_logits(self, context, pose, current):
        return pose[..., 0]


class OneStepSamplerTest(unittest.TestCase):
    def test_pose_constraints_match_contract(self):
        pose = torch.zeros(1, 2, 7)
        pose[0, 0, :3] = torch.tensor([4.0, -5.0, 2.0])
        pose[0, 1, 3:7] = torch.tensor([0.0, 0.0, 0.0, 2.0])
        result = finish_pose(torch, pose, 3.0)
        self.assertEqual(result[0, 0, :3].tolist(), [3.0, -3.0, 2.0])
        self.assertTrue(torch.allclose(result[..., 3:7].norm(dim=-1), torch.ones(1, 2)))
        self.assertEqual(result[0, 0, 6].item(), 1.0)

    def test_state_gripper_uses_predicted_pose(self):
        pose = torch.zeros(1, 3, 7)
        pose[..., 0] = torch.tensor([[-1.0, 0.0, 2.0]])
        result = decode_state_gripper(torch, FakeModel(), None, pose, torch.zeros(1))
        self.assertEqual(result.tolist(), [[-1.0, 1.0, 1.0]])


if __name__ == "__main__":
    unittest.main()

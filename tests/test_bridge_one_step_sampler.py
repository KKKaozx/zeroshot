import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from evaluate_bridge_one_step_sampler import finish_pose, decode_state_gripper, select_windows


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

    def test_window_selection_is_source_ordered_and_complete(self):
        class Dataset:
            samples = [
                {"file_path": "/x/b", "record_index": 2, "start_index": 4},
                {"file_path": "/x/a", "record_index": 1, "start_index": 8},
                {"file_path": "/x/a", "record_index": 1, "start_index": 0},
            ]
        selection = [
            {"shard": "a", "record_index": 1, "instruction": "open"},
            {"shard": "b", "record_index": 2, "instruction": "close"},
        ]
        rows = select_windows(Dataset(), {"train": [0, 2], "validation": [1]}, selection)
        self.assertEqual([row["dataset_index"] for row in rows], [2, 0, 1])
        self.assertEqual([row["task"] for row in rows], ["open", "close", "open"])


if __name__ == "__main__":
    unittest.main()

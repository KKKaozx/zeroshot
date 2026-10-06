"""Primitive indexing must stay separate from continuous training targets."""
import pickle
import sys
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from dataset import UnifiedRobotDataset, trajectory_to_cliport_primitive


class NativeContractTests(unittest.TestCase):
    def setUp(self):
        scratch = ROOT / "training_cache/exports"
        scratch.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="cliport-test-", dir=scratch)
        assert Path(self.temp.name).resolve().is_relative_to(scratch.resolve())
        self.addCleanup(self.temp.cleanup)
        self.task = Path(self.temp.name) / "task-train"
        pose = ((.41875, .096875, .039978), (0., 0., 0., 1.))
        action = {"pose0": pose, "pose1": ((.3, -.2, .05), pose[1])}
        self.values = dict(action=[action, None], color=np.zeros((2, 3, 4, 5, 3), np.uint8),
                           depth=np.zeros((2, 3, 4, 5), np.float32),
                           info=[{"lang_goal": "place red block"}, {"lang_goal": "done"}], reward=[0., 1.])
        self.path = self.task / "action/000000-0.pkl"

    def read(self):
        for key, value in self.values.items():
            folder = self.task / key
            folder.mkdir(parents=True, exist_ok=True)
            with (folder / self.path.name).open("wb") as stream:
                pickle.dump(value, stream)
        return UnifiedRobotDataset.read_cliport_native_episode(self.path)

    def test_native_world_pose_and_terminal_index_are_preserved(self):
        episode = self.read()
        self.assertEqual(episode["action"], self.values["action"])
        self.assertEqual(episode["executable_step_indices"], [0])
        self.assertEqual(episode["info"][0]["lang_goal"], "place red block")
        self.assertFalse(episode["compatible_with_unified_training"])

    def test_alignment_mismatch_is_rejected(self):
        self.values["info"].pop()
        with self.assertRaisesRegex(ValueError, "lengths differ"):
            self.read()

    def test_nonterminal_hole_is_not_filtered_or_reindexed(self):
        self.values["action"] = [None, self.values["action"][0]]
        with self.assertRaisesRegex(ValueError, "Nonterminal"):
            self.read()

    def test_invalid_quaternion_is_not_silently_replaced(self):
        self.values["action"][0]["pose0"] = ((.4, .1, .05), (0., 0., 0., 0.))
        with self.assertRaisesRegex(ValueError, "unit xyzw"):
            self.read()

    def test_unified_ingestion_and_legacy_bypass_are_blocked(self):
        dataset = UnifiedRobotDataset.__new__(UnifiedRobotDataset)
        dataset.discovered_counts = defaultdict(int)
        dataset.incompatible_counts = defaultdict(int)
        dataset.exclude_schemas = frozenset()
        dataset.samples = []
        self.assertTrue(dataset._append({"source": "cliport"}))
        self.assertEqual(dataset.samples, [])
        with self.assertRaisesRegex(ValueError, "world-frame primitives"):
            dataset._get_cliport({"source": "cliport"})


class EventAdapterTests(unittest.TestCase):
    def fixture(self, commands):
        actions = np.zeros((len(commands), 8), dtype=np.float32)
        actions[:, 6] = 1.
        actions[:, 7] = commands
        return actions

    def convert(self, actions, current_open=True):
        return trajectory_to_cliport_primitive([.4, -.1, .2], [0, 0, 0, 1],
                                               actions, current_open=current_open)

    def test_event_indices_and_rotated_frame_not_first_last(self):
        actions = self.fixture([1, 1, -1, -1, 1, 1])
        actions[2, :3] = [.2, .1, 0]
        actions[4, :3] = [-.3, .2, .1]
        primitive, meta = trajectory_to_cliport_primitive([.4, -.1, .2],
            [0, 0, np.sqrt(.5), np.sqrt(.5)], actions, current_open=True)
        self.assertEqual((meta['pick_index'], meta['place_index']), (2, 4))
        np.testing.assert_allclose(primitive['pose0'][0], [.39, -.08, .2], atol=1e-6)
        np.testing.assert_allclose(primitive['pose1'][0], [.38, -.13, .21], atol=1e-6)

    def test_missing_release_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'complete'):
            self.convert(self.fixture([1, -1, -1]))

    def test_no_grasp_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'complete'):
            self.convert(self.fixture([1, 1, 1]))

    def test_multiple_cycles_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'multiple'):
            self.convert(self.fixture([-1, 1, -1, 1]))

    def test_already_closed_reference_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'initially open'):
            self.convert(self.fixture([-1, 1]), current_open=False)

    def test_uncertain_command_is_not_rounded(self):
        with self.assertRaisesRegex(ValueError, 'binary'):
            self.convert(self.fixture([.1, -1, 1]))

    def test_zero_quaternion_is_not_repaired(self):
        actions = self.fixture([-1, 1])
        actions[0, 3:7] = 0
        with self.assertRaisesRegex(ValueError, 'quaternions'):
            self.convert(actions)


if __name__ == "__main__":
    unittest.main()

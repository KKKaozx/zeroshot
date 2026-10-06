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
from dataset import UnifiedRobotDataset


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


if __name__ == "__main__":
    unittest.main()

import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from diagnose_multiscale_noise import select_training, restoration_comparison
from evaluate_bridge_endpoint_metrics import pose_errors


class NoiseSelectionTest(unittest.TestCase):
    def test_independent_training_episodes_and_same_task_donors(self):
        rows = []
        for task in range(5):
            for record in range(10):
                for start in (0, 4):
                    rows.append(dict(partition="train", task=str(task), shard=str(task),
                                     record_index=record, start_index=start))
        rows.insert(0, dict(partition="validation", task="0", shard="heldout", record_index=0))
        selected, donors = select_training(rows)
        self.assertEqual(len(selected), 40)
        self.assertEqual(len({(r['shard'], r['record_index']) for r in selected}), 40)
        for i, donor in enumerate(donors):
            self.assertNotEqual(i, donor)
            self.assertEqual(selected[i]['task'], selected[donor]['task'])
            self.assertEqual(selected[i]['partition'], 'train')
        self.assertTrue(all(r['start_index'] == 0 for r in selected))

    def test_incomplete_task_population_is_rejected(self):
        with self.assertRaises(ValueError):
            select_training([])

    def test_restoration_improvement_sign_and_quaternion_equivalence(self):
        target = np.zeros((1, 2, 7), np.float32)
        target[..., 6] = 1
        noisy = target.copy()
        noisy[..., 0] = .2
        noisy[..., 3:] = [0, 0, np.sin(np.pi/12), np.cos(np.pi/12)]
        restored = target.copy()
        restored[..., 6] = -1
        result = restoration_comparison(noisy, noisy, restored, target, pose_errors)
        self.assertAlmostEqual(result['input_position_cm'], 2, places=5)
        self.assertAlmostEqual(result['rotation_improvement_deg'], 30, places=3)
        self.assertEqual(result['rotation_improved_target_fraction'], 1)
        worse = restoration_comparison(target, target, noisy, target, pose_errors)
        self.assertLess(worse['position_improvement_cm'], 0)
        self.assertLess(worse['rotation_improvement_deg'], 0)


if __name__ == '__main__':
    unittest.main()

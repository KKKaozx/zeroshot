import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from diagnose_multiscale_noise import select_training


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


if __name__ == '__main__':
    unittest.main()

"""Ensure test omission is explicit and cannot relax ordinary split validation."""
import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from train import validate_split_indices
from evaluate_multitask_pilot import metrics


class FakeDataset:
    def __init__(self,selection=None): self.bridge_episode_selection=selection
    def __len__(self): return 4
    def group_key(self,i): return i//2


class PilotSplitTests(unittest.TestCase):
    def test_train_development_only(self):
        data=FakeDataset([{'partition':'train'},{'partition':'validation'}])
        validate_split_indices(data,{'train':[0,1],'validation':[2,3],'test':[]})

    def test_ordinary_empty_test_rejected(self):
        with self.assertRaises(ValueError):
            validate_split_indices(FakeDataset(),{'train':[0,1],'validation':[2,3],'test':[]})

    def test_pilot_trajectory_leakage_rejected(self):
        data=FakeDataset([{'partition':'train'},{'partition':'validation'}])
        with self.assertRaises(ValueError):
            validate_split_indices(data,{'train':[0,2],'validation':[1,3],'test':[]})

    def test_pilot_missing_window_rejected(self):
        data=FakeDataset([{'partition':'train'},{'partition':'validation'}])
        with self.assertRaises(ValueError):
            validate_split_indices(data,{'train':[0,1],'validation':[2],'test':[]})

    def test_metrics_quaternion_sign_and_target_pairs(self):
        target=np.zeros((3,16,8),np.float32)
        target[...,6]=1
        target[:,:8,7]=1
        target[:,8:,7]=-1
        prediction=target.copy()
        prediction[...,3:7]*=-1
        result=metrics(prediction,target)
        self.assertEqual(result['position_cm'],0)
        self.assertEqual(result['rotation_deg'],0)
        self.assertEqual(result['balanced_accuracy'],1)
        self.assertEqual(result['open_to_closed_pairs'],3)
        self.assertEqual(result['open_to_closed_pair_accuracy'],1)
        self.assertEqual(result['closed_to_open_pairs'],0)
        self.assertIsNone(result['closed_to_open_pair_accuracy'])


if __name__=='__main__': unittest.main()

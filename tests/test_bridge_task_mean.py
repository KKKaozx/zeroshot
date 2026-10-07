import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'code'))
from evaluate_bridge_task_mean import mean_pose


class MeanTests(unittest.TestCase):
    def test_opposite_quaternion_signs_do_not_cancel(self):
        poses = np.zeros((2,16,7),np.float32)
        poses[0,:,6] = 1
        poses[1,:,6] = -1
        mean = mean_pose(poses)
        np.testing.assert_allclose(np.abs(mean[:,6]),1)
        np.testing.assert_allclose(np.linalg.norm(mean[:,3:7],axis=-1),1)

    def test_horizon_means_remain_distinct(self):
        poses = np.zeros((2,16,7),np.float32)
        poses[:,:,6] = 1
        poses[0,:,0] = np.arange(16)
        poses[1,:,0] = np.arange(16)+2
        np.testing.assert_allclose(mean_pose(poses)[:,0],np.arange(16)+1)


if __name__ == '__main__':
    unittest.main()

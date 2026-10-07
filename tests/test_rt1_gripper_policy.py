import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from dataset import rt1_relative_gripper_commands


class RelativeGripperTests(unittest.TestCase):
    def test_hold_after_close_and_reopen(self):
        np.testing.assert_array_equal(rt1_relative_gripper_commands([0,1,0,0,-1,0]),[1,-1,-1,-1,1,1])

    def test_initial_closed_before_first_open(self):
        np.testing.assert_array_equal(rt1_relative_gripper_commands([0,0,-1,0]),[-1,-1,1,1])

    def test_no_change_assumes_open(self):
        np.testing.assert_array_equal(rt1_relative_gripper_commands([0,0,0]),[1,1,1])

    def test_deadband(self):
        np.testing.assert_array_equal(rt1_relative_gripper_commands([0,.05,.1,.2,.1,0]),[1,1,1,-1,-1,-1])

    def test_invalid_commands(self):
        for value in ([],[float('nan')],[float('inf')]):
            with self.assertRaises(ValueError):rt1_relative_gripper_commands(value)


if __name__=='__main__':unittest.main()

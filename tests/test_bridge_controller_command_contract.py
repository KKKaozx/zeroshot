"""CPU algebra tests; no simulator, learned policy or task-success claim."""
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'code'))
from bridge_controller_command_contract import (encode_bridge_increment, map_increment,
    advance_control_goal, native_feedback_for_goal)


def packet():
    return encode_bridge_increment([.01,-.02,.005,.12,-.08,.19,1.],
                                   position_divisor=.1, open_positive_command=1.)


def transform(rotation, position):
    T = np.eye(4); T[:3,:3] = rotation; T[:3,3] = position
    return T


class CommandContractTests(unittest.TestCase):
    def test_noncommuting_euler_delta_matches_homogeneous_collector_goal_update(self):
        p = packet(); pivot = np.array([.3,.1,.2]); prior = pivot+[.02,-.01,.03]
        R0 = Rotation.from_euler('zyx',[.4,.2,-.3]).as_matrix()
        Rx = Rotation.from_rotvec([.12,0,0]).as_matrix()
        Ry = Rotation.from_rotvec([0,-.08,0]).as_matrix()
        Rz = Rotation.from_rotvec([0,0,.19]).as_matrix()
        T = transform(np.eye(3),pivot) @ transform(Rz@Ry@Rx,[.01,-.02,.005]) @ transform(np.eye(3),-pivot) @ transform(R0,prior)
        out = advance_control_goal(p, previous_goal_position=prior,
            previous_goal_quaternion=Rotation.from_matrix(R0).as_quat(), measured_pivot_position=pivot, workspace_bounds=None)
        np.testing.assert_allclose(out['position'],T[:3,3],atol=1e-14)
        np.testing.assert_allclose(Rotation.from_quat(out['quaternion_xyzw']).as_matrix(),T[:3,:3],atol=1e-14)
        self.assertGreater(np.linalg.norm(out['position']-(pivot+[.01,-.02,.005])),.01)

    def test_nonidentity_axis_and_unit_mapping_before_goal_update(self):
        p = packet(); A = Rotation.from_euler('z',.6).as_matrix()
        mapped = map_increment(p, source_axes_to_target=A, target_meters_per_source_unit=.01,
            target_position_divisor=.1, common_control_point_convention='Synthetic coincident control points only')
        np.testing.assert_allclose(mapped['values'][:3]*.1,.01*A@np.array([.01,-.02,.005]),atol=1e-14)
        expected = A@Rotation.from_euler('xyz',[.12,-.08,.19]).as_matrix()@A.T
        np.testing.assert_allclose(Rotation.from_quat(mapped['values'][3:7]).as_matrix(),expected,atol=1e-14)
        self.assertFalse(mapped['physical_calibration_verified'])

    def test_same_current_state_action_allows_different_goals_when_history_missing(self):
        p = packet(); current = np.array([.3,.1,.2]); q = [0,0,0,1]
        a = advance_control_goal(p, previous_goal_position=current, previous_goal_quaternion=q,
                                 measured_pivot_position=current, workspace_bounds=None)
        b = advance_control_goal(p, previous_goal_position=current+[.02,0,0], previous_goal_quaternion=q,
                                 measured_pivot_position=current, workspace_bounds=None)
        self.assertAlmostEqual(np.linalg.norm(a['position']-b['position']),.02)

    def test_feedback_reuses_goal_without_fourfold_accumulation_and_clips_only_explicitly(self):
        p = encode_bridge_increment([.01,0,0,0,0,0,1],position_divisor=.1,open_positive_command=1.)
        goal = advance_control_goal(p,previous_goal_position=[.3,0,.2],previous_goal_quaternion=[0,0,0,1],
                                    measured_pivot_position=[.3,0,.2],workspace_bounds=None)
        saved = goal['position'].copy()
        for _ in range(4):
            out = native_feedback_for_goal(goal,measured_position=[.3,0,.2],measured_quaternion=[0,0,0,1],
                output_scales=[.05]*3+[.5]*3,saturation_policy='reject')
            np.testing.assert_allclose(out['native_action'],[.2,0,0,0,0,0,-1],atol=1e-14)
        np.testing.assert_array_equal(goal['position'],saved)
        clipped = advance_control_goal(p,previous_goal_position=[.3,0,.2],previous_goal_quaternion=[0,0,0,1],
            measured_pivot_position=[.3,0,.2],workspace_bounds=([0,-1,0],[.305,1,1]))
        self.assertTrue(clipped['workspace_clipped'])
        self.assertAlmostEqual(clipped['position'][0],.305)

    def test_old_pose_contract_unknown_grip_and_missing_history_are_not_silently_substituted(self):
        p = packet(); p['format']='source_cartesian_delta_pose_open_positive_v1'
        args=dict(previous_goal_position=[0,0,0],previous_goal_quaternion=[0,0,0,1],measured_pivot_position=[0,0,0],workspace_bounds=None)
        with self.assertRaisesRegex(ValueError,'different contract'):advance_control_goal(p,**args)
        p = encode_bridge_increment([0,0,0,0,0,0,.5],position_divisor=.1,open_positive_command=None)
        g = advance_control_goal(p,**args)
        with self.assertRaisesRegex(ValueError,'Unknown gripper'):
            native_feedback_for_goal(g,measured_position=[0,0,0],measured_quaternion=[0,0,0,1],
                                     output_scales=[.05]*3+[.5]*3,saturation_policy='reject')
        with self.assertRaises(TypeError):advance_control_goal(packet(),measured_pivot_position=[0,0,0],workspace_bounds=None)


if __name__=='__main__':
    unittest.main(verbosity=2)

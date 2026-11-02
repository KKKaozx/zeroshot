"""Focused CPU checks of new static frame reader, never a robot experiment."""
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from libero_eef_frame_definition import fixed_body_to_site, controller_quaternion


class FrameDefinitionTest(unittest.TestCase):
    def test_independent_rigid_geometry_and_left_delta(self):
        #90deg body->child rotation and a translated child: expected site offset
        #is [1,1,0], not the naive sum [2,0,0].
        xml = '<mujoco><worldbody><body name="hand"><body name="mount" pos="1 0 0" quat="1 0 0 1"><site name="grip" pos="1 0 0"/></body></body></worldbody></mujoco>'
        spec = fixed_body_to_site(xml, quaternion_body_name='hand', position_site_name='grip')
        np.testing.assert_allclose(spec['position_site_in_quaternion_body_model_units'], [1, 1, 0], atol=1e-12)
        a = Rotation.from_euler('xyz', [20, -30, 40], degrees=True)
        b = Rotation.from_euler('xyz', [-45, 50, 15], degrees=True)
        ac = Rotation.from_quat(controller_quaternion(a.as_quat(), spec))
        bc = Rotation.from_quat(controller_quaternion(b.as_quat(), spec))
        np.testing.assert_allclose(ac.as_matrix(), a.as_matrix() @ np.array([[0,-1,0],[1,0,0],[0,0,1]]), atol=1e-12)
        np.testing.assert_allclose((bc * ac.inv()).as_matrix(), (b * a.inv()).as_matrix(), atol=1e-12)

    def test_reject_unresolved_or_moving_frames(self):
        cases = [
            '<mujoco><worldbody><body name="hand"/><body name="other"><site name="grip"/></body></worldbody></mujoco>',
            '<mujoco><worldbody><body name="hand"><body name="finger"><joint/><site name="grip"/></body></body></worldbody></mujoco>',
            '<mujoco><worldbody><body name="hand"><site name="grip" euler="0 0 1"/></body></worldbody></mujoco>',
            '<mujoco><default><site quat="1 0 0 1"/></default><worldbody><body name="hand"><site name="grip"/></body></worldbody></mujoco>']
        for xml in cases:
            with self.subTest(xml=xml), self.assertRaises(ValueError):
                fixed_body_to_site(xml, quaternion_body_name='hand', position_site_name='grip')


if __name__ == '__main__':
    unittest.main()

"""Saved-array geometry/partial-window checks; no neural network or CUDA execution."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('fixed8_array_audit',ROOT/'tools/audit_oxe_fixed8_download_233426.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


class Tests(unittest.TestCase):
    def test_quaternion_sign_normalization_and_known_angles(self):
        q=np.array([0.,0.,np.sqrt(.5),np.sqrt(.5)]);identity=np.array([0.,0.,0.,1.])
        self.assertAlmostEqual(float(audit.angle_deg(q,-q)),0,places=6)
        self.assertAlmostEqual(float(audit.angle_deg(q*3,identity)),90,places=10)
        self.assertAlmostEqual(float(audit.angle_deg([1,0,0,0],identity)),180,places=10)

    def test_partial_window_endpoint_and_draw_dispersion(self):
        target=np.zeros((8,16,8));target[...,6]=1
        mask=np.ones_like(target);mask[:,2:]=0
        pred=np.repeat(target[None],3,0)
        pred[:,:,1,3:7]=[0,0,np.sqrt(.5),np.sqrt(.5)]
        pred[:,:,2:]=99  # invalid tail cannot change endpoint or any scored error
        result=audit.rotation_details(pred,target,mask)
        self.assertAlmostEqual(result['path_rotation_deg'],45,places=10)
        self.assertTrue(all(r['valid_rotation_steps']==2 for r in result['per_window']))
        self.assertAlmostEqual(result['per_window'][0]['endpoint_rotation_deg'],90,places=10)
        self.assertAlmostEqual(result['between_draw_pairwise_rotation_deg'],0,places=5)
        self.assertEqual(result['by_future_step']['4']['valid_windows'],0)

    def test_actual_download_raw_metrics_history_and_gate(self):
        result=audit.audit(ROOT/'reports/oxe_fixed8_result_233426/cluster')
        self.assertTrue(result['all_original_metrics_and_fit_gate_recomputed'])
        self.assertTrue(result['all512history_lr_indices_noise_records_checked'])
        self.assertFalse(result['fit_decision']['fixed8_pose_fit_gate_passed'])
        self.assertEqual(result['fit_decision']['windows_passing_all_four'],1)
        self.assertTrue(all(r['path_rotation_improved_vs_before'] for r in result['per_window']))
        self.assertEqual(result['model_calls'],0)


if __name__=='__main__':unittest.main(verbosity=2)

"""CPU tests of pose preservation, original features and execution conversion."""
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
import torch
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT / 'code'))
if (ROOT / 'vendor/bridge_growth128_v1').is_dir():
    sys.path.append(str(ROOT / 'vendor/bridge_growth128_v1'))
from dino_gripper_sidecar import (GripperSidecar, replace_gripper, SOURCE_SHA,
    immutable_pose_digest, source_state_features, measured_feature, gripper_metrics)
from legacy_dino_gripper_execution import tool_waypoints_to_target,bind_panda_measured_observation
from libero_pose_transfer import SourceTargetPoseMapping


class SidecarTest(unittest.TestCase):
    def test_original_sampler_pose_and_rng_preserved_after_gripper_update(self):
        torch.set_num_threads(2)
        path = ROOT / 'vendor/bridge_growth128_v1/models.py'
        if not path.exists(): path = ROOT / 'cluster_packages/bridge_backbone_pair_a40_v1/frozen/bridge_growth128_v1/models.py'
        spec = importlib.util.spec_from_file_location('sidecar_original_models_test',path)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        class Decoder(torch.nn.Module):
            def forward(self,x,t,c):
                result = x*.1; result[...,6] += .9
                return result
        class Source(torch.nn.Module):
            sample = module.RobotAdapterModel.sample
            def __init__(self):
                super().__init__(); self.keep = torch.nn.Parameter(torch.ones(1),requires_grad=False)
                self.diffusion_decoder = Decoder(); self.action_dim=8; self.chunk_size=16
                self.separate_gripper_head=True; self.decoder_type='diffusion'; self.num_diffusion_steps=100
                self.diffusion_prediction_type='sample'; self.clip_denoised=True; self.max_normalized_position=3
                self.gripper_target_mode='state'
                betas=module.diffusion_betas(100,'squaredcos_cap_v2'); alphas=1-betas; bars=torch.cumprod(alphas,0)
                previous=torch.cat([torch.ones(1),bars[:-1]])
                for key,value in [('alphas',alphas),('alpha_bars',bars),
                    ('posterior_mean_x0',betas*previous.sqrt()/(1-bars)),
                    ('posterior_mean_xt',(1-previous)*alphas.sqrt()/(1-bars)),
                    ('posterior_variance',betas*(1-previous)/(1-bars))]: self.register_buffer(key,value)
            def predict_gripper_logits(self,c,poses,current): return torch.zeros(len(c),16)
        source = Source().eval().requires_grad_(False); before=immutable_pose_digest(source)
        torch.manual_seed(42); head=GripperSidecar(); context=torch.randn(2,1024); current=torch.zeros(2,1)
        torch.manual_seed(1101); baseline=source.sample(context,current); rng=torch.get_rng_state().clone()
        opt=torch.optim.Adam(head.parameters(),lr=1e-3)
        logits=head(context,baseline[...,:7],current)
        torch.nn.functional.binary_cross_entropy_with_logits(logits,torch.zeros_like(logits)).backward(); opt.step()
        self.assertTrue(any(p.grad is not None for p in head.parameters()))
        self.assertTrue(all(p.grad is None for p in source.parameters()))
        head.eval(); torch.manual_seed(1101)
        actual=replace_gripper(source,head,context,current,source_checkpoint_sha256=SOURCE_SHA)
        self.assertTrue(torch.equal(actual[...,:7],baseline[...,:7]))
        self.assertTrue(torch.equal(torch.get_rng_state(),rng))
        self.assertEqual(before,immutable_pose_digest(source))
        with self.assertRaises(ValueError): replace_gripper(source,head,context,current,source_checkpoint_sha256='wrong')

    def test_original_state_and_tool_geometry(self):
        rotation=Rotation.from_euler('xyz',[.2,-.3,.5]); p=np.array([.3,.1,.2])
        stats={'xyz_mean':[.1,.2,.3],'xyz_std':[.2,.3,.4]}
        features=source_state_features(p,rotation.as_quat(),stats)
        np.testing.assert_allclose(features[:3],(p-stats['xyz_mean'])/stats['xyz_std'],atol=1e-7)
        np.testing.assert_allclose(features[3:],rotation.as_matrix()[:,:2].T.reshape(-1),atol=1e-7)
        self.assertAlmostEqual(float(measured_feature(1.1)[0]),1.2,places=6) #No clipping.
        mapping=SourceTargetPoseMapping({'format':'source-target-pose-mapping-v1','source':'Bridge',
            'target_meters_per_source_unit':1.,'source_axes_to_target':np.eye(3).tolist(),
            'source_origin_in_target_m':[0,0,0],'right_tool_alignment':np.eye(3).tolist(),
            'source_eef_to_point_source_units':[0,0,0],'target_eef_to_point_m':[0,0,0],'evidence':'test only'})
        values=np.zeros((16,8)); values[:,:3]=[1,0,0]; values[:,3:7]=Rotation.from_euler('x',.4).as_quat(); values[:,7]=-1
        actual=tool_waypoints_to_target(values,mapping,p,rotation.as_quat())
        np.testing.assert_allclose(actual[0,:3],rotation.apply([1,0,0]),atol=1e-12)
        np.testing.assert_allclose((Rotation.from_quat(actual[0,3:7])*rotation).as_matrix(),
            (rotation*Rotation.from_euler('x',.4)).as_matrix(),atol=1e-12)

    def test_transition_metrics_and_causal_measured_binding(self):
        truth=np.ones((2,16),bool); truth[0,5:]=False; truth[1,:8]=False
        metrics=gripper_metrics(truth[None],truth)
        self.assertEqual(metrics['balanced_accuracy'],1.)
        self.assertEqual(metrics['transitions']['closing']['f1'],1.)
        self.assertEqual(metrics['transitions']['opening']['f1'],1.)
        obs={'robot0_gripper_qpos':np.array([.02,-.02]),'robot0_eef_pos':np.ones(3),
             'robot0_eef_quat':np.array([0,0,0,1.])}
        callback=bind_panda_measured_observation(lambda request:request,obs)
        request={'reference_position_m':np.ones(3),'reference_quaternion_xyzw':np.array([0,0,0,1.])}
        self.assertEqual(callback(request)['measured_open_fraction'],.5)
        request['reference_position_m']=np.zeros(3)
        with self.assertRaises(ValueError): callback(request)


if __name__ == '__main__': unittest.main()

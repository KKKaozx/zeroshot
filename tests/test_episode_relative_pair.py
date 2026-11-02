"""Representation invariance and the new training/loading path, not task scores."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation
import torch

ROOT=Path(__file__).resolve().parents[1]
BUNDLE=ROOT/'deployment/episode_relative_pose_pair_v1'
sys.path[:0]=[str(ROOT/'code'),str(BUNDLE)]
from episode_relative_pose_state import pose_features,absolute_pose6_features,EpisodeStateReference,RELATIVE_FORMAT,ABSOLUTE_FORMAT
from pose_core import CachedPoseCore,RobotAdapterModel
from pair_metrics import summarize,decision

STATS={'xyz_mean':[.3,.01,.09],'xyz_std':[.07,.08,.06]}


class RelativePairTests(unittest.TestCase):
    def test_rigid_world_invariance_and_reference(self):
        rng=np.random.default_rng(42);p=rng.normal(size=(8,3));p0=rng.normal(size=(8,3))
        r=Rotation.random(8,random_state=rng);r0=Rotation.random(8,random_state=rng)
        a=Rotation.from_euler('xyz',[.4,-.7,1.2]);b=np.array([3.,-5.,10.])
        before=pose_features(p,r.as_quat(),STATS,mode=RELATIVE_FORMAT,initial_position=p0,initial_quaternion=r0.as_quat())
        after=pose_features(a.apply(p)+b,(a*r).as_quat(),STATS,mode=RELATIVE_FORMAT,
            initial_position=a.apply(p0)+b,initial_quaternion=(a*r0).as_quat())
        np.testing.assert_allclose(before,after,rtol=0,atol=2e-6)
        initial=pose_features(p0,r0.as_quat(),STATS,mode=RELATIVE_FORMAT,initial_position=p0,initial_quaternion=r0.as_quat())
        np.testing.assert_allclose(initial,np.tile([0,0,0,1,0,0,0,1,0],(8,1)),atol=1e-6)
        absolute=pose_features(p,r.as_quat(),STATS,mode=ABSOLUTE_FORMAT)
        shifted=pose_features(p+b,r.as_quat(),STATS,mode=ABSOLUTE_FORMAT)
        self.assertGreater(np.max(np.abs(absolute-shifted)),10)

    def test_anchor_copy_explicit_reset_and_old_control_formula(self):
        reference=EpisodeStateReference(STATS,RELATIVE_FORMAT);p=np.array([0.,0.,1.]);q=np.array([0.,0.,0.,1.])
        with self.assertRaises(ValueError):reference(p,q,STATS)
        reference.begin_episode(p,q);p[2]=2
        value=reference(p,q,STATS)
        self.assertAlmostEqual(float(value[2]),1/.06,places=5)
        reference.begin_episode(p,q)
        self.assertEqual(float(reference(p,q,STATS)[2]),0.)
        with self.assertRaises(ValueError):reference.begin_episode(np.zeros((2,3)),np.tile(q,(2,1)))
        observed=np.array([[.4,.2,.3,.1,.2,.3]],dtype=np.float32)
        expected=pose_features(observed[:,:3],Rotation.from_euler('xyz',observed[:,3:]).as_quat(),STATS,mode=ABSOLUTE_FORMAT)
        np.testing.assert_allclose(absolute_pose6_features(observed,STATS),expected,atol=1e-6)

    def test_native_methods_toy_pair_and_optimizer_roundtrip(self):
        torch.set_num_threads(2)
        cfg={'frozen_context_dim':32,'action':{'max_normalized_position':3.},'model':{
            'num_adapter_layers':1,'cached_text_dim':24,'attention_dim':16,'num_attention_heads':4,
            'dropout':.1,'adapter_pooling':'cls_patch_mean','diffusion_prediction_type':'sample',
            'clip_denoised':True,'beta_schedule':'squaredcos_cap_v2','diffusion_down_dims':[8,16],
            'diffusion_step_embed_dim':16}}
        original=CachedPoseCore(cfg);initial=original.pose_state()
        self.assertIs(CachedPoseCore.sample,RobotAdapterModel.sample)
        self.assertIs(CachedPoseCore.diffusion_loss,RobotAdapterModel.diffusion_loss)
        visual=torch.randn(2,5,32);language=torch.randn(2,4,24);mask=torch.ones(2,4,dtype=torch.long)
        target=torch.randn(2,16,8);current=torch.zeros(2,1);states=[torch.randn(2,9),torch.randn(2,9)]
        with tempfile.TemporaryDirectory(dir=ROOT/'reports') as folder:
            for arm,state in enumerate(states):
                model=CachedPoseCore(cfg).restore(initial);opt=torch.optim.Adam(model.parameters(),lr=1e-5)
                for step in range(2):
                    model.train();torch.manual_seed(430000+step);ctx=model.context(visual,language,mask,state)
                    torch.manual_seed(420000+step);pred,truth,_=model.diffusion_loss(target,ctx,current)
                    self.assertTrue(torch.equal(truth,target[...,:7]))
                    loss=torch.nn.functional.mse_loss(pred,truth);opt.zero_grad();loss.backward();opt.step()
                path=Path(folder)/f'{arm}.pt';torch.save({'state':model.pose_state(),'optimizer':opt.state_dict()},path)
                saved=torch.load(path,weights_only=False);other=CachedPoseCore(cfg).restore(saved['state'])
                self.assertTrue(all(torch.equal(v,other.pose_state()[k]) for k,v in saved['state'].items()))
                self.assertTrue(all(int(v['step'])==2 for v in saved['optimizer']['state'].values()))
                model.eval()
                with torch.inference_mode():
                    ctx=model.context(visual,language,mask,state);sample=model.sample(ctx,current)
                self.assertEqual(tuple(sample.shape),(2,16,8));self.assertTrue(torch.isfinite(sample).all())
                self.assertTrue(all(torch.equal(v,original.pose_state()[k]) for k,v in initial.items()))

    def test_real_input_binding_and_prospective_decision(self):
        folder=ROOT/'reports/episode_relative_pair_preparation_20261031'
        with np.load(folder/'inputs.npz',allow_pickle=False) as z:
            self.assertEqual(z['absolute_features'].shape,(1876,9));self.assertEqual(z['relative_features'].shape,(1876,9))
            self.assertTrue(np.array_equal(z['absolute_features'],absolute_pose6_features(z['observed_pose6'],json.loads((folder/'inputs_receipt.json').read_text())['state_statistics'])))
        metric={'episode_position_cm':5.,'episode_rotation_deg':12.}
        baseline={role:metric.copy() for role in ('train','development')}
        arms={mode:copy.deepcopy(baseline) for mode in ('absolute','relative')}
        self.assertTrue(decision(arms,baseline)['candidate_for_review_of_one_engineering_trial'])
        arms['relative']['development']['episode_position_cm']=6.
        self.assertFalse(decision(arms,baseline)['candidate_for_review_of_one_engineering_trial'])
        self.assertFalse(decision(arms,baseline)['automatic_rollout'])

    def test_old_checkpoint_cannot_be_relabelled(self):
        from online_policy import load_trained_policy
        with tempfile.TemporaryDirectory(dir=ROOT/'reports') as folder:
            path=Path(folder)/'old.pt';torch.save({'format':'old','update':8192},path)
            with self.assertRaisesRegex(ValueError,'trained versioned'):
                load_trained_policy(ROOT,path,clip_assets='',dino_assets='',mapping={})


if __name__=='__main__':unittest.main()

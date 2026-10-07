"""Check that tracing preserves production sampling and paired initial noise."""
import sys
from pathlib import Path
from types import MethodType, SimpleNamespace
import unittest
import torch
from torch import nn

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from models import RobotAdapterModel, diffusion_betas
from probe_bridge_conditioning import sample_with_trace, select_windows


class Oracle(nn.Module):
    def __init__(self,clean):
        super().__init__()
        self.clean=clean
    def forward(self,noisy,timestep,context):
        return self.clean


def fixture():
    clean=torch.zeros(2,16,7)
    clean[:,:,:3]=.2
    clean[:,:,6]=1
    beta=diffusion_betas(100,'squaredcos_cap_v2')
    alpha=1-beta
    bars=alpha.cumprod(0)
    prior=torch.cat([torch.ones(1),bars[:-1]])
    model=SimpleNamespace(action_dim=8,chunk_size=16,separate_gripper_head=True,
        decoder_type='diffusion',num_diffusion_steps=100,diffusion_prediction_type='sample',
        clip_denoised=False,max_normalized_position=3.,gripper_target_mode='state',
        alphas=alpha,alpha_bars=bars,posterior_variance=(beta*(1-prior)/(1-bars)).clamp(min=1e-20),
        diffusion_decoder=Oracle(clean),predict_gripper_logits=lambda c,p,g:torch.ones(2,16))
    model.sample=MethodType(RobotAdapterModel.sample,model)
    return model,clean


class TraceTests(unittest.TestCase):
    def test_full_selection_retains_overlapping_windows_and_excludes_reserved(self):
        dataset=SimpleNamespace(samples=[dict(file_path='shard',record_index=r,start_index=s)
            for r,s in ((0,0),(0,4),(1,0),(2,0))])
        selection=[dict(shard='shard',record_index=r,instruction='task') for r in range(3)]
        rows=select_windows(dataset,{'train':[1,0],'validation':[2],'test':[3]},
            selection,8,all_train=True)
        self.assertEqual([r['dataset_index'] for r in rows],[0,1,2])
        self.assertEqual([r['partition'] for r in rows],['train','train','validation'])

    def test_hook_does_not_change_output_or_rng(self):
        model,clean=fixture()
        context,current=torch.zeros(2,1024),torch.zeros(2,1)
        torch.manual_seed(12)
        expected=model.sample(context,current)
        rng=torch.get_rng_state().clone()
        torch.manual_seed(12)
        actual,trace=sample_with_trace(model,context,current,(99,74,0))
        self.assertTrue(torch.equal(expected[:,:,:7],actual))
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertLess(float((actual-clean).abs().max()),2e-5)
        self.assertEqual(len(model.diffusion_decoder._forward_hooks),0)

    def test_noise_suppression_keeps_initial_state_and_prediction(self):
        model,_=fixture()
        context,current=torch.zeros(2,1024),torch.zeros(2,1)
        variance=model.posterior_variance.clone()
        torch.manual_seed(12)
        _,native=sample_with_trace(model,context,current,(99,74,0))
        try:
            model.posterior_variance.zero_()
            torch.manual_seed(12)
            _,mean=sample_with_trace(model,context,current,(99,74,0))
            self.assertTrue((native[99][0]==mean[99][0]).all())
            self.assertTrue((native[99][1]==mean[99][1]).all())
            self.assertFalse((native[74][0]==mean[74][0]).all())
        finally:
            model.posterior_variance.copy_(variance)
        self.assertTrue(torch.equal(model.posterior_variance,variance))

    def test_hook_removed_on_sampler_error(self):
        model,_=fixture()
        def fail(*args):
            raise RuntimeError('Synthetic sampler failure')
        model.sample=fail
        with self.assertRaises(RuntimeError):
            sample_with_trace(model,torch.zeros(2,1024),torch.zeros(2,1),(99,))
        self.assertEqual(len(model.diffusion_decoder._forward_hooks),0)


if __name__=='__main__':
    unittest.main()

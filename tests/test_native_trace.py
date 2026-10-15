"""A trace must not alter stochastic sampling and must clean up on failure."""
import importlib.util
from pathlib import Path
import unittest
import torch

spec = importlib.util.spec_from_file_location('probe',Path(__file__).resolve().parents[1]/'code/diagnose_multiscale_noise.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class Decoder(torch.nn.Module):
    def forward(self, noisy, timestep, context):
        return noisy * .5 + context


class Model(torch.nn.Module):
    def __init__(self, fail=False):
        super().__init__()
        self.diffusion_decoder = Decoder()
        self.num_diffusion_steps = 4
        self.fail = fail

    def sample(self, context, current):
        latent = torch.randn(2,3,7)
        for step in reversed(range(4)):
            latent = self.diffusion_decoder(latent,torch.full((2,),step),context)
            if self.fail:
                raise RuntimeError('intentional failure')
            if step:
                latent = latent + torch.randn_like(latent) * .1
        return latent


class TraceTests(unittest.TestCase):
    def test_output_rng_and_observation_preserved(self):
        model = Model()
        torch.manual_seed(81)
        expected = model.sample(torch.tensor(.3),None)
        expected_rng = torch.get_rng_state().clone()
        torch.manual_seed(81)
        actual, trace = probe.traced_native(model,torch.tensor(.3),None,(3,0))
        self.assertTrue(torch.equal(expected,actual))
        self.assertTrue(torch.equal(expected_rng,torch.get_rng_state()))
        self.assertEqual(set(trace),{3,0})
        for latent, estimate in trace.values():
            self.assertTrue(torch.equal(estimate,latent*.5+.3))
        self.assertEqual(len(model.diffusion_decoder._forward_hooks),0)

    def test_cleanup_after_sampler_failure(self):
        model = Model(fail=True)
        with self.assertRaisesRegex(RuntimeError,'intentional'):
            probe.traced_native(model,torch.tensor(.3),None,(3,0))
        self.assertEqual(len(model.diffusion_decoder._forward_hooks),0)

    def test_actual_noise_capture_and_cleanup(self):
        model=Model(); original=torch.randn_like
        torch.manual_seed(91)
        expected=model.sample(torch.tensor(.3),None)
        expected_rng=torch.get_rng_state().clone()
        torch.manual_seed(91)
        actual,trace,noises=probe.capture_native_noise(torch,model,torch.tensor(.3),None)
        self.assertTrue(torch.equal(expected,actual))
        self.assertTrue(torch.equal(expected_rng,torch.get_rng_state()))
        self.assertIs(torch.randn_like,original)
        self.assertEqual(set(noises),{1,2,3})
        for step in (3,2,1):
            self.assertTrue(torch.equal(trace[step][1]+noises[step]*.1,trace[step-1][0]))
        with self.assertRaisesRegex(RuntimeError,'intentional'):
            probe.capture_native_noise(torch,Model(fail=True),torch.tensor(.3),None)
        self.assertIs(torch.randn_like,original)


if __name__ == '__main__':
    unittest.main()

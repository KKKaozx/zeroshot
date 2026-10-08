import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from evaluate_bridge_ddim_steps import ddim_sample, timestep_schedule


class OracleDecoder:
    def __init__(self, clean):
        self.clean = clean

    def __call__(self, noisy, timesteps, context):
        return self.clean.expand_as(noisy)


class OracleModel:
    max_normalized_position = 3.0

    def __init__(self, clean):
        betas = torch.linspace(0.0001, 0.02, 100)
        self.alpha_bars = torch.cumprod(1 - betas, dim=0)
        self.diffusion_decoder = OracleDecoder(clean)
        self.diffusion_prediction_type = "sample"


class EpsilonOracleDecoder:
    def __init__(self, model, clean):
        self.model = model
        self.clean = clean

    def __call__(self, noisy, timesteps, context):
        alpha = self.model.alpha_bars[timesteps].reshape(-1, 1, 1)
        return (noisy - alpha.sqrt() * self.clean) / (1 - alpha).sqrt()


class EpsilonOracleModel(OracleModel):
    def __init__(self, clean):
        super().__init__(clean)
        self.diffusion_prediction_type = "epsilon"
        self.diffusion_decoder = EpsilonOracleDecoder(self, clean)


class DdimStepsTest(unittest.TestCase):
    def test_schedules_have_requested_unique_endpoints(self):
        self.assertEqual(timestep_schedule(1), [99])
        for count in (2, 4, 8, 16, 32, 50, 100):
            schedule = timestep_schedule(count)
            self.assertEqual((schedule[0], schedule[-1]), (99, 0))
            self.assertEqual(len(schedule), len(set(schedule)))

    def test_oracle_x0_is_recovered_for_every_schedule(self):
        clean = torch.tensor([[[0.2, -0.3, 0.4, 0.0, 0.0, 0.0, 1.0]]])
        model = OracleModel(clean)
        initial = torch.randn_like(clean)
        for count in (1, 2, 4, 8, 16, 32, 50, 100):
            result = ddim_sample(torch, model, None, initial, count)
            self.assertTrue(torch.allclose(result, clean, atol=1e-6), count)

    def test_one_step_does_not_add_intermediate_component_clipping(self):
        clean = torch.tensor([[[0.0, 0.0, 0.0, 2.0, -2.0, 0.0, 1.0]]])
        result = ddim_sample(torch, OracleModel(clean), None, torch.zeros_like(clean), 1)
        self.assertTrue(torch.equal(result, clean))

    def test_epsilon_oracle_recovers_x0_for_every_schedule(self):
        clean = torch.tensor([[[0.2, -0.3, 0.4, 0.0, 0.0, 0.0, 1.0]]])
        model = EpsilonOracleModel(clean)
        initial = torch.randn_like(clean)
        for count in (1, 2, 4, 8, 16, 32, 50, 100):
            result = ddim_sample(torch, model, None, initial, count)
            self.assertTrue(torch.allclose(result, clean, atol=2e-5), count)


if __name__ == "__main__":
    unittest.main()

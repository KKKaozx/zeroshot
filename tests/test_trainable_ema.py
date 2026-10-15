"""Numerical EMA correctness and isolation from stochastic optimization."""
import importlib.util
from pathlib import Path
import unittest
import torch

spec = importlib.util.spec_from_file_location('ema_runner', Path(__file__).resolve().parents[1] / 'code/run_bridge_official_unet_training.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class EMATests(unittest.TestCase):
    def test_warmup_recurrence_and_frozen_exclusion(self):
        model = torch.nn.Linear(1, 1, bias=False).double()
        model.register_parameter('frozen', torch.nn.Parameter(torch.tensor([11.]), requires_grad=False))
        ema = runner.TrainableEMA(model)
        self.assertEqual(set(ema.shadow), {'weight'})
        for index, value in enumerate((2., 4., 8.)):
            with torch.no_grad():
                model.weight.fill_(value)
            ema.step(model)
            if index < 2:
                self.assertEqual(ema.shadow['weight'].item(), value)
        decay = 1 - 2 ** -0.75
        self.assertAlmostEqual(ema.shadow['weight'].item(), 4 * decay + 8 * (1 - decay), places=12)
        self.assertEqual(model.weight.item(), 8.)
        self.assertEqual(model.frozen.item(), 11.)
        self.assertEqual(ema.optimization_step, 3)

    def test_ema_does_not_change_raw_optimizer_or_rng(self):
        def optimize(use_ema):
            torch.manual_seed(42)
            model = torch.nn.Sequential(torch.nn.Linear(3, 5), torch.nn.Dropout(.25), torch.nn.Linear(5, 2))
            optimizer = torch.optim.AdamW(model.parameters(), lr=.003)
            ema = runner.TrainableEMA(model) if use_ema else None
            for _ in range(30):
                x, target = torch.randn(4, 3), torch.randn(4, 2)
                optimizer.zero_grad()
                (model(x) - target).square().mean().backward()
                optimizer.step()
                if ema:
                    before = torch.get_rng_state().clone()
                    live = {k: v.detach().clone() for k, v in model.named_parameters()}
                    ema.step(model)
                    self.assertTrue(torch.equal(before, torch.get_rng_state()))
                    self.assertTrue(all(torch.equal(v, live[k]) for k, v in model.named_parameters()))
            return model.state_dict(), torch.get_rng_state()
        control, control_rng = optimize(False)
        observed, observed_rng = optimize(True)
        self.assertTrue(all(torch.equal(control[k], observed[k]) for k in control))
        self.assertTrue(torch.equal(control_rng, observed_rng))


if __name__ == '__main__':
    unittest.main()

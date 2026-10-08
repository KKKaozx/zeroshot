import unittest
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from diffusion_decoder import (
    ConditionalDiffusionDecoder,
    MultiscaleConditionalUnet1D,
)


class MultiscaleConditionalUnetTest(unittest.TestCase):
    def test_shape_gradient_and_condition_dependence(self):
        torch.manual_seed(7)
        model = MultiscaleConditionalUnet1D(
            action_dim=7,
            chunk_size=16,
            context_dim=32,
            down_dims=(16, 32, 64),
            diffusion_step_embed_dim=16,
        )
        actions = torch.randn(2, 16, 7, requires_grad=True)
        context = torch.randn(2, 32, requires_grad=True)
        timesteps = torch.tensor([3, 91])
        output = model(actions, timesteps, context)
        self.assertEqual(tuple(output.shape), (2, 16, 7))
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(actions.grad).all())
        self.assertGreater(torch.count_nonzero(context.grad).item(), 0)

        with torch.no_grad():
            changed_context = model(actions.detach(), timesteps, context.detach().flip(0))
            changed_time = model(actions.detach(), timesteps.flip(0), context.detach())
        self.assertGreater(torch.max(torch.abs(output.detach() - changed_context)).item(), 0)
        self.assertGreater(torch.max(torch.abs(output.detach() - changed_time)).item(), 0)

    def test_full_layout_is_larger_than_compact_control(self):
        compact = ConditionalDiffusionDecoder(
            action_dim=7, chunk_size=16, context_dim=1024, hidden_dim=256
        )
        multiscale = MultiscaleConditionalUnet1D(
            action_dim=7, chunk_size=16, context_dim=1024
        )
        compact_parameters = sum(p.numel() for p in compact.parameters())
        multiscale_parameters = sum(p.numel() for p in multiscale.parameters())
        self.assertGreater(multiscale_parameters, compact_parameters)

    def test_rejects_incompatible_sequence_length(self):
        with self.assertRaises(ValueError):
            MultiscaleConditionalUnet1D(chunk_size=10)


if __name__ == "__main__":
    unittest.main()

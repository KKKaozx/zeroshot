import unittest
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from diffusion_decoder import (
    ConditionalDiffusionDecoder,
    MultiscaleConditionalUnet1D,
)
from models import RobotAdapterModel


class Encoder(torch.nn.Module):
    def __init__(self, hidden_size=32):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))

    def forward(self, pixel_values=None, input_ids=None, attention_mask=None):
        values = pixel_values if pixel_values is not None else input_ids
        features = values.float().reshape(values.shape[0], -1, 1) * self.weight
        return SimpleNamespace(
            last_hidden_state=features,
            pooler_output=features[:, 0],
        )


class FakeClip(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_model = Encoder()
        self.text_model = Encoder()


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

    @patch("models.CLIPModel.from_pretrained", return_value=FakeClip())
    def test_robot_model_selects_multiscale_without_changing_default(self, _):
        base = {
            "model": {
                "name": "fake",
                "action_dim": 8,
                "chunk_size": 16,
                "decoder_type": "diffusion",
                "fusion_type": "cross_attention",
                "num_adapter_layers": 1,
                "attention_dim": 16,
                "num_attention_heads": 4,
                "dropout": 0,
                "separate_gripper_head": True,
            },
            "action": {},
        }
        compact = RobotAdapterModel(base)
        self.assertIsInstance(compact.diffusion_decoder, ConditionalDiffusionDecoder)
        multiscale_config = {"model": dict(base["model"]), "action": {}}
        multiscale_config["model"].update(
            diffusion_architecture="multiscale",
            diffusion_down_dims=(16, 32, 64),
            diffusion_step_embed_dim=16,
        )
        multiscale = RobotAdapterModel(multiscale_config)
        self.assertIsInstance(multiscale.diffusion_decoder, MultiscaleConditionalUnet1D)


if __name__ == "__main__":
    unittest.main()

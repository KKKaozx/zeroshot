"""Isolated post-Adapter readout prototype; not wired into any policy or launcher.

Input is already fused CLS + patch tokens. No encoder, labels or future inputs.
Apply the existing Adapter.output_norm to the returned pooled vector externally.
"""
import torch
from torch import nn
from torch.nn import functional as F


class SpatialPatchReadout(nn.Module):
    def __init__(self, embed_dim=1024, grid_height=16, grid_width=16):
        super().__init__()
        if min(embed_dim, grid_height, grid_width) < 1:
            raise ValueError("Positive embedding and grid dimensions required")
        self.embed_dim = embed_dim
        self.grid_height, self.grid_width = grid_height, grid_width
        self.content_score = nn.Parameter(torch.zeros(embed_dim))
        self.position_score = nn.Parameter(torch.zeros(2))
        y, x = torch.meshgrid(
            torch.linspace(-1., 1., grid_height),
            torch.linspace(-1., 1., grid_width), indexing="ij")
        self.register_buffer("patch_xy", torch.stack((x, y), -1).reshape(-1, 2))

    def forward(self, fused_tokens, *, return_attention=False):
        expected = (1 + self.grid_height * self.grid_width, self.embed_dim)
        if fused_tokens.ndim != 3 or tuple(fused_tokens.shape[1:]) != expected:
            raise ValueError(f"Expected [batch,{expected[0]},{expected[1]}] fused tokens")
        if not fused_tokens.is_floating_point():
            raise TypeError("Floating-point fused tokens required")
        patches = fused_tokens[:, 1:]
        scores = (F.layer_norm(patches, (self.embed_dim,)) @ self.content_score
                  + self.patch_xy @ self.position_score)
        attention = scores.softmax(dim=-1)
        # The residual expression is algebraically a weighted mean. At zero
        # scores its correction is exactly zero, retaining legacy mean arithmetic
        # while allowing gradients to reach BOTH new score parameter vectors.
        correction = ((attention - 1. / patches.shape[1]).unsqueeze(-1) * patches).sum(1)
        patch_readout = patches.mean(1) + correction
        pooled = .5 * (fused_tokens[:, 0] + patch_readout)
        return (pooled, attention) if return_attention else pooled

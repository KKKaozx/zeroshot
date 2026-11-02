"""Project Adapter plus the 1024->64 boundary, consuming frozen CLIP tokens."""
from __future__ import annotations

from pathlib import Path
import torch
from torch import nn


def adapter_class():
    # Postpone annotations for the existing source's Tensor | None on Python3.9.
    # No executable Adapter operation/parameter is changed.
    source = Path(__file__).with_name("frozen_adapter.py")
    namespace = {"__name__": "pinned_project_adapter"}
    exec(compile("from __future__ import annotations\n" + source.read_text(encoding="utf-8"),
                 str(source), "exec"), namespace)
    return namespace["CrossAttentionAdapter"]


class ProjectObservationEncoder(nn.Module):
    def __init__(self, text_tokens, text_mask, state_columns, adapter=None, width=1024):
        super().__init__()
        self.adapter = adapter if adapter is not None else adapter_class()(
            num_layers=8, embed_dim=1024, text_dim=768, attention_dim=512,
            num_heads=8, dropout=0.1, pooling="cls_patch_mean")
        self.projection = nn.Linear(width, 64)
        if len(state_columns) != 2 or len(set(state_columns)) != 2 or any(i < 0 or i >= 66 for i in state_columns):
            raise ValueError("Expected two distinct native agent_pos columns")
        self.state_columns = tuple(state_columns)
        self.rgb_columns = tuple(i for i in range(66) if i not in self.state_columns)
        self.register_buffer("text_tokens", text_tokens.detach().clone())
        self.register_buffer("text_mask", text_mask.detach().clone())
        self.cached_tokens = None

    def output_shape(self):
        return (66,)

    def set_tokens(self, tokens):
        if tokens.requires_grad:
            raise ValueError("CLIP tokens must be detached")
        self.cached_tokens = tokens

    def forward(self, obs):
        visual = self.cached_tokens
        if visual is None or visual.shape[0] != obs["agent_pos"].shape[0]:
            raise ValueError("Missing or misaligned precomputed CLIP tokens")
        if obs["agent_pos"].shape[-1] != 2:
            raise ValueError("Expected native normalized agent_pos with width 2")
        n = visual.shape[0]
        fused = self.adapter(visual, self.text_tokens.expand(n, -1, -1),
                             self.text_mask.expand(n, -1))
        rgb = self.projection(fused)
        result = rgb.new_empty((n, 66))
        result[:, self.rgb_columns] = rgb
        result[:, self.state_columns] = obs["agent_pos"]
        return result

"""Matched pre-Adapter bypasses into the unchanged64-dimensional RGB boundary."""
from __future__ import annotations

import torch
from torch import nn
from project_frontend import ProjectObservationEncoder

MODES = ("global_control", "spatial_candidate")


def moment_weights(device, dtype):
    axis = torch.linspace(-1, 1, 16, device=device, dtype=dtype)
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack((torch.ones_like(x), x, y, x * y)).reshape(4, 256) / 256


class SpatialBypassEncoder(ProjectObservationEncoder):
    def __init__(self, base, mode, center=None, scale=None):
        # Reuse original parameter objects/names so their Adam history survives.
        nn.Module.__init__(self)
        if mode not in MODES or not isinstance(base, ProjectObservationEncoder):
            raise ValueError("Expected a pinned project encoder and declared bypass mode")
        if hasattr(base, "bypass_head") or base.cached_tokens is not None:
            raise ValueError("Only an unbound original encoder can be augmented once")
        self.mode = mode
        self.adapter, self.projection = base.adapter, base.projection
        self.state_columns, self.rgb_columns = base.state_columns, base.rgb_columns
        self.register_buffer("text_tokens", base.text_tokens.detach().clone())
        self.register_buffer("text_mask", base.text_mask.detach().clone())
        self.cached_tokens = None
        self.width = int(base.projection.in_features)
        device, dtype = base.projection.weight.device, base.projection.weight.dtype
        self.register_buffer("spatial_weights", moment_weights(device, dtype))
        center = torch.zeros(4, self.width) if center is None else torch.as_tensor(center)
        scale = torch.ones(4, self.width) if scale is None else torch.as_tensor(scale)
        if (center.shape != (4, self.width) or scale.shape != center.shape or not torch.isfinite(center).all()
                or not torch.isfinite(scale).all() or not (scale > 0).all()):
            raise ValueError("Expected fixed finite train-only center and positive scale")
        self.register_buffer("bypass_center", center.detach().clone().to(device=device, dtype=dtype))
        self.register_buffer("bypass_scale", scale.detach().clone().to(device=device, dtype=dtype))
        # Creating the new zero head must not consume the original training RNG.
        with torch.random.fork_rng(devices=[]):
            self.bypass_head = nn.Linear(4 * self.width, 64, bias=False)
            nn.init.zeros_(self.bypass_head.weight)
        self.bypass_head.to(device=device, dtype=dtype)
        self.train(base.training)

    def bypass_features(self, visual):
        if (visual.ndim != 3 or tuple(visual.shape[1:]) != (257, self.width)
                or visual.requires_grad or not visual.is_floating_point()
                or visual.device != self.spatial_weights.device or visual.dtype != self.spatial_weights.dtype):
            raise ValueError("Expected detached, aligned frozen257-token CLIP features")
        moments = torch.einsum("kc,ncd->nkd", self.spatial_weights, visual[:, 1:])
        if self.mode == "global_control":
            # Same head size and raw-CLIP access, but no explicit patch layout.
            moments = moments[:, :1].expand(-1, 4, -1)
        # Preserve frame-to-frame coordinate amplitudes; no per-frame LayerNorm.
        normalized = (moments - self.bypass_center) / self.bypass_scale
        return normalized.reshape(len(visual), 4 * self.width) / (4 * self.width) ** .5

    def forward(self, obs):
        visual = self.cached_tokens
        if visual is None:
            raise ValueError("Missing frozen tokens")
        bypass = self.bypass_head(self.bypass_features(visual))
        original = super().forward(obs)
        result = original.clone()
        result[:, self.rgb_columns] = original[:, self.rgb_columns] + bypass
        return result


def attach(policy, averaged, optimizer, mode, statistics):
    if len(optimizer.param_groups) != 1:
        raise ValueError("Only original single-group AdamW is supported")
    old_ids = [id(p) for p in optimizer.param_groups[0]["params"]]
    if mode not in statistics:
        raise ValueError("Missing mode-bound train-only bypass statistics")
    center, scale = statistics[mode]["center"], statistics[mode]["scale"]
    policy.obs_encoder = SpatialBypassEncoder(policy.obs_encoder, mode, center, scale)
    averaged.obs_encoder = SpatialBypassEncoder(averaged.obs_encoder, mode, center, scale)
    averaged.requires_grad_(False).eval()
    added = list(policy.obs_encoder.bypass_head.parameters())
    optimizer.param_groups[0]["params"].extend(added)
    if [id(p) for p in optimizer.param_groups[0]["params"][:-len(added)]] != old_ids:
        raise ValueError("Original optimizer parameter objects changed")
    if any(p in optimizer.state for p in added):
        raise ValueError("New bypass Adam history must be empty")
    return {"old_parameter_objects_preserved": True, "old_optimizer_history_preserved": True,
            "new_parameters": sum(p.numel() for p in added), "new_adam_state_initially_empty": True,
            "output_feature_dimensions": 66, "visual_dimensions": 64,
            "new_ema_parameters_frozen": all(not p.requires_grad for p in averaged.obs_encoder.bypass_head.parameters())}

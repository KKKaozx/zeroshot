"""Isolated adapter around the archived full model; production files untouched.

The interface owns one base model, not a duplicate U-Net. Frozen CLIP tokens,
the two SAME-current Adapter calls, history port and current-state port match
the207558 runner. The gripper remains frozen and this is a pose-only experiment.
"""
import weakref
import torch
from torch import nn

from bridge_requested_prefix_diffusion_v1 import (
    RequestedPrefixDenoiser, training_objective, sample_requested,
)


class _CoreReference:
    # Not nn.Module: registering the same decoder under two parents would
    # duplicate state_dict keys and make checkpoint provenance ambiguous.
    def __init__(self, model):
        self.model = weakref.ref(model)

    def __call__(self, noisy, timesteps, context):
        model = self.model()
        if model is None:
            raise RuntimeError("Base model no longer exists")
        return model.diffusion_decoder(noisy, timesteps, context)


class PrefixModelInterface(nn.Module):
    def __init__(self, base):
        super().__init__()
        required = dict(chunk_size=16, action_dim=8, num_diffusion_steps=100,
            diffusion_prediction_type="sample", beta_schedule="squaredcos_cap_v2",
            clip_denoised=True, separate_gripper_head=True)
        for name, expected in required.items():
            if getattr(base, name, None) != expected:
                raise ValueError(f"Unsupported active contract: {name}")
        if float(base.max_normalized_position) != 3.:
            raise ValueError("Position limit must match the pinned active baseline")
        if not hasattr(base,"history_projection") or not hasattr(base,"state_projection"):
            raise ValueError("Construct/load the active history and state ports before wrapping")
        if base.state_projection.in_features != 9 or base.state_projection.out_features != 1024:
            raise ValueError("Expected the pinned9-to1024 current-state port")
        self.base = base
        device = base.state_projection.weight.device
        # Adding a zero-initialized request port must not perturb subsequent
        # dropout/noise RNG state in the original model.
        devices = [device.index or 0] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            self.prefix = RequestedPrefixDenoiser(_CoreReference(base),1024).to(
                device=device,dtype=base.state_projection.weight.dtype)

    def train(self, mode=True):
        super().train(mode)
        self.base.vision_encoder.eval();self.base.text_encoder.eval()
        return self

    def context_from_tokens(self, visual, language, attention_mask, current_state_features):
        if current_state_features.shape != (len(visual),9):
            raise ValueError("Expected only the9D encoded CURRENT observed state")
        if not torch.isfinite(current_state_features).all():
            raise ValueError("Current state must be finite")
        now = self.base.adapter(visual,language,attention_mask)
        repeated = self.base.adapter(visual,language,attention_mask)
        return (now+self.base.history_projection(repeated-now)
                +self.base.state_projection(current_state_features))

    def context_from_observation(self, pixels, input_ids, attention_mask, current_state_features):
        with torch.no_grad():
            visual = self.base.vision_encoder(pixel_values=pixels).last_hidden_state
            language = self.base.text_encoder(input_ids=input_ids,
                attention_mask=attention_mask).last_hidden_state
        return self.context_from_tokens(visual,language,attention_mask,current_state_features)

    def coefficients(self):
        return {k:getattr(self.base,k) for k in ("alphas","alpha_bars",
            "posterior_variance","posterior_mean_x0","posterior_mean_xt")}

    def loss(self, targets, availability, context, requested_horizon, timesteps=None, noise=None):
        # targets are7D poses ONLY. Availability is a training eligibility guard,
        # not an input to the Adapter/context or the inference method.
        if timesteps is None:
            timesteps=torch.randint(0,100,(len(targets),),device=targets.device)
        if noise is None:
            noise=torch.randn_like(targets)
        return training_objective(self.prefix,targets,availability,context,timesteps,
            noise,self.base.alpha_bars,requested_horizon)

    @torch.no_grad()
    def sample_pose(self, context, requested_horizon):
        return sample_requested(self.prefix,context,self.coefficients(),requested_horizon)

    def trainable_payload(self):
        """Same old base keys plus a separate explicit new request-port section."""
        trainable={n:p.detach().cpu().clone() for n,p in self.base.named_parameters() if p.requires_grad}
        return dict(format="bridge_requested_prefix_pose_v1",
            base_trainable_state_dict=trainable,
            request_projection_state_dict={k:v.detach().cpu().clone()
                for k,v in self.prefix.request_projection.state_dict().items()})

    def load_trainable_payload(self, payload):
        if payload.get("format") != "bridge_requested_prefix_pose_v1":
            raise ValueError("New format must not be guessed from an old checkpoint")
        expected={n for n,p in self.base.named_parameters() if p.requires_grad}
        values=payload["base_trainable_state_dict"]
        if set(values) != expected:
            raise ValueError("Base trainable keys differ; frozen/unknown keys cannot be silently loaded")
        result=self.base.load_state_dict(values,strict=False)
        if result.unexpected_keys:
            raise ValueError(result.unexpected_keys)
        self.prefix.request_projection.load_state_dict(payload["request_projection_state_dict"],strict=True)

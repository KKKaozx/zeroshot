"""Experimental CPU-validated DDPM contract; no production integration/launcher.

K is a requested prediction horizon (1 or16), NEVER remaining episode length.
The stochastic state lives in R^(K*7). The 16-slot U-Net receives an embedding
with deterministic zero auxiliary slots, at training and EVERY reverse step.
"""
import torch
from torch import nn
from torch.nn import functional as F

from bridge_partial_future_contract_v1 import masked_pose_mse


def request_mask(batch_size, requested_horizon, device):
    if type(requested_horizon) is not int or requested_horizon not in (1, 16):
        raise ValueError("Explicit prediction request must be1 or16, not episode remaining length")
    if batch_size < 1:
        raise ValueError("Empty batches are not supported")
    return (torch.arange(16, device=device)[None, :] < requested_horizon).expand(batch_size, -1)


def project_auxiliary(value, mask):
    if value.ndim != 3 or value.shape[1:] != (16, 7) or mask.shape != value.shape[:2]:
        raise ValueError("Expected B-by16-by7 latent and B-by16 mask")
    if mask.dtype != torch.bool:
        raise ValueError("Mask must be Boolean")
    # where, not value*mask: the latter propagates invalid NaN placeholders.
    return torch.where(mask[..., None], value, torch.zeros_like(value))


class RequestedPrefixDenoiser(nn.Module):
    """Same U-Net core plus explicit caller-request conditioning.

    Full16 request contributes zero context offset even after the new projection
    learns. This preserves the frozen core's full16 function, not the result of
    future joint optimization. Production1024-context adds16384 parameters.
    """
    def __init__(self, core, context_dim):
        super().__init__()
        self.core = core
        self.request_projection = nn.Linear(16, context_dim, bias=False)
        nn.init.zeros_(self.request_projection.weight)

    def forward(self, noisy, timesteps, context, requested_horizon):
        mask = request_mask(len(noisy), requested_horizon, noisy.device)
        projected = project_auxiliary(noisy, mask)
        offset = self.request_projection(mask.to(context.dtype)-1.)
        prediction = self.core(projected, timesteps, context+offset)
        return project_auxiliary(prediction, mask)


def training_objective(denoiser, targets, target_validity, context, timesteps,
                       noise, alpha_bars, requested_horizon):
    """Availability only checks eligibility; it NEVER enters model context.

    Every3394 candidate observation supports the sameK1 request. Only the old
    1836 full observations supportK16. Do not automatically setK=min(16,L).
    """
    mask = request_mask(len(targets), requested_horizon, targets.device)
    if targets.shape != noise.shape or targets.shape[1:] != (16,7):
        raise ValueError("Targets and noise must be B-by16-by7")
    if target_validity.shape != mask.shape or target_validity.dtype != torch.bool:
        raise ValueError("Expected Boolean supervision availability")
    if not torch.all(target_validity[mask]):
        raise ValueError("Requested horizon has missing real supervision; do not invent/shrink labels")
    if not torch.isfinite(targets[mask]).all() or not torch.isfinite(noise[mask]).all():
        raise ValueError("Real targets and active noise must be finite")
    clean = project_auxiliary(targets, mask)
    active_noise = project_auxiliary(noise, mask)
    alpha_bar = alpha_bars[timesteps].reshape(-1,1,1)
    noisy = alpha_bar.sqrt()*clean + (1.-alpha_bar).sqrt()*active_noise
    prediction = denoiser(noisy, timesteps, context, requested_horizon)
    # Preserve the archived reduction order for the old complete-window branch.
    # Equal-length per-window means are mathematically identical but float32
    # may round differently from one flattened mean.
    loss = F.mse_loss(prediction, targets) if requested_horizon == 16 else masked_pose_mse(prediction, targets, mask)
    return loss, noisy, prediction


def schedule_from_betas(betas):
    """Same x0-DDPM coefficients as the archived active model."""
    alphas = 1.-betas
    bars = torch.cumprod(alphas, dim=0)
    previous = F.pad(bars[:-1], (1,0), value=1.)
    return dict(alphas=alphas, alpha_bars=bars,
        posterior_variance=(betas*(1.-previous)/(1.-bars)).clamp(min=1e-20),
        posterior_mean_x0=betas*previous.sqrt()/(1.-bars),
        posterior_mean_xt=(1.-previous)*alphas.sqrt()/(1.-bars))


@torch.no_grad()
def sample_requested(denoiser, context, coefficients, requested_horizon):
    """Always100 denoising steps here; K1 is one future action, not one DDPM step.

    No episode length, terminal pose, target array or supervision mask argument.
    Caller choosesK1 for a next-action policy. K16 remains available for the old
    offline comparison. No claim that unrecorded16-step futures are supervised.
    """
    mask = request_mask(len(context), requested_horizon, context.device)
    bars = coefficients["alpha_bars"]
    if len(bars) != 100:
        raise ValueError("This prototype pins the active100-step schedule")
    actions = project_auxiliary(torch.randn(len(context),16,7,device=context.device), mask)
    for step in reversed(range(100)):
        times = torch.full((len(context),),step,device=context.device,dtype=torch.long)
        x0 = denoiser(actions,times,context,requested_horizon)
        alpha = coefficients["alphas"][step]
        bar = bars[step]
        # Keep active-source arithmetic, including the x0->eps->x0 roundtrip,
        # rather than claiming numerical replay of a simplified expression.
        epsilon = (actions-bar.sqrt()*x0)/(1.-bar).sqrt()
        clean = (actions-(1.-bar).sqrt()*epsilon)/bar.sqrt()
        clean = torch.cat([clean[...,:3].clamp(-3.,3.),clean[...,3:].clamp(-1.,1.)],dim=-1)
        clean = project_auxiliary(clean,mask)
        mean = coefficients["posterior_mean_x0"][step]*clean + coefficients["posterior_mean_xt"][step]*actions
        if step>0:
            actions = mean+coefficients["posterior_variance"][step].sqrt()*torch.randn_like(actions)
        else:
            actions = mean
        actions = project_auxiliary(actions,mask)
    # Return ONLY the requested physical pose slots. Auxiliary zero quaternions
    # are computational placeholders and must not become executable targets.
    poses=actions[:,:requested_horizon].clone()
    poses[...,:3]=poses[...,:3].clamp(-3.,3.)
    quaternion=poses[...,3:7]
    identity=torch.zeros_like(quaternion);identity[...,3]=1.
    poses[...,3:7]=torch.where(quaternion.norm(dim=-1,keepdim=True)>1e-6,
        F.normalize(quaternion,dim=-1),identity)
    return poses

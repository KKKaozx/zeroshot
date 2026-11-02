"""Restore all four229116 trainable modules; run pinned native loss/sampler.

Frozen feature caches replace encoder execution only. No gripper is trained or
scored here. Native sampling's final gripper column is a discarded placeholder.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent/'vendor'))
import torch
import torch.nn.functional as F
from models import RobotAdapterModel, diffusion_betas
from adapter import CrossAttentionAdapter
from diffusion_decoder import MultiscaleConditionalUnet1D

PREFIXES = ('adapter.', 'diffusion_decoder.', 'history_projection.', 'state_projection.')


class CachedPoseCore(RobotAdapterModel):
    def __init__(self, config):
        torch.nn.Module.__init__(self)
        cfg = config['model']; dim = config.get('frozen_context_dim', 1024)
        self.action_dim, self.chunk_size, self.num_diffusion_steps = 8, 16, 100
        self.decoder_type, self.diffusion_architecture = 'diffusion', 'multiscale'
        self.diffusion_prediction_type = cfg['diffusion_prediction_type']
        self.clip_denoised, self.beta_schedule = cfg['clip_denoised'], cfg['beta_schedule']
        self.max_normalized_position = config['action']['max_normalized_position']
        self.separate_gripper_head, self.trajectory_conditioned_gripper = True, True
        self.gripper_target_mode, self.condition_on_current_gripper = 'state', True
        self.adapter = CrossAttentionAdapter(num_layers=cfg['num_adapter_layers'], embed_dim=dim,
            text_dim=cfg.get('cached_text_dim', 768), attention_dim=cfg['attention_dim'],
            num_heads=cfg['num_attention_heads'], dropout=cfg['dropout'], pooling=cfg['adapter_pooling'])
        self.diffusion_decoder = MultiscaleConditionalUnet1D(action_dim=7, chunk_size=16, context_dim=dim,
            down_dims=tuple(cfg['diffusion_down_dims']), diffusion_step_embed_dim=cfg.get('diffusion_step_embed_dim', 128),
            kernel_size=cfg.get('diffusion_kernel_size', 5), num_steps=100)
        self.history_projection = torch.nn.Linear(dim, dim, bias=False)
        self.state_projection = torch.nn.Linear(9, dim, bias=False)
        betas = diffusion_betas(100, self.beta_schedule)
        alphas = 1-betas; bars = torch.cumprod(alphas, dim=0); prev = F.pad(bars[:-1], (1, 0), value=1.)
        for name, value in {'betas': betas, 'alphas': alphas, 'alpha_bars': bars,
                'posterior_variance': (betas*(1-prev)/(1-bars)).clamp(min=1e-20),
                'posterior_mean_x0': betas*prev.sqrt()/(1-bars),
                'posterior_mean_xt': (1-prev)*alphas.sqrt()/(1-bars)}.items():
            self.register_buffer(name, value)

    def train(self, mode=True):
        return torch.nn.Module.train(self, mode)

    def predict_gripper_logits(self, context, pose_actions, current_gripper=None):
        return context.new_zeros((len(context), 16))

    def restore(self, state):
        expected = {name for name, _ in self.named_parameters()}
        if set(state) != expected or any(not name.startswith(PREFIXES) for name in expected):
            raise ValueError('Source pose modules missing or unexpected; no partial restore')
        result = self.load_state_dict(state, strict=False)
        if result.unexpected_keys or expected.intersection(result.missing_keys):
            raise ValueError('Pose restore incomplete')
        if any(not torch.isfinite(v).all() for v in state.values()): raise ValueError('Nonfinite source')
        return self

    def context(self, visual, language, mask, state):
        now = self.adapter(visual, language, mask)
        repeated = self.adapter(visual, language, mask)
        return now+self.history_projection(repeated-now)+self.state_projection(state)

    def pose_state(self):
        return {name: value.detach().cpu().clone() for name, value in self.named_parameters()}

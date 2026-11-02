"""Restore four229116 source modules; v2 trains only decoder and state port.

Archived post-Adapter contexts bypass encoders and the frozen Adapter/history.
No gripper is trained/scored. Native sampling's last column is a placeholder.
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
TRAINABLE_PREFIXES = ('diffusion_decoder.', 'state_projection.')


def freeze_adapter_history(model):
    for name, value in model.named_parameters():
        value.requires_grad_(name.startswith(TRAINABLE_PREFIXES))
    model.frozen_adapter_context_only=True
    model.adapter.eval();model.history_projection.eval()
    return [value for value in model.parameters() if value.requires_grad]


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
        torch.nn.Module.train(self, mode)
        if getattr(self,'frozen_adapter_context_only',False):
            self.adapter.eval();self.history_projection.eval()
        return self

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

    def context_from_archive(self, archived_context, original_features, original_weight, features):
        # Subtract projections before adding the archived context: control at
        # initialization has an exactly zero delta, with no cancellation loss.
        if archived_context.requires_grad or original_weight.requires_grad:
            raise ValueError('Archived contexts and original projection must remain frozen')
        delta=self.state_projection(features)-F.linear(original_features,original_weight)
        return archived_context+delta

    def pose_state(self):
        return {name: value.detach().cpu().clone() for name, value in self.named_parameters()}

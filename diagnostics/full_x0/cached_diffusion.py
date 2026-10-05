"""Generated from audited project methods; real frozen contexts supplied externally."""
import math
from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusion_decoder import ConditionalDiffusionDecoder

def extract(values: torch.Tensor, timesteps: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    result = values.gather(0, timesteps)
    return result.reshape(timesteps.shape[0], *(1,) * (target.ndim - 1))

def diffusion_betas(steps: int, schedule: str) -> torch.Tensor:
    """100 步线性表未接近纯噪声；新训练采用 Diffusion Policy 的余弦表。"""
    if steps < 2:
        raise ValueError('diffusion steps must be at least 2')
    if schedule == 'linear':
        return torch.linspace(0.0001, 0.02, steps, dtype=torch.float32)
    if schedule != 'squaredcos_cap_v2':
        raise ValueError(f'Unknown beta schedule: {schedule}')
    t = torch.linspace(0, 1, steps + 1, dtype=torch.float64)
    cumulative = torch.cos((t + 0.008) / 1.008 * math.pi / 2).square()
    return (1 - cumulative[1:] / cumulative[:-1]).clamp(max=0.999).float()

def build_gripper_readout(hidden_dim: int, head_type: str) -> nn.Sequential:
    """只替换夹爪读出网络；输入投影、时间编码和开合定义保持不变。"""
    if head_type == 'legacy':
        return nn.Sequential(nn.LayerNorm(hidden_dim), nn.Mish(), nn.Linear(hidden_dim, 1))
    if head_type == 'mlp2':
        return nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
    raise ValueError(f'Unknown gripper head type: {head_type}')

class RobotAdapterModel(nn.Module):
    """冻结 CLIP，只训练跨模态 Adapter 和扩散动作头。"""

    def __init__(self, config: Dict[str, Any], cache_dir: Optional[str]=None) -> None:
        super().__init__()
        model_config = config['model']
        model_name = str(model_config['name'])
        self.action_dim = int(model_config.get('action_dim', 8))
        self.chunk_size = int(model_config.get('chunk_size', 16))
        self.decoder_type = str(model_config.get('decoder_type', 'diffusion'))
        if self.decoder_type not in {'diffusion', 'regression'}:
            raise ValueError(f'Unknown decoder type: {self.decoder_type}')
        self.num_diffusion_steps = int(model_config.get('num_diffusion_steps', 100))
        self.beta_schedule = str(model_config.get('beta_schedule', 'linear'))
        self.clip_denoised = bool(model_config.get('clip_denoised', False))
        self.diffusion_prediction_type = str(model_config.get('diffusion_prediction_type', 'epsilon'))
        if self.diffusion_prediction_type not in {'epsilon', 'sample'}:
            raise ValueError('diffusion_prediction_type must be epsilon or sample')
        self.separate_gripper_head = bool(model_config.get('separate_gripper_head', False))
        self.trajectory_conditioned_gripper = bool(model_config.get('trajectory_conditioned_gripper', False))
        self.gripper_head_type = str(model_config.get('gripper_head_type', 'legacy'))
        if self.gripper_head_type not in {'legacy', 'mlp2'}:
            raise ValueError('gripper_head_type must be legacy or mlp2')
        if self.gripper_head_type != 'legacy' and (not self.trajectory_conditioned_gripper):
            raise ValueError('mlp2 requires trajectory-conditioned gripper features')
        self.condition_on_current_gripper = bool(model_config.get('condition_on_current_gripper', False))
        self.gripper_target_mode = str(model_config.get('gripper_target_mode', 'state'))
        if self.gripper_target_mode not in {'state', 'transition'}:
            raise ValueError(f"gripper_target_mode must be 'state' or 'transition', got {self.gripper_target_mode!r}")
        if model_config.get('current_gripper_encoding') in {'bridge_measured_open_fraction_v1', 'bridge_measured_affine_unbounded_v1'} and self.gripper_target_mode != 'state':
            raise ValueError('Bridge连续测量不能作为transition先前命令，必须预测绝对state命令')
        self.gripper_transition_decode = str(model_config.get('gripper_transition_decode', 'cumulative'))
        if self.gripper_transition_decode not in {'cumulative', 'single_switch'}:
            raise ValueError(f"gripper_transition_decode must be 'cumulative' or 'single_switch', got {self.gripper_transition_decode!r}")
        self.gripper_switch_threshold = float(model_config.get('gripper_switch_threshold', 0.0))
        self.gripper_loss_weight = float(model_config.get('gripper_loss_weight', 0.25))
        self.regression_rotation_weight = float(model_config.get('regression_rotation_weight', 1.0))
        if not math.isfinite(self.regression_rotation_weight) or self.regression_rotation_weight <= 0:
            raise ValueError('Regression rotation weight must be finite and positive')
        self.gripper_change_weight = float(model_config.get('gripper_change_weight', 1.0))
        self.balanced_gripper_loss = bool(model_config.get('balanced_gripper_loss', True))
        action_config = config.get('action', {})
        self.max_normalized_position = action_config.get('max_normalized_position')
        self.vision_encoder = nn.Identity()
        self.text_encoder = nn.Identity()
        context_dim = int(config['frozen_context_dim'])
        self.fusion_type = str(model_config.get('fusion_type', 'cross_attention'))
        diffusion_action_dim = self.action_dim - 1 if self.separate_gripper_head else self.action_dim
        decoder_hidden_dim = int(model_config.get('decoder_hidden_dim', 256))
        if self.decoder_type == 'regression':
            if not self.separate_gripper_head:
                raise ValueError('Regression baseline requires a separate gripper head')
            self.regression_head = nn.Sequential(nn.LayerNorm(context_dim), nn.Linear(context_dim, decoder_hidden_dim), nn.Mish(), nn.Linear(decoder_hidden_dim, self.chunk_size * 7))
            with torch.no_grad():
                self.regression_head[-1].bias.zero_()
                self.regression_head[-1].bias.reshape(self.chunk_size, 7)[:, 6] = 1
        else:
            self.diffusion_decoder = ConditionalDiffusionDecoder(action_dim=diffusion_action_dim, chunk_size=self.chunk_size, context_dim=context_dim, hidden_dim=decoder_hidden_dim, num_steps=self.num_diffusion_steps)
        if self.separate_gripper_head:
            if self.trajectory_conditioned_gripper:
                self.gripper_context_projection = nn.Linear(context_dim, decoder_hidden_dim)
                self.gripper_pose_projection = nn.Linear(7, decoder_hidden_dim)
                if self.condition_on_current_gripper:
                    self.current_gripper_projection = nn.Linear(1, decoder_hidden_dim)
                self.gripper_time_embedding = nn.Embedding(self.chunk_size, decoder_hidden_dim)
                self.gripper_head = build_gripper_readout(decoder_hidden_dim, self.gripper_head_type)
            else:
                self.gripper_head = nn.Sequential(nn.Linear(context_dim, decoder_hidden_dim), nn.Mish(), nn.Linear(decoder_hidden_dim, self.chunk_size))
        betas = diffusion_betas(self.num_diffusion_steps, self.beta_schedule)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_previous = F.pad(alpha_bars[:-1], (1, 0), value=1.0)
        posterior_variance = betas * (1.0 - alpha_bars_previous) / (1.0 - alpha_bars)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)
        self.register_buffer('posterior_variance', posterior_variance.clamp(min=1e-20))
        self.register_buffer('posterior_mean_x0', betas * alpha_bars_previous.sqrt() / (1 - alpha_bars))
        self.register_buffer('posterior_mean_xt', (1 - alpha_bars_previous) * alphas.sqrt() / (1 - alpha_bars))

    def train(self, mode: bool=True) -> 'RobotAdapterModel':
        super().train(mode)
        self.vision_encoder.eval()
        self.text_encoder.eval()
        return self

    def diffusion_loss(self, actions: torch.Tensor, context: torch.Tensor, current_gripper: torch.Tensor | None=None) -> Tuple[torch.Tensor, ...]:
        if actions.shape[1:] != (self.chunk_size, self.action_dim):
            raise ValueError(f'Expected actions [B, {self.chunk_size}, {self.action_dim}], got {tuple(actions.shape)}')
        diffusion_targets = actions[..., :7] if self.separate_gripper_head else actions
        if self.decoder_type == 'regression':
            poses = self.regression_head(context).reshape(-1, self.chunk_size, 7)
            return (poses, diffusion_targets, self.predict_gripper_logits(context, actions[..., :7], current_gripper))
        batch_size = diffusion_targets.shape[0]
        timesteps = torch.randint(0, self.num_diffusion_steps, (batch_size,), device=actions.device)
        noise = torch.randn_like(diffusion_targets)
        alpha_bar = extract(self.alpha_bars, timesteps, diffusion_targets)
        noisy_actions = alpha_bar.sqrt() * diffusion_targets + (1.0 - alpha_bar).sqrt() * noise
        predicted_noise = self.diffusion_decoder(noisy_actions, timesteps, context)
        training_target = diffusion_targets if self.diffusion_prediction_type == 'sample' else noise
        if self.separate_gripper_head:
            return (predicted_noise, training_target, self.predict_gripper_logits(context, actions[..., :7], current_gripper))
        return (predicted_noise, training_target)

    def predict_gripper_logits(self, context: torch.Tensor, pose_actions: torch.Tensor, current_gripper: torch.Tensor | None=None) -> torch.Tensor:
        """Predict state logits or switch-event logits for an action chunk."""
        if not self.separate_gripper_head:
            raise RuntimeError('This checkpoint has no separate gripper head')
        if self.trajectory_conditioned_gripper:
            if pose_actions.shape[1:] != (self.chunk_size, 7):
                raise ValueError(f'Expected pose actions [B, {self.chunk_size}, 7], got {tuple(pose_actions.shape)}')
            time_indices = torch.arange(self.chunk_size, device=pose_actions.device)
            features = self.gripper_context_projection(context).unsqueeze(1) + self.gripper_pose_projection(pose_actions) + self.gripper_time_embedding(time_indices).unsqueeze(0)
            if self.condition_on_current_gripper:
                if current_gripper is None:
                    raise ValueError('Current gripper state is required by this checkpoint')
                current_gripper = current_gripper.reshape(context.shape[0], 1)
                features = features + self.current_gripper_projection(current_gripper).unsqueeze(1)
            return self.gripper_head(features).squeeze(-1)
        return self.gripper_head(context)

    @torch.no_grad()
    def sample(self, context: torch.Tensor, current_gripper: torch.Tensor | None=None) -> torch.Tensor:
        batch_size = context.shape[0]
        diffusion_action_dim = self.action_dim - 1 if self.separate_gripper_head else self.action_dim
        if self.decoder_type == 'regression':
            actions = self.regression_head(context).reshape(batch_size, self.chunk_size, 7)
        else:
            actions = torch.randn(batch_size, self.chunk_size, diffusion_action_dim, device=context.device)
        for step in reversed(range(self.num_diffusion_steps if self.decoder_type == 'diffusion' else 0)):
            timesteps = torch.full((batch_size,), step, device=context.device, dtype=torch.long)
            predicted_noise = self.diffusion_decoder(actions, timesteps, context)
            alpha = self.alphas[step]
            alpha_bar = self.alpha_bars[step]
            if self.diffusion_prediction_type == 'sample':
                predicted_noise = (actions - alpha_bar.sqrt() * predicted_noise) / (1 - alpha_bar).sqrt()
            mean = (actions - (1.0 - alpha) * predicted_noise / (1.0 - alpha_bar).sqrt()) / alpha.sqrt()
            if self.clip_denoised:
                clean = (actions - (1 - alpha_bar).sqrt() * predicted_noise) / alpha_bar.sqrt()
                position_limit = float(self.max_normalized_position or 3.0)
                clean = torch.cat([clean[..., :3].clamp(-position_limit, position_limit), clean[..., 3:].clamp(-1, 1)], dim=-1)
                mean = self.posterior_mean_x0[step] * clean + self.posterior_mean_xt[step] * actions
            if step > 0:
                actions = mean + self.posterior_variance[step].sqrt() * torch.randn_like(actions)
            else:
                actions = mean
        if self.max_normalized_position is not None:
            position_limit = float(self.max_normalized_position)
            actions[..., :3] = actions[..., :3].clamp(-position_limit, position_limit)
        quaternion = actions[..., 3:7]
        identity = torch.zeros_like(quaternion)
        identity[..., 3] = 1.0
        quaternion = torch.where(quaternion.norm(dim=-1, keepdim=True) > 1e-06, F.normalize(quaternion, dim=-1), identity)
        actions[..., 3:7] = quaternion
        if self.separate_gripper_head:
            gripper_logits = self.predict_gripper_logits(context, actions, current_gripper)
            if self.gripper_target_mode == 'transition':
                if current_gripper is None:
                    raise ValueError('Current gripper state is required for transition decoding')
                if self.gripper_transition_decode == 'single_switch':
                    (event_logits, event_indices) = gripper_logits.max(dim=1)
                    switch_events = torch.zeros_like(gripper_logits, dtype=torch.bool)
                    switch_events.scatter_(1, event_indices.unsqueeze(1), (event_logits >= self.gripper_switch_threshold).unsqueeze(1))
                else:
                    switch_events = gripper_logits >= self.gripper_switch_threshold
                initial_open = current_gripper.reshape(batch_size, 1) >= 0.0
                parity = torch.cumsum(switch_events.to(torch.int64), dim=1) % 2 == 1
                open_states = torch.logical_xor(initial_open, parity)
                gripper = torch.where(open_states, torch.ones_like(gripper_logits), -torch.ones_like(gripper_logits))
            else:
                gripper = torch.where(gripper_logits >= 0.0, torch.ones_like(gripper_logits), -torch.ones_like(gripper_logits))
            actions = torch.cat([actions, gripper.unsqueeze(-1)], dim=-1)
        else:
            actions[..., 7] = actions[..., 7].clamp(-1.0, 1.0)
        return actions

def policy_loss(model: RobotAdapterModel, model_output: Tuple[torch.Tensor, ...], actions: torch.Tensor, criterion: nn.Module, current_grippers: torch.Tensor | None=None, supervision_masks: torch.Tensor | None=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Combine pose diffusion loss with binary gripper classification loss."""
    (predicted_noise, true_noise) = model_output[:2]
    if model.decoder_type == 'regression':
        mask = supervision_masks if supervision_masks is not None else torch.ones_like(actions)
        xyz_mask = mask[..., :3]
        xyz_loss = ((predicted_noise[..., :3] - true_noise[..., :3]).square() * xyz_mask).sum() / xyz_mask.sum().clamp_min(1)
        predicted_q = nn.functional.normalize(predicted_noise[..., 3:7], dim=-1)
        target_q = nn.functional.normalize(true_noise[..., 3:7], dim=-1)
        rotation_mask = (mask[..., 3:7] > 0.5).all(dim=-1)
        rotation_error = 1 - (predicted_q * target_q).sum(dim=-1).abs().clamp(max=1)
        rotation_loss = rotation_error[rotation_mask].mean() if rotation_mask.any() else xyz_loss * 0
        pose_loss = xyz_loss + getattr(model, 'regression_rotation_weight', 1.0) * rotation_loss
        valid_gripper = mask[..., 7] > 0.5
    elif supervision_masks is None:
        pose_loss = criterion(predicted_noise, true_noise)
        valid_gripper = None
    else:
        pose_mask = supervision_masks[..., :7].to(predicted_noise.dtype)
        squared_error = (predicted_noise - true_noise).square()
        pose_loss = (squared_error * pose_mask).sum() / pose_mask.sum().clamp_min(1.0)
        valid_gripper = supervision_masks[..., 7] > 0.5
    gripper_loss = torch.zeros((), device=pose_loss.device)
    if len(model_output) == 3:
        gripper_logits = model_output[2]
        state_targets = actions[..., 7] > 0.0
        if model.gripper_target_mode == 'transition':
            if current_grippers is None:
                raise ValueError('Transition targets require current gripper state')
            current_open = current_grippers.reshape(-1, 1) >= 0.0
            previous_states = torch.cat([current_open, state_targets[:, :-1]], dim=1)
            gripper_targets = (state_targets != previous_states).float()
        else:
            gripper_targets = state_targets.float()
        per_step_bce = nn.functional.binary_cross_entropy_with_logits(gripper_logits, gripper_targets, reduction='none')
        step_weights = torch.ones_like(per_step_bce)
        if model.gripper_target_mode == 'transition':
            if model.gripper_change_weight > 1.0:
                step_weights = torch.where(gripper_targets.bool(), torch.full_like(step_weights, model.gripper_change_weight), step_weights)
        elif current_grippers is not None and model.gripper_change_weight > 1.0:
            current_open = (current_grippers >= 0.0).expand_as(gripper_targets)
            change_mask = gripper_targets.bool() != current_open
            step_weights = torch.where(change_mask, torch.full_like(step_weights, model.gripper_change_weight), step_weights)
        if model.balanced_gripper_loss:
            class_losses = []
            for class_value in (0.0, 1.0):
                class_mask = gripper_targets == class_value
                if valid_gripper is not None:
                    class_mask = class_mask & valid_gripper
                if class_mask.any():
                    selected_weights = step_weights[class_mask]
                    class_losses.append((per_step_bce[class_mask] * selected_weights).sum() / selected_weights.sum())
            if class_losses:
                gripper_loss = torch.stack(class_losses).mean()
        else:
            valid = valid_gripper if valid_gripper is not None else torch.ones_like(gripper_targets, dtype=torch.bool)
            if valid.any():
                selected_weights = step_weights[valid]
                gripper_loss = (per_step_bce[valid] * selected_weights).sum() / selected_weights.sum().clamp_min(1.0)
    total_loss = pose_loss + model.gripper_loss_weight * gripper_loss
    return (total_loss, pose_loss, gripper_loss)

def metrics(pred, target, teacher=None):
    (pred, target) = (np.asarray(pred), np.asarray(target))
    pn = pred[..., 3:7] / np.linalg.norm(pred[..., 3:7], axis=-1, keepdims=True)
    tn = target[..., 3:7] / np.linalg.norm(target[..., 3:7], axis=-1, keepdims=True)
    pos = np.linalg.norm(pred[..., :3] - target[..., :3], axis=-1) * 10
    angle = np.degrees(2 * np.arccos(np.clip(np.abs((pn * tn).sum(-1)), 0, 1)))
    (truth, guessed) = (target[..., 7] >= 0, pred[..., 7] >= 0)
    (opened, closed) = (truth, ~truth)
    op_recall = float(guessed[opened].mean()) if opened.any() else None
    cl_recall = float((~guessed[closed]).mean()) if closed.any() else None
    m = {'windows': len(pred), 'action_targets': truth.size, 'position_error_cm': float(pos.mean()), 'zero_motion_position_error_cm': float((np.linalg.norm(target[..., :3], axis=-1) * 10).mean()), 'rotation_error_deg': float(angle.mean()), 'identity_rotation_error_deg': float(np.degrees(2 * np.arccos(np.clip(np.abs(tn[..., 3]), 0, 1))).mean()), 'gripper_accuracy': float((truth == guessed).mean()), 'open_recall': op_recall, 'closed_recall': cl_recall, 'balanced_accuracy': (op_recall + cl_recall) / 2 if op_recall is not None and cl_recall is not None else None, 'true_open_rate': float(truth.mean()), 'predicted_open_rate': float(guessed.mean()), 'position_error_by_waypoint_cm': pos.mean(0).tolist(), 'rotation_error_by_waypoint_deg': angle.mean(0).tolist()}
    if teacher is not None:
        m['teacher_pose_gripper_accuracy'] = float(((np.asarray(teacher) >= 0) == truth).mean())
    for (name, pair) in (('open_to_closed', truth[:, :-1] & ~truth[:, 1:]), ('closed_to_open', ~truth[:, :-1] & truth[:, 1:])):
        correct = (guessed[:, :-1] == truth[:, :-1]) & (guessed[:, 1:] == truth[:, 1:])
        m[name + '_pairs'] = int(pair.sum())
        m[name + '_correct'] = int((pair & correct).sum())
    return m

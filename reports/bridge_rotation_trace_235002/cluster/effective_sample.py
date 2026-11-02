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
        RTRACE_HOOK.raw(step, actions, predicted_noise)
        alpha = self.alphas[step]
        alpha_bar = self.alpha_bars[step]
        if self.diffusion_prediction_type == 'sample':
            predicted_noise = (actions - alpha_bar.sqrt() * predicted_noise) / (1 - alpha_bar).sqrt()
        mean = (actions - (1.0 - alpha) * predicted_noise / (1.0 - alpha_bar).sqrt()) / alpha.sqrt()
        if self.clip_denoised:
            clean = (actions - (1 - alpha_bar).sqrt() * predicted_noise) / alpha_bar.sqrt()
            position_limit = float(self.max_normalized_position or 3.0)
            RTRACE_HOOK.reconstructed(step, clean)
            clean = torch.cat([clean[..., :3].clamp(-position_limit, position_limit), clean[..., 3:].clamp(-1, 1)], dim=-1)
            RTRACE_HOOK.clipped(step, clean)
            mean = self.posterior_mean_x0[step] * clean + self.posterior_mean_xt[step] * actions
        if step > 0:
            actions = mean + self.posterior_variance[step].sqrt() * torch.randn_like(actions)
        else:
            actions = mean
    if self.max_normalized_position is not None:
        position_limit = float(self.max_normalized_position)
        actions[..., :3] = actions[..., :3].clamp(-position_limit, position_limit)
    RTRACE_HOOK.final_raw(actions)
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

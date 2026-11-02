"""Opt-in paired x0 timestep weighting; architecture/action/sampler unchanged.

Min-SNR-x0 weight min(SNR,5), divided by its fixed100-step grid mean.
This fixed normalization preserves expected pose-loss scale against gripper BCE;
it is a disclosed variant of the published absolute weighting, not a claim that
this objective has repaired the policy. Never changes historical training code.
"""
import hashlib

import torch

MODES = ('uniform', 'min_snr5_grid_mean_one')


def timestep_weights(alpha_bars, timesteps, mode):
    if (mode not in MODES or alpha_bars.ndim != 1 or len(alpha_bars) != 100
            or not torch.isfinite(alpha_bars).all() or not ((alpha_bars > 0) & (alpha_bars < 1)).all()
            or timesteps.ndim != 1 or timesteps.dtype != torch.int64
            or not len(timesteps) or (timesteps < 0).any() or (timesteps >= len(alpha_bars)).any()):
        raise ValueError('Explicit supported x0 mode and finite100-step schedule/timesteps required')
    if mode == 'uniform':
        return torch.ones_like(timesteps, dtype=alpha_bars.dtype, device=alpha_bars.device)
    grid = (alpha_bars / (1 - alpha_bars)).clamp(max=5)
    grid = grid / grid.mean()  # Fixed full-grid constant, never minibatch/dev statistics.
    return grid[timesteps]


def masked_x0_mse(prediction, target, mask, weights):
    if (prediction.shape != target.shape or prediction.shape != mask.shape
            or prediction.ndim != 3 or prediction.shape[-1] != 7
            or weights.shape != (len(prediction),) or not torch.isfinite(weights).all()
            or (weights < 0).any() or not torch.isfinite(mask).all()
            or not ((mask == 0) | (mask == 1)).all()):
        raise ValueError('Aligned masked7D x0 predictions and per-example weights required')
    # Keep the original global valid-element denominator, including ragged tails.
    return ((prediction - target).square() * mask * weights[:, None, None]).sum() / mask.sum().clamp_min(1)


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def backward(b, indices):
    """Compatible with the frozen bcap_core guard/train loop; no optimizer step."""
    model, port = b['model'], b['port']
    if model.diffusion_prediction_type != 'sample':
        raise ValueError('This repair is only defined for an actually trained x0 model')
    model.train(); port.train(); b['optimizer'].zero_grad(set_to_none=True)
    x = b['batch'](indices)
    context = model.get_context_vector(x['images'], x['tokens']['input_ids'], x['tokens']['attention_mask']) + port(x['pose'])
    captured = []
    def capture(module, args):
        captured.append((args[0].detach(), args[1].detach().clone()))
    handle = model.diffusion_decoder.register_forward_pre_hook(capture)
    try:
        out = model.diffusion_loss(x['actions'], context, x['current'])
    finally:
        handle.remove()
    if len(captured) != 1 or not torch.equal(out[1], x['actions'][..., :7]):
        raise ValueError('Unexpected native x0 forward/noising contract')
    noisy, t = captured[0]
    original_total, original_pose, gripper = b['api'].policy_loss(model, out, x['actions'], torch.nn.MSELoss(),
        current_grippers=x['current'], supervision_masks=x['masks'])
    weights = timestep_weights(model.alpha_bars, t, b['repair_mode'])
    # Uniform arm takes the exact old computation graph, not a reimplementation.
    pose = original_pose if b['repair_mode'] == 'uniform' else masked_x0_mse(
        out[0], out[1], x['masks'][..., :7].to(out[0].dtype), weights)
    loss = original_total if b['repair_mode'] == 'uniform' else pose + model.gripper_loss_weight * gripper
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite paired repair loss')
    loss.backward()
    parameters = [p for p in model.parameters() if p.requires_grad] + list(port.parameters())
    if (port.weight.grad is None or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters)
            or any(p.requires_grad or p.grad is not None for enc in (model.vision_encoder, model.text_encoder) for p in enc.parameters())):
        raise RuntimeError('Invalid gradients or an unfrozen encoder')
    from pose_runtime import digest
    from step_training_state import capture_rng
    row = {'loss': float(loss.detach()), 'pose_loss': float(pose.detach()),
        'gripper_loss': float(gripper.detach()), 'unweighted_pose_loss': float(original_pose.detach()),
        'timesteps': t.cpu().tolist(), 'noisy_input_sha256': tensor_sha(noisy),
        'rng_after_backward_sha256': digest(capture_rng(include_cuda=b['cuda'])),
        'timestep_weight_mean': float(weights.mean()), 'timestep_weight_min': float(weights.min()),
        'timestep_weight_max': float(weights.max())}
    return row, parameters


def check_matched_random_stream(control, candidate):
    if len(control) != len(candidate):
        raise ValueError('Matched arms must have equal completed updates')
    keys = ('update', 'indices', 'lr', 'timesteps', 'noisy_input_sha256', 'rng_after_backward_sha256')
    for i, (a, b) in enumerate(zip(control, candidate)):
        if any(a[k] != b[k] for k in keys):
            raise ValueError('Matched stream differs at update ' + str(i + 1))
    return {'matched_updates': len(control), 'indices_lr_timesteps_noisy_inputs_rng_all_equal': True}

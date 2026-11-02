"""CPU data-contract prototype, deliberately NOT wired into production training.

No sampler is provided: validity masking alone does not establish an equivalent
DDPM training/inference contract. Targets outside the record are NaN sentinels.
"""
import numpy as np


HORIZON = 16


def make_row(episode, start, stride=4):
    n = int(episode["steps"])
    if start < 0 or start >= n - 1 or start % stride:
        raise ValueError("Start must be on the stride lattice with a real next observation")
    length = min(HORIZON, n - start - 1)
    return {"shard": episode["shard"], "record_index": episode["record_index"],
            "start_index": start, "task": episode["instruction"], "partition": "train",
            "valid_future_length": length,
            "future_state_indices": list(range(start + 1, start + length + 1)),
            "gripper_command_indices": list(range(start, start + length)),
            "validity": [True] * length + [False] * (HORIZON - length)}


def build_numeric_sample(states, resolved_valid_gripper, start, encode_pose, euler_to_quaternion):
    """Use one episode only; command[i] drives the reached state[i+1].

    Image/language access belongs to the existing observation loader at `start`;
    this function does not read/decode images or create CLIP features.
    """
    states = np.asarray(states, dtype=np.float32)
    gripper = np.asarray(resolved_valid_gripper, dtype=np.float32)
    if states.ndim != 2 or states.shape[1] != 7 or gripper.shape != (len(states) - 1,):
        raise ValueError("Expected one T-by7 episode and exactly T-1 valid commands")
    if not np.isfinite(states).all() or not np.isin(gripper, (0., 1.)).all():
        raise ValueError("States must be finite; unresolved gripper labels are not accepted")
    if not isinstance(start, (int, np.integer)) or not 0 <= start < len(states) - 1:
        raise ValueError("The final observation has no next-observation target")
    length = min(HORIZON, len(states) - start - 1)
    target = np.full((HORIZON, 8), np.nan, dtype=np.float32)
    q = euler_to_quaternion(states[start, 3:6])
    for k in range(length):
        target[k] = encode_pose(states[start, :3], q, states[start+k+1, :3],
                                euler_to_quaternion(states[start+k+1, 3:6]),
                                2. * float(gripper[start+k]) - 1.)
    valid = np.arange(HORIZON) < length
    return dict(target=target, valid=valid,
        current_pose=states[start, :6].copy(),
        current_gripper=np.array([2. * float(states[start, 6]) - 1.], dtype=np.float32),
        input_index=int(start), future_state_indices=np.arange(start+1, start+length+1),
        command_indices=np.arange(start, start+length))


def masked_pose_mse(predicted, targets, validity):
    """Mean per-window MSE over REAL pose elements, then equal window mean.

    Boolean indexing avoids propagating NaN sentinels through zero*NaN. Invalid
    output gradients are zero; this does not imply invalid INPUT independence.
    """
    import torch
    if predicted.ndim != 3 or predicted.shape[-2:] != (HORIZON, 7):
        raise ValueError("Expected B-by16-by7 pose outputs")
    if targets.shape != predicted.shape or validity.shape != predicted.shape[:2]:
        raise ValueError("Targets and temporal mask must match the prediction")
    if validity.dtype != torch.bool or not torch.all(validity.any(dim=1)):
        raise ValueError("Every window needs at least one real future target")
    # Contract is a contiguous real prefix, not arbitrary hidden holes.
    lengths = validity.sum(dim=1)
    expected = torch.arange(HORIZON, device=validity.device)[None, :] < lengths[:, None]
    if not torch.equal(validity, expected):
        raise ValueError("Temporal validity must be a prefix")
    losses = []
    for p, t, m in zip(predicted, targets, validity):
        if not torch.isfinite(p[m]).all() or not torch.isfinite(t[m]).all():
            raise ValueError("Real predictions and targets must be finite")
        losses.append((p[m] - t[m]).square().mean())
    return torch.stack(losses).mean()


def candidate_forward_noise(targets, validity, noise, alpha_bar):
    """Shape/numerical probe ONLY, not an accepted diffusion extension.

    NaNs are replaced BEFORE forward noise with a canonical zero translation,
    identity quaternion auxiliary value. Those values are not physical labels.
    The U-Net still receives noisy auxiliary slots; cross-slot effects remain.
    """
    import torch
    if targets.shape != noise.shape or targets.shape[-2:] != (HORIZON, 7):
        raise ValueError("Expected matching B-by16-by7 pose tensors")
    if validity.shape != targets.shape[:2] or validity.dtype != torch.bool:
        raise ValueError("Expected B-by16 Boolean validity")
    if not torch.isfinite(targets[validity]).all() or not torch.isfinite(noise).all():
        raise ValueError("Real targets and all noise must be finite")
    canonical = torch.zeros_like(targets)
    canonical[..., 6] = 1.
    clean = torch.where(validity[..., None], targets, canonical)
    return alpha_bar.sqrt() * clean + (1. - alpha_bar).sqrt() * noise

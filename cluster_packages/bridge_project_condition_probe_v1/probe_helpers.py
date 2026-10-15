"""Read-only observers for the frozen production x0 sampler; no sampling replacement."""
import hashlib
import numpy as np
import torch


STEPS = (99, 74, 49, 24, 0)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def rng_state(device):
    return (torch.get_rng_state().clone(),
            torch.cuda.get_rng_state(device).clone() if device.type == 'cuda' else None)


def restore_rng(state, device):
    torch.set_rng_state(state[0])
    if state[1] is not None:
        torch.cuda.set_rng_state(state[1], device)


def rng_equal(a, b):
    return torch.equal(a[0], b[0]) and (
        a[1] is None and b[1] is None or
        a[1] is not None and b[1] is not None and torch.equal(a[1], b[1]))


def replace_rows(context, current, selected, donors, all_context, all_current):
    c, g = context.clone(), current.clone()
    for slot, donor in zip(selected, donors):
        c[slot] = all_context[donor]
        g[slot] = all_current[donor]
    return c, g


@torch.inference_mode()
def sample_with_trace(model, context, current):
    """Observe actual decoder calls, including native float32 x0 conversion/clipping.

    Intermediate clean estimates are not samples executed in a robot. No truth is
    injected. Hook returns None, does not replace an output, and consumes no RNG.
    """
    assert model.diffusion_prediction_type == 'sample' and model.clip_denoised
    trace, calls = {}, []

    def observer(module, args, output):
        noisy, timestep, condition = args
        step = int(timestep[0])
        assert bool(torch.all(timestep == step))
        calls.append(step)
        if step not in STEPS:
            return
        assert torch.equal(condition, context)
        ab = model.alpha_bars[step]
        epsilon = (noisy - ab.sqrt() * output) / (1 - ab).sqrt()
        clean = (noisy - (1 - ab).sqrt() * epsilon) / ab.sqrt()
        limit = float(model.max_normalized_position or 3.0)
        clean = torch.cat([clean[..., :3].clamp(-limit, limit),
                           clean[..., 3:].clamp(-1, 1)], dim=-1)
        trace[step] = dict(raw=output.detach().cpu().numpy().copy(),
                           clean=clean.detach().cpu().numpy().copy())
        if step == 99:
            trace['initial_noise'] = noisy.detach().cpu().numpy().copy()

    handle = model.diffusion_decoder.register_forward_hook(observer)
    try:
        result = model.sample(context, current)
    finally:
        handle.remove()
    assert calls == list(reversed(range(100)))
    assert all(np.isfinite(trace[s][k]).all() for s in STEPS for k in ('raw', 'clean'))
    return result, trace


def pose_metrics(prediction, target):
    # 1 XYZ unit = 0.1 m. Quaternion sign invariance; match final production
    # fallback for a near-zero intermediate quaternion, and report its count.
    p, t = np.asarray(prediction, dtype=np.float64), np.asarray(target, dtype=np.float64)
    distance = np.linalg.norm(p[..., :3] - t[..., :3], axis=-1) * 10.0
    q, tq = p[..., 3:7].copy(), t[..., 3:7].copy()
    norms = np.linalg.norm(q, axis=-1, keepdims=True)
    bad = norms[..., 0] <= 1e-6
    q /= np.maximum(norms, 1e-8)
    q[bad] = [0, 0, 0, 1]
    tq /= np.maximum(np.linalg.norm(tq, axis=-1, keepdims=True), 1e-8)
    angle = 2 * np.arccos(np.clip(np.abs(np.sum(q * tq, axis=-1)), 0, 1)) * 180 / np.pi
    return dict(position_cm=float(distance.mean()), rotation_deg=float(angle.mean()),
                endpoint_position_cm=float(distance[..., -1].mean()),
                endpoint_rotation_deg=float(angle[..., -1].mean()),
                degenerate_quaternions=int(bad.sum()))

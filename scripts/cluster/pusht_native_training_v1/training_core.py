"""Shared deterministic plan, original loss adapter, and complete checkpoint state.

No alternative diffusion objective: compute_loss is the pinned author's method.
The mask-generator hook separates diffusion draws from frontend RNG consumption.
"""
import contextlib
import hashlib
import math
import os
from pathlib import Path
import random

import numpy as np
import torch


def seed_for(stream, update, micro=0):
    value = ("pusht-matched-v1:%s:%d:%d" % (stream, update, micro)).encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:4], "little")


class BatchPlan:
    def __init__(self, count=10726, size=64):
        self.count, self.size = count, size
        self.steps_per_epoch = math.ceil(count / size)
        self.epoch, self.order = None, None

    def batch(self, update):
        if update < 0:
            raise ValueError("Update index must be nonnegative")
        epoch, offset = divmod(update, self.steps_per_epoch)
        if self.epoch != epoch:
            self.order = np.random.RandomState(42 + epoch).permutation(self.count)
            self.epoch = epoch
        return self.order[offset * self.size:(offset + 1) * self.size].copy()

    def digest(self, updates):
        h = hashlib.sha256()
        for update in range(updates):
            row = self.batch(update).astype("<i8")
            h.update(np.asarray([len(row)], dtype="<i8").tobytes())
            h.update(row.tobytes())
        return h.hexdigest()


def tree_digest(value):
    h = hashlib.sha256()

    def visit(item):
        if torch.is_tensor(item):
            item = item.detach().cpu().contiguous()
            h.update(str((item.dtype, tuple(item.shape))).encode())
            h.update(item.numpy().tobytes())
        elif isinstance(item, np.ndarray):
            h.update(str((item.dtype, item.shape)).encode())
            h.update(item.tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=lambda x: (str(type(x)), str(x))):
                h.update(str(key).encode())
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            for part in item:
                visit(part)
        else:
            h.update(repr(item).encode())
    visit(value)
    return h.hexdigest()


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "cpu": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["cpu"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])


def capture(policy, averaged, optimizer, scheduler, ema, completed, protocol_sha, plan_sha):
    return {"format": "pusht-native-complete-v1", "protocol_sha256": protocol_sha,
            "plan_sha256": plan_sha, "completed_updates": completed,
            "raw": policy.state_dict(), "ema": averaged.state_dict(),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "ema_optimization_step": ema.optimization_step, "ema_decay": ema.decay,
            "rng": rng_state()}


def restore(payload, policy, averaged, optimizer, scheduler, ema, protocol_sha, plan_sha):
    if payload["protocol_sha256"] != protocol_sha or payload["plan_sha256"] != plan_sha:
        raise ValueError("Resume protocol or sample plan differs")
    if payload["format"] != "pusht-native-complete-v1":
        raise ValueError("Unsupported checkpoint")
    policy.load_state_dict(payload["raw"], strict=True)
    averaged.load_state_dict(payload["ema"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    # Torch 1.12 normally casts optimizer tensors when loading, enforce device.
    device = next(policy.parameters()).device
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value) and key != "step":
                state[key] = value.to(device)
    scheduler.load_state_dict(payload["scheduler"])
    ema.optimization_step = payload["ema_optimization_step"]
    ema.decay = payload["ema_decay"]
    restore_rng(payload["rng"])
    policy.train()
    averaged.eval()
    return int(payload["completed_updates"])


def atomic_checkpoint(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + ".writing")
    with temporary.open("wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


@contextlib.contextmanager
def planned_loss_rng(policy, update, micro):
    """Only reset diffusion RNG AFTER original mask generation, before randn.

    Original compute_loss still constructs both its noisy input and its epsilon
    target from the same local noise variable; no scheduler/noise replacement.
    """
    device = next(policy.parameters()).device
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    count = [0]

    def after_mask(module, inputs, output):
        count[0] += 1
        if bool(output.any()):
            raise ValueError("Only the pinned global-condition, action-unmasked objective is supported")
        torch.manual_seed(seed_for("diffusion", update, micro))

    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed_for("frontend", update, micro))
        handle = policy.mask_generator.register_forward_hook(after_mask)
        try:
            yield
            if count[0] != 1:
                raise ValueError("Expected exactly one original mask-generator call")
        finally:
            handle.remove()


def audited_original_loss(policy, batch, update, micro):
    """One real first-micro audit: check draw identity, loss, and a gradient.

    Hooks observe the original add_noise and U-Net calls without altering them.
    """
    import torch.nn.functional as F
    recorded = {}
    original = policy.noise_scheduler.add_noise

    def observe(trajectory, noise, timesteps):
        recorded["noise"], recorded["timesteps"] = noise, timesteps
        return original(trajectory, noise, timesteps)

    def output_hook(module, inputs, output):
        recorded["prediction"] = output

    policy.noise_scheduler.add_noise = observe
    handle = policy.model.register_forward_hook(output_hook)
    try:
        with planned_loss_rng(policy, update, micro):
            loss = policy.compute_loss(batch)
    finally:
        policy.noise_scheduler.add_noise = original
        handle.remove()
    noise = recorded["noise"]
    generator = torch.Generator(device=noise.device).manual_seed(seed_for("diffusion", update, micro))
    expected = torch.randn(noise.shape, device=noise.device, generator=generator)
    expected_t = torch.randint(0, 100, (len(noise),), device=noise.device, generator=generator).long()
    if not torch.equal(noise, expected) or not torch.equal(recorded["timesteps"], expected_t):
        raise ValueError("Original noise/timestep draws did not match the independent plan")
    # Author's 'b ... -> b (...)' only flattens; the subsequent mean is global.
    reference = F.mse_loss(recorded["prediction"], noise, reduction="none").flatten(1).mean()
    torch.testing.assert_close(loss, reference, rtol=0, atol=0)
    parameter = next(reversed(list(policy.model.parameters())))
    grad_original = torch.autograd.grad(loss, parameter, retain_graph=True)[0]
    grad_reference = torch.autograd.grad(reference, parameter, retain_graph=True)[0]
    torch.testing.assert_close(grad_original, grad_reference, rtol=0, atol=0)
    return loss, {"original_compute_loss_used": True, "common_noise_exact": True,
                  "common_timesteps_exact": True, "epsilon_objective_loss_exact": True,
                  "representative_backend_gradient_exact": True}


def update(policy, optimizer, scheduler, ema, batch, completed, micro_size=8, audit=False):
    policy.train()
    optimizer.zero_grad(set_to_none=True)
    device = next(policy.parameters()).device
    size = len(batch["action"])
    total, check = 0.0, None
    for micro, start in enumerate(range(0, size, micro_size)):
        stop = min(start + micro_size, size)
        part = {"obs": {k: v[start:stop].to(device) for k, v in batch["obs"].items()},
                "action": batch["action"][start:stop].to(device)}
        if audit and micro == 0:
            loss, check = audited_original_loss(policy, part, completed, micro)
        else:
            with planned_loss_rng(policy, completed, micro):
                loss = policy.compute_loss(part)
        if not bool(torch.isfinite(loss)):
            raise ValueError("Nonfinite training loss")
        weight = (stop - start) / size
        (loss * weight).backward()
        total += float(loss.detach()) * weight
    if audit:
        for name, module in (("frontend", policy.obs_encoder), ("backend", policy.model)):
            gradients = [p.grad for p in module.parameters() if p.requires_grad and p.grad is not None]
            if not gradients or not all(bool(torch.isfinite(g).all()) for g in gradients) or not any(bool(g.abs().max() > 0) for g in gradients):
                raise ValueError("Invalid first gradients: " + name)
        check["frontend_and_backend_gradients_valid"] = True
    optimizer.step()
    scheduler.step()
    ema.step(policy)
    return total, check

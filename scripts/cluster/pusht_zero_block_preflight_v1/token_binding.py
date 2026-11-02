"""Bind detached global-frame CLIP tokens to unchanged author microbatch calls."""
from contextlib import contextmanager
import json
from pathlib import Path

import numpy as np
import torch

from cache_tools import sha


def observation_ids(rows):
    rows = np.asarray(rows)
    if rows.ndim != 2 or rows.shape[1] != 4 or not np.issubdtype(rows.dtype, np.integer):
        raise ValueError("Expected native integer sampling rows")
    result = []
    for start, stop, sample_start, sample_stop in rows:
        if not (stop > start and 0 <= sample_start < sample_stop <= 16
                and stop - start == sample_stop - sample_start):
            raise ValueError("Invalid native padded window")
        frames = np.pad(np.arange(start, stop), (sample_start, 16 - sample_stop), mode="edge")
        result.append(frames[:2])
    return np.stack(result)


class FrozenCache:
    def __init__(self, directory, contract, data):
        directory = Path(directory)
        if sha(directory / "report.json") != contract["cache_report_sha256"]:
            raise ValueError("Frozen cache report changed")
        report = json.loads((directory / "report.json").read_text())
        if not (report["cache_complete"] and not report["errors"]
                and report["training_updates"] == 0 and not report["development_or_final_scored"]
                and report["worker"]["all_frozen"] and report["worker"]["weights_unchanged"]
                and report["reference128"]["passed"]):
            raise ValueError("Frozen cache has not passed its original contract")
        for name, info in report["files"].items():
            path = directory / name
            if path.stat().st_size != info["bytes"] or sha(path) != info["sha256"]:
                raise ValueError("Frozen cache file changed: " + name)
        if report["training_data"] != data.evidence:
            raise ValueError("Training data differs from cached inputs")
        self.visual = np.load(directory / "visual.npy", mmap_mode="r", allow_pickle=False)
        self.ids = np.load(directory / "frame_ids.npy", allow_pickle=False)
        expected_ids = np.flatnonzero(data.lookup >= 0)
        if not (self.visual.shape == (11356, 257, 1024) and self.visual.dtype == np.float32
                and self.ids.dtype == np.int64 and np.array_equal(self.ids, expected_ids)):
            raise ValueError("Global cache frame identity differs")
        with np.load(directory / "text.npz", allow_pickle=False) as text:
            self.text = torch.from_numpy(text["text"].copy())
            self.mask = torch.from_numpy(text["text_mask"].copy())
        self.rows = data.rows
        self.report = report

    def tokens(self, window_indices):
        ids = observation_ids(self.rows[window_indices]).reshape(-1)
        local = np.searchsorted(self.ids, ids)
        if (local >= len(self.ids)).any() or not np.array_equal(self.ids[local], ids):
            raise ValueError("Training observation escaped the frozen training cache")
        tokens = np.asarray(self.visual[local]).copy()
        if not np.isfinite(tokens).all():
            raise ValueError("Nonfinite frozen training features")
        return torch.from_numpy(tokens)


@contextmanager
def bind_tokens(encoder, tokens, logical_size, micro_size=8):
    """Author flattens [micro,2,...] in sample-major order exactly once per loss."""
    if tokens.requires_grad or tokens.ndim != 3 or len(tokens) != 2 * logical_size:
        raise ValueError("Misaligned or trainable frozen features")
    if encoder.cached_tokens is not None:
        raise ValueError("A previous feature binding was not cleared")
    cursor = [0]
    calls = [0]

    def before(module, inputs):
        count = len(inputs[0]["agent_pos"])
        remaining = len(tokens) - cursor[0]
        expected = 2 * min(micro_size, logical_size - cursor[0] // 2)
        if count != expected or remaining < count or count <= 0:
            raise ValueError("Author observation flattening/microbatch changed")
        device = next(module.parameters()).device
        module.set_tokens(tokens[cursor[0]:cursor[0] + count].to(device))
        cursor[0] += count
        calls[0] += 1

    def after(module, inputs, output):
        module.cached_tokens = None

    pre = encoder.register_forward_pre_hook(before)
    post = encoder.register_forward_hook(after)
    try:
        yield
        if cursor[0] != len(tokens) or calls[0] != (logical_size + micro_size - 1) // micro_size:
            raise ValueError("Not all observation features were consumed exactly once")
    finally:
        pre.remove()
        post.remove()
        encoder.cached_tokens = None


def project_update(core, policy, optimizer, scheduler, ema, batch, tokens, completed, audit=False):
    with bind_tokens(policy.obs_encoder, tokens, len(batch["action"])):
        loss, check = core.update(policy, optimizer, scheduler, ema, batch, completed, audit=audit)
    if audit:
        for name in ("adapter", "projection"):
            gradients = [p.grad for p in getattr(policy.obs_encoder, name).parameters() if p.requires_grad]
            if (not gradients or any(g is None or not bool(torch.isfinite(g).all()) for g in gradients)
                    or not any(bool(g.abs().max() > 0) for g in gradients)):
                raise ValueError("Invalid project gradients: " + name)
            check[name + "_gradients_valid"] = True
    return loss, check

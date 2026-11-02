"""Read-only source restore and train-frame-only feature statistics for new bypass."""
import copy
import gc
import json
from pathlib import Path
import sys
import numpy as np
import torch

import training_core as core
from native_data import TrainingData, sha
from token_binding import FrozenCache
from resource_probe import state_hash
from spatial_bypass import moment_weights, MODES


def prepare(root, output, protocol):
    bundle = Path(__file__).resolve().parent
    for name, digest in protocol["reused_sources_sha256"].items():
        if sha(bundle / name) != digest:
            raise ValueError("Pinned source changed: " + name)
    base = json.loads((bundle / "base_protocol.json").read_text())
    original = json.loads((bundle / "project_contract.json").read_text())
    source = root / "paper_reproduction/official_pusht_v1/source"
    for name, digest in base["author_file_sha256"].items():
        if sha(source / name) != digest:
            raise ValueError("Author file changed: " + name)
    sys.path.insert(0, str(source))
    data = TrainingData(root, base, output)
    cache = FrozenCache(root / original["cache_directory"], original, data)
    if data.evidence["raw_training_images_sha256"] != protocol["raw_training_images_sha256"]:
        raise ValueError("Original train90 pixels changed")
    plan = core.BatchPlan()
    if plan.digest(84168) != original["sample_order_sha256"]:
        raise ValueError("Original batch order changed")
    path = root / protocol["source_checkpoint"]
    sidecar = json.loads(path.with_suffix(".json").read_text())
    if (sidecar["sha256"] != protocol["source_checkpoint_sha256"] or sha(path) != sidecar["sha256"]
            or sidecar["completed_updates"] != 84168
            or sidecar["project_contract_sha256"] != sha(bundle / "project_contract.json")):
        raise ValueError("Wrong project endpoint")
    payload = torch.load(path, map_location="cpu")
    if (payload.get("project_contract_sha256") != sidecar["project_contract_sha256"]
            or payload["completed_updates"] != 84168 or payload["protocol_sha256"] != sha(bundle / "base_protocol.json")
            or payload["plan_sha256"] != original["sample_plan_sha256"]):
        raise ValueError("Wrong endpoint payload contract")
    return base, original, data, cache, plan, payload


def fit_statistics(cache, output, device=None):
    """No states/actions/labels or nontrain images are available to this function."""
    device = torch.device("cuda") if device is None else device
    weights = moment_weights(device, torch.float32)
    total = np.zeros((4, 1024), dtype=np.float64)
    squared = np.zeros_like(total)
    with torch.no_grad():
        for start in range(0, len(cache.visual), 32):
            tokens = torch.from_numpy(np.asarray(cache.visual[start:start + 32]).copy()).to(device)
            moments = torch.einsum("kc,ncd->nkd", weights, tokens[:, 1:]).cpu().numpy().astype(np.float64)
            total += moments.sum(0)
            squared += (moments * moments).sum(0)
            if start % 2048 == 0:
                print("BYPASS_TRAIN_STATISTICS", min(start + 32, len(cache.visual)), "/", len(cache.visual), flush=True)
    center = total / len(cache.visual)
    scale = np.sqrt(np.maximum(squared / len(cache.visual) - center * center, 0))
    scale = np.where(scale < 1e-6, 1., scale)
    stats = {"spatial_candidate": {"center": center.astype(np.float32), "scale": scale.astype(np.float32)},
             "global_control": {"center": np.repeat(center[:1], 4, axis=0).astype(np.float32),
                                "scale": np.repeat(scale[:1], 4, axis=0).astype(np.float32)}}
    path = output / "bypass_statistics.npz"
    np.savez(path, frame_ids=cache.ids, **{mode + "_" + field: value for mode in MODES
                                        for field, value in stats[mode].items()})
    return stats, {"frames": len(cache.visual), "scope": "Original90 train episodes only",
                   "labels_used": False, "nontrain_frames_used": False,
                   "sha256": sha(path), "per_frame_normalization": False,
                   "formula": "Train-only per-channel center/std, std<1e-6 replaced by1; divide flattened features by sqrt(4096)"}


def restore_original(cache, data, payload, base, original):
    import hydra
    from omegaconf import OmegaConf
    from diffusion_policy.model.common.normalizer import LinearNormalizer
    from diffusion_policy.common.normalize_util import get_image_range_normalizer
    from diffusion_policy.model.common.lr_scheduler import get_scheduler
    from project_frontend import ProjectObservationEncoder
    bundle = Path(__file__).resolve().parent
    cfg = OmegaConf.load(bundle / "checkpoint_config.yaml")
    torch.manual_seed(42)
    policy = hydra.utils.instantiate(cfg.policy)
    torch.manual_seed(43)
    policy.obs_encoder = ProjectObservationEncoder(cache.text, cache.mask, (64, 65))
    normalizer = LinearNormalizer()
    normalizer.fit({"action": data.train_actions, "agent_pos": data.train_positions}, last_n_dims=1, mode="limits")
    normalizer["image"] = get_image_range_normalizer()
    policy.set_normalizer(normalizer)
    policy.cuda().train()
    averaged = copy.deepcopy(policy).eval()
    ema = hydra.utils.instantiate(cfg.ema, model=averaged)
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=policy.parameters())
    scheduler = get_scheduler("cosine", optimizer=optimizer, num_warmup_steps=500, num_training_steps=base["lr_horizon_updates"])
    completed = core.restore(payload, policy, averaged, optimizer, scheduler, ema,
                             payload["protocol_sha256"], original["sample_plan_sha256"])
    restored = core.capture(policy, averaged, optimizer, scheduler, ema, completed,
                            payload["protocol_sha256"], original["sample_plan_sha256"])
    restored["project_contract_sha256"] = payload["project_contract_sha256"]
    if core.tree_digest(restored) != core.tree_digest(payload) or completed != 84168:
        raise ValueError("Original raw/EMA/Adam/LR/EMA counter/all RNG restore differs")
    if state_hash(policy.normalizer) != base["normalizer_sha256"] or state_hash(averaged.normalizer) != base["normalizer_sha256"]:
        raise ValueError("Original train-only normalizer changed")
    del restored
    gc.collect()
    return policy, averaged, optimizer, scheduler, ema


def encoder_context(policy, batch, tokens):
    encoder = policy.obs_encoder
    before = encoder.training
    encoder.eval()
    obs = policy.normalizer.normalize({k: v.cuda() for k, v in batch["obs"].items()})
    obs = {k: v[:, :2].reshape((-1,) + tuple(v.shape[2:])) for k, v in obs.items()}
    encoder.set_tokens(tokens.cuda())
    try:
        with torch.no_grad():
            return encoder(obs).detach().cpu()
    finally:
        encoder.cached_tokens = None
        encoder.train(before)


def sample_actions(averaged, batch, tokens):
    averaged.eval()
    averaged.obs_encoder.set_tokens(tokens.cuda())
    try:
        torch.manual_seed(20261021)
        with torch.no_grad():
            value = averaged.predict_action({k: v.cuda() for k, v in batch["obs"].items()})
        if any(not torch.isfinite(v).all() for v in value.values()):
            raise ValueError("Nonfinite smoke actions")
        return {k: v.cpu().clone() for k, v in value.items()}
    finally:
        averaged.obs_encoder.cached_tokens = None

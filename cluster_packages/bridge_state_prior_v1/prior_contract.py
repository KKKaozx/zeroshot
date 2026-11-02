"""Fixed observed-state position prior; residual actions retain original units."""
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rotation_features(observed):
    r, p, y = np.asarray(observed, dtype=float)[:, 3:6].T
    cr, cp, cy = np.cos(r), np.cos(p), np.cos(y)
    sr, sp, sy = np.sin(r), np.sin(p), np.sin(y)
    return np.stack([cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr,
                     sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr,
                     -sp, cp*sr, cp*cr], axis=-1)


def predict_prior(observed, tasks, specification):
    """Output project XYZ units (one unit=0.1m); never accepts future targets."""
    observed = np.asarray(observed, dtype=float)
    assert observed.ndim == 2 and observed.shape[1] == 6
    rotations = rotation_features(observed)
    result = []
    for i, task in enumerate(tasks):
        cfg = specification['tasks'][task]
        x = np.concatenate([(observed[i, :3]-cfg['xyz_mean'])/cfg['xyz_scale'],
                            (rotations[i]-cfg['rotation_mean'])/cfg['rotation_scale']])/np.sqrt(2)
        result.append((np.asarray(cfg['intercept'])+x@np.asarray(cfg['weights'])).reshape(16, 3))
    result = np.asarray(result, dtype='<f4')
    assert np.isfinite(result).all()
    return result


def residual_targets(actions, prior):
    result = actions.clone()
    assert tuple(prior.shape) == tuple(actions.shape[:2])+(3,)
    result[..., :3] -= prior
    return result


def restore_predictions(residual, prior, position_limit=3.):
    result = residual.copy()
    result[..., :3] += np.asarray(prior)[None]
    before = result[..., :3].copy()
    result[..., :3] = np.clip(before, -position_limit, position_limit)
    return result, int(np.count_nonzero(before != result[..., :3]))


def evaluate_restored(evaluate, score, model, context, current, truth, seeds, prior, batch):
    # Existing sampler and random draws remain unchanged. Score only reconstructed actions.
    residual, _ = evaluate(model, context, current, truth, seeds, batch=batch)
    result, clipped = restore_predictions(residual, prior, model.max_normalized_position or 3.)
    evaluate_restored.last_residual = residual
    evaluate_restored.last_clipped = clipped
    return result, score(result, truth)


def encode_or_reuse(model, items, texts, cache_path, identity, digest):
    """One cache per input/model/software identity, shared by preflight and formal job."""
    import torch
    cache_path = Path(cache_path)
    key = dict(identity=identity, clip=[digest(model.vision_encoder), digest(model.text_encoder)],
               torch_version=str(torch.__version__), tokenizer_inputs={k:sha_tensor(v) for k, v in texts.items()})
    if cache_path.exists():
        payload = torch.load(cache_path, map_location='cpu', weights_only=True)
        assert payload['key'] == key, 'Token cache identity changed; do not silently reuse/overwrite.'
        assert payload['hashes'] == {k:sha_tensor(payload[k]) for k in ['visual', 'language']}
        return payload['visual'].cuda(), payload['language'].cuda(), True
    with torch.no_grad():
        language = torch.cat([model.text_encoder(**{k:v[i:i+16].cuda() for k,v in texts.items()}).last_hidden_state
                              for i in range(0, len(items), 16)]).detach()
        visual = torch.cat([model.vision_encoder(pixel_values=torch.stack([v[1] for v in items[i:i+2]]).cuda()).last_hidden_state
                            for i in range(0, len(items), 2)]).detach()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(key=key, visual=visual.cpu(), language=language.cpu())
    payload['hashes'] = {k:sha_tensor(payload[k]) for k in ['visual', 'language']}
    temporary = cache_path.with_suffix('.partial')
    torch.save(payload, temporary)
    assert temporary.stat().st_size < 256*1024**2
    temporary.replace(cache_path)
    return visual, language, False


def sha_tensor(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def verify_loss_identity(model, actions, context, current, seed):
    """One raw forward replay, not a retrained control arm."""
    import torch
    with torch.no_grad():
        seed(420001)
        a = model.diffusion_loss(actions, context.detach(), current)
        seed(420001)
        b = model.diffusion_loss(residual_targets(actions, torch.zeros_like(actions[..., :3])), context.detach(), current)
    maximum = max(float((x-y).abs().max()) for x,y in zip(a,b))
    assert maximum == 0
    seed(420001)
    return maximum

"""Frozen EMA loading and read-only prediction; no optimizers or training path."""
import gc
import json
from pathlib import Path
import numpy as np
import torch
from resource_probe import sha, state_hash


def load_endpoint(root, bundle, spec, base, project):
    import hydra
    from omegaconf import OmegaConf
    path = root / spec['path']
    sidecar = json.loads(path.with_suffix('.json').read_text())
    if sidecar['sha256'] != spec['sha256'] or sha(path) != spec['sha256'] or sidecar['completed_updates'] != 84168:
        raise ValueError('Wrong checkpoint identity')
    payload = torch.load(path, map_location='cpu')
    if not (payload['format'] == 'pusht-native-complete-v1' and payload['completed_updates'] == 84168
            and payload['protocol_sha256'] == spec['protocol_sha256'] and payload['plan_sha256'] == spec['sample_plan_sha256']):
        raise ValueError('Endpoint metadata differs')
    if project and payload.get('project_contract_sha256') != spec['project_contract_sha256']:
        raise ValueError('Wrong project family')
    state = payload['ema']
    del payload
    gc.collect()
    torch.manual_seed(42)
    policy = hydra.utils.instantiate(OmegaConf.load(bundle / 'checkpoint_config.yaml').policy)
    if project:
        from project_frontend import ProjectObservationEncoder
        policy.obs_encoder = ProjectObservationEncoder(state['obs_encoder.text_tokens'], state['obs_encoder.text_mask'], (64,65))
    policy.load_state_dict(state, strict=True)
    del state
    gc.collect()
    # Loading normalizer ParameterDict reconstructs CPU tensors: move AFTER load.
    policy.requires_grad_(False).eval().cuda()
    if any(v.device.type != 'cuda' for v in list(policy.parameters()) + list(policy.buffers())):
        raise ValueError('Restored tensor on wrong device')
    if (policy.horizon, policy.n_obs_steps, policy.n_action_steps, policy.num_inference_steps) != (16,2,8,100):
        raise ValueError('Policy shape/sampling contract changed')
    if not policy.obs_as_global_cond or policy.noise_scheduler.config.prediction_type != 'epsilon':
        raise ValueError('Only pinned global-condition epsilon policy supported')
    if state_hash(policy.normalizer) != base['normalizer_sha256']:
        raise ValueError('Train-only normalizer differs')
    return policy


def tensor_sha(x):
    import hashlib
    return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def frozen_batch(policy, obs, targets, tokens, seed, batch_index):
    if policy.training or any(p.requires_grad or p.grad is not None for p in policy.parameters()):
        raise ValueError('Expected frozen eval policy')
    encoder = policy.obs_encoder
    if tokens is not None:
        encoder.set_tokens(tokens)
    captured = {}
    calls = [0]
    def before_model(module, args):
        calls[0] += 1
        if calls[0] == 1:
            captured['initial_noise_sha256'] = tensor_sha(args[0])
    hook = policy.model.register_forward_pre_hook(before_model)
    try:
        torch.manual_seed(seed + batch_index * 10000)
        with torch.no_grad():
            result = policy.predict_action(obs)
        captured['sampling_rng_after_sha256'] = tensor_sha(torch.get_rng_state())
        if policy.device.type == 'cuda':
            captured['sampling_cuda_rng_after_sha256'] = tensor_sha(torch.cuda.get_rng_state(policy.device))
        if calls[0] != 100 or result['action_pred'].shape[1:] != (16,2):
            raise ValueError('Not original DDPM100 full-horizon output')
        if not torch.equal(result['action'], result['action_pred'][:,1:9]):
            raise ValueError('Executed action slice differs')
    except BaseException:
        if tokens is not None:
            encoder.cached_tokens = None
        raise
    finally:
        hook.remove()
    try:
        with torch.no_grad():
            normalized = policy.normalizer.normalize(obs)
            flat = {k: v[:,:2].reshape(-1,*v.shape[2:]) for k,v in normalized.items()}
            context = encoder(flat).reshape(len(targets),-1)
            clean = policy.normalizer['action'].normalize(targets)
            torch.manual_seed(seed + 2000000 + batch_index * 10000)
            noise = torch.randn_like(clean)
            captured['teacher_noise_sha256'] = tensor_sha(noise)
            mse = []
            for t in (0,49,99):
                times = torch.full((len(clean),), t, device=clean.device, dtype=torch.long)
                noisy = policy.noise_scheduler.add_noise(clean, noise, times)
                output = policy.model(noisy, times, local_cond=None, global_cond=context)
                mse.append((output-noise).square().flatten(1).mean(1).cpu().numpy())
        return result['action_pred'].cpu().numpy(), np.stack(mse), captured
    finally:
        if tokens is not None:
            encoder.cached_tokens = None

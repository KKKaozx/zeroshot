"""Restore explicitly supplied training constants, retaining exact frozen identity.

Never replaces encoder/learned parameters. A different frozen tensor, order,
configuration, code identity or final digest is rejected with no retained edit.
"""
import copy
import hashlib
import json
from pathlib import Path

import torch

SCHEDULE_KEYS = {'betas','alphas','alpha_bars','posterior_variance','posterior_mean_x0','posterior_mean_xt'}


def metadata(value):
    cpu = value.detach().cpu().contiguous()
    raw = cpu.reshape(-1).view(torch.uint8).numpy().tobytes()
    return {'dtype':str(cpu.dtype),'shape':list(cpu.shape),'sha256':hashlib.sha256(raw).hexdigest()}


def normalized_config(config):
    value = copy.deepcopy(config)
    # Asset paths may be symlink aliases or point to the local machine. Tensor
    # identities, pinned DINO assets and the final frozen digest remain exact.
    value['model'].pop('name', None)
    value['model'].pop('dino_assets', None)
    return value


def restore_training_constants(model, learned_keys, contract, profile_path):
    from step_training_runner import frozen_digest
    source = Path(profile_path).resolve()
    raw = source.read_bytes()
    profile = json.loads(raw)
    if profile.get('legacy_frozen_sha256') != contract['frozen_state_sha256']:
        raise ValueError('Frozen profile is not bound to this training contract')
    if (profile.get('tokenizer_before_model') != contract['tokenizer']
            or profile.get('tokenizer_after_model') != contract['tokenizer']
            or profile.get('expected_tokenizer') != contract['tokenizer']):
        raise ValueError('Frozen profile tokenizer identity differs')
    if normalized_config(profile['config']) != normalized_config(contract['config']):
        raise ValueError('Frozen profile model/action configuration differs')
    for filename, digest in profile['core_source_sha256'].items():
        if contract['source_code_sha256'].get(filename) != digest:
            raise ValueError('Frozen profile implementation differs: ' + filename)
    expected = profile['tensors']
    actual = {key:value for key,value in model.state_dict().items() if key not in learned_keys}
    if set(actual) != set(expected) or list(actual) != profile['key_order']:
        raise ValueError('Frozen tensor keys or ordering differ')
    replacements = {}
    errors = {}
    for name, value in actual.items():
        current = metadata(value)
        if current == expected[name]:
            continue
        if (name not in SCHEDULE_KEYS or name not in model._buffers
                or name in dict(model.named_parameters()) or name in learned_keys):
            raise ValueError('Frozen parameter differs; constants cannot repair it: ' + name)
        if current['dtype'] != expected[name]['dtype'] or current['shape'] != expected[name]['shape']:
            raise ValueError('Derived constant shape/dtype differs: ' + name)
        saved = torch.tensor(profile['schedule_values'][name], dtype=value.dtype)
        if metadata(saved) != expected[name] or not bool(torch.isfinite(saved).all()):
            raise ValueError('Saved constant bytes do not match the profile: ' + name)
        replacements[name] = saved.to(device=value.device)
        difference = (saved.double() - value.detach().cpu().double()).abs()
        errors[name] = {'max_abs_before_restore':float(difference.max()),
                       'unequal_values_before_restore':int(torch.count_nonzero(difference))}
    # All validation precedes any copy. Only known root buffers can be copied.
    before = {key:model.get_buffer(key).detach().clone() for key in replacements}
    before_digest = frozen_digest(model, learned_keys)
    try:
        with torch.no_grad():
            for key, value in replacements.items():
                model.get_buffer(key).copy_(value)
        restored_digest = frozen_digest(model, learned_keys)
        if restored_digest != contract['frozen_state_sha256']:
            raise ValueError('Exact full frozen digest still differs after saved constants; rejecting')
    except Exception:
        with torch.no_grad():
            for key, value in before.items():
                model.get_buffer(key).copy_(value)
        raise
    return {'profile_path':str(source),'profile_file_sha256':hashlib.sha256(raw).hexdigest(),
        'frozen_sha256_before':before_digest,'frozen_sha256_after':restored_digest,
        'restored_buffer_names':sorted(replacements),'differences':errors,
        'encoder_parameters_written':False,'learned_parameters_written':False,
        'exact_original_training_digest_verified':True,'identity_tolerance_used':False}

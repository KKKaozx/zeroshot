"""Passive hooks on frozen endpoint modules; no policy optimizer or decoder call."""
import numpy as np
import torch


def pooled(tokens):
    return .5 * (tokens[:, 0] + tokens[:, 1:].mean(1))


def spatial_moments(tokens):
    if tokens.ndim != 3 or tokens.shape[1] != 257:
        raise ValueError("Expected CLS plus a 16x16 patch grid")
    axis = torch.linspace(-1, 1, 16, device=tokens.device, dtype=tokens.dtype)
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    weights = torch.stack((torch.ones_like(x), x, y, x * y)).reshape(4, 256) / 256
    return torch.einsum("kc,ncd->nkd", weights, tokens[:, 1:]).reshape(len(tokens), -1)


def project_features(encoder, visual):
    if encoder.training or any(p.requires_grad for p in encoder.parameters()):
        raise ValueError("Diagnostic requires frozen eval-mode encoder")
    found, hooks = {}, []

    def record(name, value):
        found[name] = value.detach().cpu().numpy().copy()

    def layer_hook(index):
        def hook(module, args, output):
            if index in (1, 4):
                record("adapter%d_pool" % index, pooled(output))
            if index == 8:
                record("adapter8_pre_norm", pooled(output))
                record("adapter8_spatial", spatial_moments(output))
        return hook

    record("clip_pool", pooled(visual))
    record("clip_spatial", spatial_moments(visual))
    for i, layer in enumerate(encoder.adapter.layers, 1):
        hooks.append(layer.register_forward_hook(layer_hook(i)))
    hooks.append(encoder.adapter.register_forward_hook(lambda m, a, o: record("adapter8_pool", o)))
    hooks.append(encoder.projection.register_forward_hook(lambda m, a, o: record("projection64", o)))
    encoder.set_tokens(visual)
    try:
        with torch.no_grad():
            out = encoder({"agent_pos": visual.new_zeros((len(visual), 2))})
        rgb = out[:, encoder.rgb_columns].cpu().numpy()
        if not np.array_equal(rgb, found["projection64"]):
            raise ValueError("Hook output differs from original frontend RGB boundary")
    finally:
        encoder.cached_tokens = None
        for hook in hooks:
            hook.remove()
    if len(found) != 8 or any(not np.isfinite(v).all() for v in found.values()):
        raise ValueError("Incomplete or invalid frozen features")
    return found

"""Install an isolated post-Adapter readout without renaming legacy weights."""
import types
import torch
from bridge_spatial_readout_prototype_v1 import SpatialPatchReadout


def _forward_with_readout(self, visual_tokens, text_tokens, text_attention_mask=None):
    tokens = visual_tokens
    for layer in self.layers:
        tokens = layer(tokens, text_tokens, text_attention_mask)
    return self.output_norm(self.spatial_readout(tokens))


def install_readout(adapter, *, candidate):
    if adapter.pooling != 'cls_patch_mean' or hasattr(adapter, 'spatial_readout'):
        raise ValueError('Require an unmodified archived cls_patch_mean Adapter')
    dimension = adapter.output_norm.normalized_shape[0]
    reference = adapter.output_norm.weight
    adapter.spatial_readout = SpatialPatchReadout(dimension).to(
        device=reference.device, dtype=reference.dtype)
    adapter.spatial_readout.requires_grad_(candidate)
    adapter.forward = types.MethodType(_forward_with_readout, adapter)
    return adapter.spatial_readout


def context_from_tokens(model, visual, language, mask, current_state):
    if current_state.shape != (len(visual), 9):
        raise ValueError('Require nine CURRENT state features; no future/record flag')
    now = model.adapter(visual, language, mask)
    repeated = model.adapter(visual, language, mask)
    return now + model.history_projection(repeated-now) + model.state_projection(current_state)


def reset_readout(readout, candidate):
    with torch.no_grad():
        readout.content_score.zero_()
        readout.position_score.zero_()
    readout.requires_grad_(candidate)

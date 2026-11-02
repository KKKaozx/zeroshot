"""Remove only the eight cross-attention/FFN blocks after matched initialization."""
import torch
from torch import nn
from resource_probe import state_hash


def remove_blocks(encoder):
    if getattr(encoder,'zero_block_control',False):
        raise ValueError('Already converted')
    adapter=encoder.adapter
    if len(adapter.layers)!=8 or adapter.pooling!='cls_patch_mean':
        raise ValueError('Expected pinned eight-block mean/CLS frontend')
    if encoder.projection.in_features!=1024 or encoder.projection.out_features!=64 or encoder.state_columns!=(64,65):
        raise ValueError('Changed common boundary')
    old_pool,old_projection=adapter.output_norm,encoder.projection
    before_rng=torch.get_rng_state().clone()
    initial_frontend_sha=state_hash(encoder)
    norm_sha,projection_sha=state_hash(old_pool),state_hash(old_projection)
    old_parameters=sum(p.numel() for p in encoder.parameters())
    removed_parameters=sum(p.numel() for p in adapter.layers.parameters())
    adapter.layers=nn.ModuleList()
    encoder.zero_block_control=True
    if not (adapter.output_norm is old_pool and encoder.projection is old_projection
            and state_hash(old_pool)==norm_sha and state_hash(old_projection)==projection_sha
            and torch.equal(before_rng,torch.get_rng_state())):
        raise ValueError('Removing blocks changed shared initialization/RNG')
    remaining=sum(p.numel() for p in encoder.parameters())
    if remaining!=old_parameters-removed_parameters or remaining!=67648:
        raise ValueError('Unexpected remaining frontend parameters')
    return dict(initial_eight_frontend_sha256=initial_frontend_sha,common_norm_sha256=norm_sha,
        common_projection_sha256=projection_sha,original_frontend_parameters=old_parameters,
        removed_block_parameters=removed_parameters,remaining_frontend_parameters=remaining,
        common_parameter_objects_preserved=True,rng_unchanged=True,active_cross_attention_blocks=0,
        active_text_conditioning=False,
        scope='No attention or FFN blocks; retain original pooled output LayerNorm and1024-to64 projection')


def formula_check(encoder,tokens,positions):
    if not getattr(encoder,'zero_block_control',False) or len(encoder.adapter.layers):
        raise ValueError('Expected zero-block frontend')
    encoder.set_tokens(tokens)
    try:
        actual=encoder({'agent_pos':positions})
    finally:
        encoder.cached_tokens=None
    expected_rgb=encoder.projection(encoder.adapter.output_norm(.5*(tokens[:,0]+tokens[:,1:].mean(1))))
    if not torch.equal(actual[:,:64],expected_rgb) or not torch.equal(actual[:,64:],positions):
        raise ValueError('Zero-block formula or state columns differ')
    return actual

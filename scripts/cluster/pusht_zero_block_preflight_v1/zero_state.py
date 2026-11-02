"""Separate checkpoint identity for the zero-block ablation; no eight-block resume."""
import training_core as core

FORMAT='pusht-zero-block-complete-v1'


def capture(policy,averaged,optimizer,scheduler,ema,completed,base_sha,plan_sha,protocol_sha):
    if len(policy.obs_encoder.adapter.layers) or len(averaged.obs_encoder.adapter.layers):
        raise ValueError('Zero-block state only')
    p=core.capture(policy,averaged,optimizer,scheduler,ema,completed,base_sha,plan_sha)
    p.update(format=FORMAT,zero_block_protocol_sha256=protocol_sha,active_cross_attention_blocks=0)
    return p


def restore(payload,policy,averaged,optimizer,scheduler,ema,base_sha,plan_sha,protocol_sha):
    if (payload.get('format')!=FORMAT or payload.get('zero_block_protocol_sha256')!=protocol_sha
            or payload.get('active_cross_attention_blocks')!=0):
        raise ValueError('Wrong ablation identity')
    if len(policy.obs_encoder.adapter.layers) or len(averaged.obs_encoder.adapter.layers):
        raise ValueError('Cannot restore into eight-block model')
    view=dict(payload,format='pusht-native-complete-v1')
    return core.restore(view,policy,averaged,optimizer,scheduler,ema,base_sha,plan_sha)

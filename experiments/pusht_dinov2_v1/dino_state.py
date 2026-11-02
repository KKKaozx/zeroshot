"""Complete optimizer/RNG state, strictly separated from CLIP and other ablations."""
import training_core as core

FORMAT = 'pusht-frozen-dino8-complete-v1'


def capture(policy,averaged,optimizer,scheduler,ema,completed,base_sha,plan_sha,contract_sha,binding_sha):
    if any(len(model.obs_encoder.adapter.layers) != 8 for model in (policy,averaged)):
        raise ValueError('DINO control must preserve the original eight attention blocks')
    payload = core.capture(policy,averaged,optimizer,scheduler,ema,completed,base_sha,plan_sha)
    payload.update(format=FORMAT,project_contract_sha256=contract_sha,
        dino_backbone_binding_sha256=binding_sha,active_cross_attention_blocks=8)
    return payload


def restore(payload,policy,averaged,optimizer,scheduler,ema,base_sha,plan_sha,contract_sha,binding_sha):
    if (payload.get('format') != FORMAT or payload.get('project_contract_sha256') != contract_sha
            or payload.get('dino_backbone_binding_sha256') != binding_sha
            or payload.get('active_cross_attention_blocks') != 8):
        raise ValueError('Wrong DINO/cache/contract identity; original CLIP/native checkpoints forbidden')
    if any(len(model.obs_encoder.adapter.layers) != 8 for model in (policy,averaged)):
        raise ValueError('DINO checkpoint destination must preserve eight attention blocks')
    return core.restore(dict(payload,format='pusht-native-complete-v1'),policy,averaged,
                        optimizer,scheduler,ema,base_sha,plan_sha)

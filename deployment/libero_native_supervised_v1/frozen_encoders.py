"""Existing pinned DINO and CLIP text assets; no network, no training fallback."""
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'code'))


def text_weights(directory):
    import torch
    directory=Path(directory)
    if (directory/'model.safetensors').is_file():
        from safetensors import safe_open
        with safe_open(directory/'model.safetensors',framework='pt',device='cpu') as f:
            result={name:f.get_tensor(name) for name in f.keys() if name.startswith('text_model.') and not name.endswith('position_ids')}
    elif (directory/'pytorch_model.bin').is_file():
        all_weights=torch.load(directory/'pytorch_model.bin',map_location='cpu',weights_only=True)
        result={name:value for name,value in all_weights.items() if name.startswith('text_model.') and not name.endswith('position_ids')}
    else:raise FileNotFoundError('Existing CLIP model.safetensors or pytorch_model.bin required: '+str(directory))
    return result


def text_digest(weights):
    h=hashlib.sha256()
    for name,value in sorted(weights.items()):h.update(name.encode());h.update(value.contiguous().numpy().tobytes())
    return h.hexdigest()


def restore_text(model,weights):
    """Exact one-to-one namespace mapping for Transformers4.x/5.x text models."""
    expected=set(model.state_dict());provided=set(weights)
    if expected==provided:mapped=weights
    elif all(k.startswith('text_model.') for k in provided) and {k[len('text_model.'):] for k in provided}==expected:
        mapped={k[len('text_model.'):]:v for k,v in weights.items()}
    else:raise ValueError('CLIP text namespace/complete parameter set differs; no partial restore')
    model.load_state_dict(mapped,strict=True)
    restored={k if k.startswith('text_model.') else 'text_model.'+k:v for k,v in model.state_dict().items()}
    if text_digest(restored)!=text_digest(weights):raise ValueError('Frozen text restore changed actual tensors')
    return model


def check_assets(dino_assets,clip_assets,binding):
    from frozen_visual_backbone import DINO_ASSET_SHAS,file_sha
    for name,digest in DINO_ASSET_SHAS.items():
        path=Path(dino_assets)/name
        if not path.is_file() or file_sha(path)!=digest:raise ValueError('Existing pinned DINO asset missing/different: '+str(path))
    weights=text_weights(clip_assets)
    if sorted(weights)!=sorted(binding['text_parameter_names']) or text_digest(weights)!=binding['text_parameters_sha256']:
        raise ValueError('Existing CLIP text parameters differ from bound OpenAI snapshot')
    return weights


def encode(images,input_ids,attention_mask,dino_assets,text_state,*,progress=None):
    import torch
    from transformers import CLIPTextModel,CLIPTextConfig
    from frozen_visual_backbone import FrozenDinoVision,dino_config
    from native_data import prepare_pixels
    config=CLIPTextConfig.from_dict(json.loads((HERE/'text_config.json').read_text(encoding='utf-8')))
    config._attn_implementation='eager'
    text=restore_text(CLIPTextModel(config),text_state)
    text=text.cuda().requires_grad_(False).eval()
    with torch.no_grad():
        language=text(input_ids=torch.as_tensor(input_ids,device='cuda'),
            attention_mask=torch.as_tensor(attention_mask,device='cuda')).last_hidden_state.detach()
    del text,text_state;torch.cuda.empty_cache()
    vision=FrozenDinoVision(dino_config(dino_assets)).cuda().requires_grad_(False).eval()
    bank=torch.empty((len(images),257,1024),device='cuda',dtype=torch.float32)
    with torch.no_grad():
        for start in range(0,len(images),8):
            pixels=prepare_pixels(images[start:start+8]).cuda()
            value=vision(pixel_values=pixels).last_hidden_state
            bank[start:start+len(value)].copy_(value)
            if progress is not None:progress(start+len(value),len(images))
    if any(p.grad is not None or p.requires_grad for p in vision.parameters()):raise ValueError('Frozen vision contract failed')
    del vision,pixels,value;torch.cuda.empty_cache()
    return bank,language

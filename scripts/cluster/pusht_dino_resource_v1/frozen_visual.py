"""Independent frozen DINOv2 visual control; no changes to the CLIP main model."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

WEIGHT_SHA256='399fba97a95f22c36834418bc69373364a99af3a1153da1c0fb31db567c92e23'
REVISION='47b73eefe95e8d44ec3623f8890bd894b6ea2d6c'
MEAN=(.485,.456,.406)
STD=(.229,.224,.225)
ASSET_SHAS={
    'model.safetensors':WEIGHT_SHA256,
    'config.json':'12df51c069a2dc1305e34ba71ef58bc2407ea553b75f4722a1715c1bce3bbed0',
    'preprocessor_config.json':'14e780d86fa1861f8751f868d7f45425b5feb55c38ca26f152ca5097ab30f828',
}


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()


def model_sha(model):
    h=hashlib.sha256()
    for name,x in sorted(model.state_dict().items()):
        h.update(name.encode());h.update(x.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def raw_images(value):
    if (not isinstance(value,np.ndarray) or value.dtype!=np.float32 or value.ndim!=4
            or value.shape[1:]!=(96,96,3) or not 1<=len(value)<=32
            or not np.isfinite(value).all() or value.min()<0 or value.max()>255):
        raise ValueError('Expected raw float32 [N,96,96,3] in [0,255]')
    return value


class FrozenDinoVisual(nn.Module):
    """Final native-normalized CLS+patch tokens at224px, fixed pad-to8 execution."""
    def __init__(self,assets,device='cuda'):
        super().__init__()
        from transformers import CLIPImageProcessor,Dinov2Model
        assets=Path(assets)
        for name,digest in ASSET_SHAS.items():
            if file_sha(assets/name)!=digest:raise ValueError('Unexpected DINOv2 asset: '+name)
        cfg=json.loads((assets/'config.json').read_text())
        if (cfg['hidden_size'],cfg['patch_size'],cfg['num_hidden_layers'])!=(1024,14,24):
            raise ValueError('Expected DINOv2 large/14, no width adaptation')
        # Use the same224-resize/224-crop/PIL geometry as the original square CLIP
        # input, with native DINO ImageNet normalization. Official DINO defaults
        # resize256 then crop224: that would alter the visible field of view.
        self.processor=CLIPImageProcessor(do_resize=True,size={'shortest_edge':224},
            resample=3,do_center_crop=True,crop_size={'height':224,'width':224},
            do_rescale=True,rescale_factor=1/255,do_normalize=True,
            image_mean=list(MEAN),image_std=list(STD),do_convert_rgb=True)
        self.model=Dinov2Model.from_pretrained(str(assets),local_files_only=True,
            use_safetensors=True,attn_implementation='eager')
        self.model.requires_grad_(False).eval().to(device)
        self.train(False)

    def train(self,mode=True):
        super().train(False)
        return self

    def preprocess(self,images):
        value=raw_images(images)
        pixels=self.processor(images=list(value),return_tensors='pt',do_rescale=True)['pixel_values']
        if tuple(pixels.shape)!=(len(value),3,224,224) or not torch.isfinite(pixels).all():
            raise ValueError('Invalid preprocessed image shape/values')
        return pixels

    def forward(self,images):
        value=raw_images(images)
        result=[]
        with torch.inference_mode():
            for start in range(0,len(value),8):
                part=value[start:start+8];count=len(part)
                if count<8:part=np.concatenate((part,np.repeat(part[-1:],8-count,axis=0)))
                pixels=self.preprocess(part).to(next(self.model.parameters()).device)
                tokens=self.model(pixel_values=pixels).last_hidden_state[:count]
                if tokens.shape!=(count,257,1024) or not torch.isfinite(tokens).all():
                    raise ValueError('DINO token contract failed')
                # CPU copy out of inference tensors for downstream trainable Adapter.
                result.append(tokens.cpu().numpy().copy())
        return np.concatenate(result)

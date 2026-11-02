"""Pinned DINOv2-L/14 behind the main project's canonical image interface.

Dataset and live policy both use dataset.prepare_image. Its geometry and padding
stay identical to the CLIP control; only channel normalization changes here.
This is not the Push-T PIL/bicubic image pipeline or a task-weight migration.
"""
import hashlib
import json
from pathlib import Path

import torch
from torch import nn

DINO_REVISION='47b73eefe95e8d44ec3623f8890bd894b6ea2d6c'
DINO_ASSET_SHAS={
    'model.safetensors':'399fba97a95f22c36834418bc69373364a99af3a1153da1c0fb31db567c92e23',
    'config.json':'12df51c069a2dc1305e34ba71ef58bc2407ea553b75f4722a1715c1bce3bbed0',
    'preprocessor_config.json':'14e780d86fa1861f8751f868d7f45425b5feb55c38ca26f152ca5097ab30f828',
}
VISUAL_INPUT_CONTRACT='canonical_clip_224_letterbox_to_dino_imagenet_v1'
CLIP_MEAN=(.48145466,.4578275,.40821073)
CLIP_STD=(.26862954,.26130258,.27577711)
DINO_MEAN=(.485,.456,.406)
DINO_STD=(.229,.224,.225)


def file_sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024**2),b''):digest.update(block)
    return digest.hexdigest()


def dino_config(assets):
    if not assets:raise ValueError('DINO requires explicit --dino-assets; no remote fallback')
    return {'visual_backbone':'dinov2', 'dino_assets':str(assets),
        'visual_input_contract':VISUAL_INPUT_CONTRACT,
        'visual_asset_identity':{'repository':'facebook/dinov2-large',
            'revision':DINO_REVISION,'sha256':dict(DINO_ASSET_SHAS)}}


def validate_dino_config(settings):
    expected=dino_config(settings.get('dino_assets'))
    for key in ('visual_backbone','visual_input_contract','visual_asset_identity'):
        if settings.get(key)!=expected[key]:
            raise ValueError('DINO checkpoint/config visual binding differs: '+key)
    path=Path(settings['dino_assets'])
    if not path.is_dir():raise ValueError('Existing local DINO asset directory required')
    return path


class FrozenDinoVision(nn.Module):
    def __init__(self,settings):
        super().__init__()
        from transformers import Dinov2Model
        assets=validate_dino_config(settings)
        for name,digest in DINO_ASSET_SHAS.items():
            if file_sha(assets/name)!=digest:raise ValueError('DINO asset identity differs: '+name)
        cfg=json.loads((assets/'config.json').read_text())
        if (cfg.get('hidden_size'),cfg.get('patch_size'),cfg.get('num_hidden_layers'),
                cfg.get('model_type'))!=(1024,14,24,'dinov2'):
            raise ValueError('Expected frozen DINOv2 ViT-L/14, without width conversion')
        self.encoder=Dinov2Model.from_pretrained(str(assets),local_files_only=True,
            trust_remote_code=False,use_safetensors=True,attn_implementation='eager')
        self.config=self.encoder.config
        for name,values in (('clip_mean',CLIP_MEAN),('clip_std',CLIP_STD),
                            ('dino_mean',DINO_MEAN),('dino_std',DINO_STD)):
            self.register_buffer(name,torch.tensor(values).view(1,3,1,1),persistent=False)
        self.requires_grad_(False).eval()

    def train(self,mode=True):
        super().train(False)
        return self

    def dino_pixels(self,pixel_values):
        if pixel_values.ndim!=4 or tuple(pixel_values.shape[1:])!=(3,224,224):
            raise ValueError('Main DINO input must be canonical normalized [B,3,224,224]')
        return (pixel_values*self.clip_std+self.clip_mean-self.dino_mean)/self.dino_std

    def forward(self,*,pixel_values):
        # no_grad rather than inference_mode: these tokens feed trainable modules.
        with torch.no_grad():
            output=self.encoder(pixel_values=self.dino_pixels(pixel_values))
        if output.last_hidden_state.shape[1:]!=(257,1024):
            raise ValueError('DINO CLS+patch token interface differs')
        return output

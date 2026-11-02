"""Target-supervised native control contract. Never use OXE pose decoding here."""
import numpy as np

TASK='KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet'
INSTRUCTION='close the top drawer of the cabinet'
HORIZON=16
EXECUTE=4
NATIVE_HZ=20


def causal_chunks(actions):
    """obs[i] is POST action[i]; targets begin at action[i+1], not action[i]."""
    actions=np.asarray(actions,dtype=np.float32)
    if actions.ndim!=2 or actions.shape[1]!=7 or len(actions)<2 or not np.isfinite(actions).all():
        raise ValueError('Finite recorded native[T,7] actions required')
    if np.any(np.abs(actions)>1+1e-6):raise ValueError('Recorded native commands exceed declared controller bounds')
    count=len(actions)-1;target=np.zeros((count,HORIZON,7),np.float32);mask=np.zeros((count,HORIZON,1),np.float32)
    for i in range(count):
        n=min(HORIZON,len(actions)-i-1);target[i,:n]=actions[i+1:i+1+n];mask[i,:n]=1
    return target,mask


def fit_normalizer(proprio,unique_future_actions):
    proprio=np.asarray(proprio,dtype=np.float64);actions=np.asarray(unique_future_actions,dtype=np.float64)
    if proprio.ndim!=2 or proprio.shape[1]!=9 or actions.ndim!=2 or actions.shape[1]!=7:
        raise ValueError('Observed7 joints+2 fingers and native7 commands required')
    low,high=actions.min(0),actions.max(0);scale=(high-low)/2
    scale=np.where(scale<=1e-6,0.,scale)
    return {'proprio_mean':proprio.mean(0).tolist(),'proprio_std':np.maximum(proprio.std(0),1e-3).tolist(),
        'action_center':((high+low)/2).tolist(),'action_scale':scale.tolist(),
        'train_action_mean':actions.mean(0).tolist(),'native_low':[-1.]*7,'native_high':[1.]*7,
        'scope':'Fit train demos0..19 only; constant action dimensions decode exactly to their training constant'}


def proprio_features(proprio,normalizer):
    value=np.asarray(proprio,dtype=np.float64)
    if value.shape[-1:]!=(9,) or not np.isfinite(value).all():raise ValueError('Finite causal9D proprio required')
    return ((value-normalizer['proprio_mean'])/normalizer['proprio_std']).astype(np.float32)


def normalize_actions(actions,normalizer):
    scale=np.asarray(normalizer['action_scale']);denom=np.where(scale>0,scale,1.)
    return ((np.asarray(actions)-normalizer['action_center'])/denom).astype(np.float32)


def decode_actions(values,normalizer):
    values=np.asarray(values,dtype=np.float32)
    if values.shape[-1:]!=(7,) or not np.isfinite(values).all():raise ValueError('Finite native7 predictions required')
    return np.clip(np.asarray(normalizer['action_center'])+np.clip(values,-1,1)*normalizer['action_scale'],-1,1).astype(np.float32)


def prepare_pixels(raw_rgb):
    """Raw OpenGL bottom-up RGB128; flip once, canonical224 CLIP normalization."""
    import torch
    import torch.nn.functional as F
    array=np.asarray(raw_rgb)
    if array.ndim==3:array=array[None]
    if array.dtype!=np.uint8 or array.shape[1:]!=(128,128,3):raise ValueError('Declared raw128 RGB required')
    pixels=torch.from_numpy(array[:,::-1].copy()).permute(0,3,1,2).float()/255
    pixels=F.interpolate(pixels,size=(224,224),mode='bilinear',align_corners=False)
    mean=pixels.new_tensor([.48145466,.4578275,.40821073])[None,:,None,None]
    std=pixels.new_tensor([.26862954,.26130258,.27577711])[None,:,None,None]
    return (pixels-mean)/std

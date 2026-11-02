"""Source-only affine training coordinates; external8 command stays unchanged."""
import copy
import types

import numpy as np
import torch
import torch.nn.functional as F


class MotionStandardizer:
    def __init__(self,spec):
        self.spec=copy.deepcopy(spec)
        if (spec['format']!='bridge-command8-motion-standardization-v1' or spec['fit_partition']!='train'
                or spec['libero_data_used'] or spec['development_data_fit']):
            raise ValueError('Bound source-training-only normalizer required')
        self.mean=np.array(spec['mean'],np.float64);self.scale=np.array(spec['scale'],np.float64)
        self.low=np.array(spec['physical_lower'],np.float64);self.high=np.array(spec['physical_upper'],np.float64)
        if (any(v.shape!=(7,) or not np.isfinite(v).all() for v in (self.mean,self.scale,self.low,self.high))
                or (self.scale<=0).any() or not (self.low<self.high).all()):
            raise ValueError('Invalid motion standardization')
        self._cache={}

    def constants(self,x):
        key=(str(x.device),x.dtype)
        if key not in self._cache:
            self._cache[key]=[torch.as_tensor(v,dtype=x.dtype,device=x.device) for v in (self.mean,self.scale,self.low,self.high)]
        return self._cache[key]

    def encode(self,physical):
        mean,scale,_,_=self.constants(physical)
        return (physical-mean)/scale

    def decode(self,standardized):
        mean,scale,_,_=self.constants(standardized)
        return standardized*scale+mean

    def bound_clean(self,standardized):
        physical=self.decode(standardized);_,_,low,high=self.constants(physical)
        return self.encode(torch.maximum(torch.minimum(physical,high),low))


@torch.no_grad()
def sample_standardized(self,context,current_gripper=None):
    """DDPM in standardized space; inverse only for explicit physical operations."""
    if self.diffusion_prediction_type!='sample' or not self.separate_gripper_head:
        raise ValueError('Fixed x0/separate-gripper architecture required')
    norm=self.command_motion_standardizer
    actions=torch.randn(context.shape[0],self.chunk_size,7,device=context.device,dtype=context.dtype)
    for step in reversed(range(self.num_diffusion_steps)):
        timesteps=torch.full((context.shape[0],),step,device=context.device,dtype=torch.long)
        clean=self.diffusion_decoder(actions,timesteps,context)
        clean=norm.bound_clean(clean)
        mean=self.posterior_mean_x0[step]*clean+self.posterior_mean_xt[step]*actions
        actions=mean+self.posterior_variance[step].sqrt()*torch.randn_like(actions) if step else mean
    physical=norm.decode(actions)
    physical[...,:3]=physical[...,:3].clamp(-.5,.5)
    q=physical[...,3:7];identity=torch.zeros_like(q);identity[...,3]=1
    physical[...,3:7]=torch.where(q.norm(dim=-1,keepdim=True)>1e-6,F.normalize(q,dim=-1),identity)
    # The gripper's pose features use exactly the same standardized convention
    # as the expert motion supplied to its training loss.
    logits=self.predict_gripper_logits(context,norm.encode(physical),current_gripper)
    gripper=torch.where(logits>=0,torch.ones_like(logits),-torch.ones_like(logits))
    return torch.cat((physical,gripper[...,None]),dim=-1)


def install_standardized_sampler(model,spec):
    if (getattr(model,'decoder_type','diffusion')!='diffusion'
            or model.diffusion_prediction_type!='sample' or model.gripper_target_mode!='state'):
        raise ValueError('No regression/epsilon/event policy reinterpretation')
    # Ordinary Python metadata: frozen encoder/schedule state identity unchanged.
    model.command_motion_standardizer=MotionStandardizer(spec)
    model.sample=types.MethodType(sample_standardized,model)


def training_coordinates(x,spec):
    norm=spec if isinstance(spec,MotionStandardizer) else MotionStandardizer(spec)
    out=dict(x);out['command_actions']=x['actions']
    out['actions']=torch.cat((norm.encode(x['actions'][...,:7]),x['actions'][...,7:8]),dim=-1)
    return out

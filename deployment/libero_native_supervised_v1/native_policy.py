"""Fresh Adapter + native7 x0 DDPM. No quaternion interpretation or OXE load."""
import math
from pathlib import Path
import sys

import torch
from torch import nn
import torch.nn.functional as F

sys.path.insert(0,str(Path(__file__).resolve().parent/'vendor'))
from adapter import CrossAttentionAdapter
from diffusion_decoder import MultiscaleConditionalUnet1D


class NativeActionPolicy(nn.Module):
    def __init__(self,config):
        super().__init__();self.config=config;dim=config.get('context_dim',1024)
        self.adapter=CrossAttentionAdapter(num_layers=config.get('adapter_layers',8),embed_dim=dim,
            text_dim=config.get('text_dim',768),attention_dim=config.get('attention_dim',512),
            num_heads=config.get('heads',8),dropout=config.get('dropout',.1),pooling='cls_patch_mean')
        self.state_projection=nn.Linear(9,dim,bias=False);nn.init.zeros_(self.state_projection.weight)
        self.diffusion_decoder=MultiscaleConditionalUnet1D(action_dim=7,chunk_size=16,context_dim=dim,
            down_dims=tuple(config.get('down_dims',[256,512,1024])),
            diffusion_step_embed_dim=config.get('step_embed_dim',128),kernel_size=5,num_steps=100)
        t=torch.linspace(0,1,101,dtype=torch.float64)
        cumulative=torch.cos((t+.008)/1.008*math.pi/2).square()
        betas=(1-cumulative[1:]/cumulative[:-1]).clamp(max=.999).float()
        alpha=1-betas;bars=torch.cumprod(alpha,0);prev=F.pad(bars[:-1],(1,0),value=1.)
        for name,value in {'alpha_bars':bars,'posterior_variance':(betas*(1-prev)/(1-bars)).clamp(min=1e-20),
            'posterior_x0':betas*prev.sqrt()/(1-bars),'posterior_xt':(1-prev)*alpha.sqrt()/(1-bars)}.items():
            self.register_buffer(name,value)

    def context(self,visual,language,attention_mask,proprio):
        if proprio.shape!=(len(visual),9):raise ValueError('Causal7joint+2finger state required')
        return self.adapter(visual,language,attention_mask)+self.state_projection(proprio)

    def loss(self,target,mask,context):
        if target.shape!=(len(context),16,7) or mask.shape!=(len(context),16,1):
            raise ValueError('Native7 chunk and valid-future mask required')
        timesteps=torch.randint(0,100,(len(context),),device=context.device)
        noise=torch.randn_like(target);bar=self.alpha_bars[timesteps,None,None]
        noisy=bar.sqrt()*target+(1-bar).sqrt()*noise
        estimate=self.diffusion_decoder(noisy,timesteps,context)
        return ((estimate-target).square()*mask).sum()/(mask.sum()*7).clamp(min=1)

    @torch.no_grad()
    def sample(self,context):
        value=torch.randn((len(context),16,7),device=context.device)
        for step in reversed(range(100)):
            ts=torch.full((len(context),),step,device=context.device,dtype=torch.long)
            clean=self.diffusion_decoder(value,ts,context).clamp(-1,1)
            value=self.posterior_x0[step]*clean+self.posterior_xt[step]*value
            if step:value=value+self.posterior_variance[step].sqrt()*torch.randn_like(value)
        return value.clamp(-1,1)

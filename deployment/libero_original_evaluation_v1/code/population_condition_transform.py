"""Explicit train-population condition transform around an unchanged decoder.

Calibration is a separate operation. Forward never estimates statistics from its
current batch, so a saved checkpoint can use the same transform at batch size1.
"""
import torch
from torch import nn


class PopulationConditionDecoder(nn.Module):
    def __init__(self,core:nn.Module,condition_dim:int=1024,min_scale:float=1e-6):
        super().__init__();self.core=core;self.min_scale=min_scale;self.enabled=True
        self.register_buffer('context_mean',torch.zeros(condition_dim))
        self.register_buffer('context_scale',torch.ones(()))

    @torch.no_grad()
    def calibrate(self,training_contexts:torch.Tensor):
        assert training_contexts.ndim==2 and training_contexts.shape[1]==len(self.context_mean)
        assert len(training_contexts)>1 and torch.isfinite(training_contexts).all()
        values=training_contexts.detach().double();mean=values.mean(0)
        scale=(values-mean).square().mean().sqrt()
        self.context_mean.copy_(mean);self.context_scale.copy_(scale.clamp_min(self.min_scale))
        return dict(samples=len(values),scale=float(scale),mean_norm=float(mean.norm()),scale_clamped=bool(scale<self.min_scale))

    def transform(self,context:torch.Tensor):
        return (context-self.context_mean)/self.context_scale

    def forward(self,noisy,timestep,context):
        return self.core(noisy,timestep,self.transform(context) if self.enabled else context)

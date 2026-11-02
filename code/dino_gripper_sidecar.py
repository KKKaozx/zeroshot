"""A separate gripper for229116: no optimizer can touch the pose checkpoint.

Inputs retain the source recipe:1024D frozen context, generated tool-relative
7D poses, measured Bridge opening encoded2*x-1 (NOT command history). Labels
are the existing source's binary absolute open-positive commands.
"""
import hashlib

import numpy as np
import torch
from torch import nn

SOURCE_SHA = '730f06ecb417ff98d7e5270c114c2b062ce38347f0987c136b9d0359e8c4a6f5'
CURRENT_SEMANTICS = 'bridge_measured_affine_unbounded_v1'


class GripperSidecar(nn.Module):
    def __init__(self):
        super().__init__()
        self.context_projection = nn.Linear(1024,256)
        self.pose_projection = nn.Linear(7,256)
        self.current_projection = nn.Linear(1,256)
        self.time_embedding = nn.Embedding(16,256)
        self.readout = nn.Sequential(nn.LayerNorm(256),nn.Mish(),nn.Linear(256,1))

    def forward(self, context, generated_poses, measured_current):
        if (context.ndim != 2 or context.shape[1] != 1024
                or generated_poses.shape != (len(context),16,7)
                or measured_current.shape != (len(context),1)):
            raise ValueError('Source1024D context, generated H16x7 and measured opening scalar required')
        features = (self.context_projection(context.detach())[:,None]
            + self.pose_projection(generated_poses.detach())
            + self.current_projection(measured_current.detach())[:,None]
            + self.time_embedding(torch.arange(16,device=context.device))[None])
        return self.readout(features).squeeze(-1)


def replace_gripper(source_model, head, context, measured_current, *, source_checkpoint_sha256,
                    current_semantics=CURRENT_SEMANTICS):
    """Same source.sample invocation and unchanged first7 columns, no extra draw.

    The caller builds context using the ORIGINAL adapter/history/9D state port.
    Equal current observations are required for the pose-preservation claim;
    changed gripper actions can change later closed-loop observations.
    """
    if source_checkpoint_sha256 != SOURCE_SHA or current_semantics != CURRENT_SEMANTICS:
        raise ValueError('Only the exact229116 source and its measured-input semantics are supported')
    if source_model.training or head.training or any(p.requires_grad for p in source_model.parameters()):
        raise ValueError('Frozen eval source policy and eval sidecar required')
    with torch.inference_mode():
        original = source_model.sample(context,current_gripper=measured_current)
        if original.shape != (len(context),16,8) or not torch.isfinite(original).all():
            raise ValueError('Expected valid source H16x8 sampler output')
        commands = torch.where(head(context,original[...,:7],measured_current) >= 0,1.,-1.)
        result = original.clone()
        result[...,7] = commands
        if not torch.equal(result[...,:7],original[...,:7]):
            raise RuntimeError('Sidecar changed source pose output')
    return result


def measured_feature(open_fraction):
    value = float(open_fraction)
    if not np.isfinite(value):
        raise ValueError('Finite source-style measured opening fraction required')
    #Exact old dataset rule: do not clamp, binarize, or replace by last command.
    return np.asarray([2.0 * value - 1.0],dtype='<f4')


def source_state_features(position, quaternion_xyzw, stats):
    """Original9D source-state port, including the saved128-row XYZ statistics."""
    from scipy.spatial.transform import Rotation
    p,q = np.asarray(position,dtype=np.float64),np.asarray(quaternion_xyzw,dtype=np.float64)
    mean,std = np.asarray(stats['xyz_mean']),np.asarray(stats['xyz_std'])
    if (p.shape != (3,) or q.shape != (4,) or mean.shape != (3,) or std.shape != (3,)
            or (std <= 0).any() or not np.isfinite(np.r_[p,q,mean,std]).all()
            or abs(np.linalg.norm(q)-1) > 1e-3):
        raise ValueError('Finite measured source pose and original statistics required')
    matrix = Rotation.from_quat(q).as_matrix()
    return np.r_[(p-mean)/std,matrix[:,0],matrix[:,1]].astype('<f4')


def immutable_pose_digest(model):
    h = hashlib.sha256()
    for name,value in model.state_dict().items():
        raw = value.detach().cpu().contiguous()
        h.update(name.encode()); h.update(str(raw.dtype).encode()); h.update(str(list(raw.shape)).encode())
        h.update(raw.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def gripper_metrics(prediction, truth):
    """Absolute command and within-window transitions, no invented prior command."""
    p,t = np.asarray(prediction,dtype=bool),np.asarray(truth,dtype=bool)
    if p.ndim != 3 or p.shape[1:] != t.shape or t.shape[1] != 16:
        raise ValueError('Draws x windows x16 predictions and windows x16 commands required')
    t = np.broadcast_to(t,p.shape)
    recall = {label:float((p[t==value]==value).mean()) if (t==value).any() else None
              for label,value in [('open',True),('closed',False)]}
    transitions = {}
    for label,value in [('opening',True),('closing',False)]:
        actual = (t[:,:,1:] != t[:,:,:-1]) & (t[:,:,1:] == value)
        guessed = (p[:,:,1:] != p[:,:,:-1]) & (p[:,:,1:] == value)
        tp = int((actual&guessed).sum()); fp = int((~actual&guessed).sum()); fn = int((actual&~guessed).sum())
        transitions[label] = {'true_positive':tp,'false_positive':fp,'false_negative':fn,
            'true_events':int(actual.sum()),'f1':float(2*tp/(2*tp+fp+fn)) if actual.any() else None}
    return {'accuracy':float((p==t).mean()),'class_recall':recall,
        'balanced_accuracy':float(np.mean(list(recall.values()))) if all(x is not None for x in recall.values()) else None,
        'whole_window_exact_rate':float((p==t).all(-1).mean()),'transitions':transitions,
        'scope':'Within16 future commands only; no measured-width-to-prior-command substitution; not grasp success'}

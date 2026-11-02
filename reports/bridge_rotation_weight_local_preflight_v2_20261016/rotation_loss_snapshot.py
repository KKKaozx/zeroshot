"""Change quaternion MSE weight only; diffusion coordinates and sampling stay fixed."""
import hashlib
import numpy as np
import torch
import torch.nn.functional as F

def training_statistics(targets):
    raw=np.asarray(targets)
    assert raw.shape==(1836,16,8) and np.isfinite(raw).all()
    t=raw.astype(np.float64);q=t[...,3:7];xyz=t[...,:3]
    assert np.max(np.abs(np.linalg.norm(q,axis=-1)-1))<1e-5
    assert (q[...,3]>=0).all(), 'Original canonical quaternion labels required'
    identity=np.array([0.,0.,0.,1.])
    pe=float(np.square(xyz).sum(-1).mean())
    qe=float(np.square(q-identity).sum(-1).mean())
    assert pe>0 and qe>0
    return dict(train_windows=1836,train_targets=29376,
        training_targets_sha256=hashlib.sha256(np.ascontiguousarray(raw).tobytes()).hexdigest(),
        xyz_static_squared_sum_mean=pe,quaternion_static_squared_sum_mean=qe,
        weight=pe/qe,quaternion_negative_w_count=int((q[...,3]<0).sum()),
        xyz_std=xyz.reshape(-1,3).std(0).tolist(),quaternion_std=q.reshape(-1,4).std(0).tolist())

def rotation_pose_loss(pred,target,weight):
    assert pred.shape==target.shape and pred.shape[-1]==7
    assert np.isfinite(weight) and weight>0
    base=F.mse_loss(pred,target)
    xyz=F.mse_loss(pred[...,:3],target[...,:3])
    quat=F.mse_loss(pred[...,3:7],target[...,3:7])
    # Preserve the original xyz coefficient1/7. w=1 is the exact old expression.
    loss=base if weight==1.0 else base+((float(weight)-1.)*4./7.)*quat
    return loss,dict(unweighted_pose_mse=float(base.detach()),xyz_component_mse=float(xyz.detach()),
        quaternion_component_mse=float(quat.detach()),weighted_rotation_contribution=float((float(weight)*4./7.*quat).detach()))

def verify_loss_math(weight):
    generator=torch.Generator().manual_seed(810031)
    pred=torch.randn(2,16,7,generator=generator,dtype=torch.float64,requires_grad=True)
    target=torch.randn(2,16,7,generator=generator,dtype=torch.float64)
    baseline=F.mse_loss(pred,target)
    identity,_=rotation_pose_loss(pred,target,1.)
    assert torch.equal(baseline,identity)
    g0=torch.autograd.grad(baseline,pred,retain_graph=True)[0]
    g1=torch.autograd.grad(identity,pred,retain_graph=True)[0]
    assert torch.equal(g0,g1)
    weighted,_=rotation_pose_loss(pred,target,weight)
    g=torch.autograd.grad(weighted,pred)[0]
    assert torch.allclose(g[...,:3],g0[...,:3],atol=1e-14,rtol=1e-12)
    assert torch.allclose(g[...,3:7],weight*g0[...,3:7],atol=1e-14,rtol=1e-12)
    assert torch.isfinite(g).all()
    return dict(weight1_forward_and_gradient_exact=True,xyz_loss_gradient_coefficient_preserved=True,
        quaternion_gradient_multiplied_by_fixed_weight=True,
        maximum_xyz_gradient_difference=float((g[...,:3]-g0[...,:3]).abs().max()),
        note='Shared global gradient clipping may still affect both parameter groups; no separate-gradient claim.')

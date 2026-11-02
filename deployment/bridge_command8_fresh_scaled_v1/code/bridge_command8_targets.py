"""NumPy-only candidate source-command labels. No observed future poses."""
import numpy as np

FORMAT='bridge_controller_increment_pose8_candidate_v1'


def encode_commands(raw, clip_abs, divisor=.1):
    raw=np.asarray(raw,dtype=np.float64)
    bounds=np.asarray(clip_abs,dtype=np.float64)
    if raw.ndim!=2 or raw.shape[1]!=7 or not len(raw) or not np.isfinite(raw).all():
        raise ValueError('Finite raw source command[N,7] required')
    if bounds.shape!=(6,) or not np.isfinite(bounds).all() or (bounds<=0).any() or not np.isfinite(divisor) or divisor<=0:
        raise ValueError('Explicit source controller clip hypothesis and scale required')
    motion=np.clip(raw[:,:6],-bounds,bounds)
    roll,pitch,yaw=(motion[:,i]/2 for i in (3,4,5))
    cr,sr=np.cos(roll),np.sin(roll);cp,sp=np.cos(pitch),np.sin(pitch);cy,sy=np.cos(yaw),np.sin(yaw)
    q=np.stack([sr*cp*cy-cr*sp*sy,cr*sp*cy+sr*cp*sy,cr*cp*sy-sr*sp*cy,cr*cp*cy+sr*sp*sy],axis=1)
    q=np.where(q[:,3:4]<0,-q,q)
    known=np.isin(raw[:,6],[0.,1.])
    values=np.c_[motion[:,:3]/divisor,q,np.where(known,2*raw[:,6]-1,0.)].astype(np.float32)
    mask=np.ones_like(values);mask[:,7]=known
    return values,mask,{'motion_components_clipped':int(np.sum(motion!=raw[:,:6])),
                        'fractional_gripper_rows_masked':int(np.sum(~known))}


def command_chunk(values,mask,start,horizon=16):
    if values.shape!=mask.shape or values.ndim!=2 or values.shape[1]!=8 or not 0<=start<len(values):
        raise ValueError('One episode and a causal command start required')
    n=min(horizon,len(values)-start)
    out=np.zeros((horizon,8),np.float32);out[:,6]=1
    valid=np.zeros_like(out);out[:n]=values[start:start+n];valid[:n]=mask[start:start+n]
    return out,valid

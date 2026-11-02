"""Explicit nominal-rate reached-pose resampling, not measured timestamps.

Images and input observation rows stay unchanged. Raw position values retain
their source units and frame. Native BC-Z future setpoint slots are not used.
"""
import hashlib
import json
import math

import numpy as np

FORMAT='oxe-nominal-reached-grid-v1'
SOURCE_HZ={'Bridge':5,'Language Table':5,'BC-Z':10,'Fractal':3}
OUTPUT_HZ=5
HORIZON=16
POSITION_DIVISOR=.1


def protocol():
    return {'format':FORMAT,'label_family':'nominal_resampled_observed_reached_pose',
        'clock_basis':'user_selected_published_nominal_frequency_assumption',
        'source_hz':dict(SOURCE_HZ),'output_hz':OUTPUT_HZ,'horizon':HORIZON,
        'target_times_seconds':[j/OUTPUT_HZ for j in range(1,HORIZON+1)],
        'timestamp_origin':'relative to the actual input observation row',
        'measured_timestamps_used':False,'exact_export_row_spacing_verified':False,
        'position_interpolation':'linear in recorded source Cartesian frame',
        'orientation_interpolation':'shortest-path unit xyzw SLERP',
        'gripper_rule':'right-closed source interval command: ceil(source_row_time)-1; zero-order hold, no interpolation',
        'current_gripper_rule':'original causal history strictly before the input row',
        'tail_rule':'no extrapolation; repeat last valid target for storage and zero all padded supervision',
        'zero_valid_target_rule':'omit window; never supervise a fabricated target',
        'reference':'fixed_input_observation','action_representation':'source_cartesian_delta_pose_open_positive_v1',
        'position_divisor_in_recorded_units':POSITION_DIVISOR,
        'source_position_to_meters':{source:None for source in SOURCE_HZ},
        'missing_dimensions':'Language Table observes XY only; z,orientation,gripper masked',
        'input_images_or_observation_rows_resampled':False,
        'physical_calibration_or_formal_training_admission_granted':False}


def protocol_sha256(value=None):
    return hashlib.sha256(json.dumps(protocol() if value is None else value,sort_keys=True,
        separators=(',',':'),allow_nan=False).encode()).hexdigest()


def layout(source,start,steps):
    if source not in SOURCE_HZ or type(start) is not int or type(steps) is not int or not 0<=start<steps-1:
        raise ValueError('Known source and a real input row with future observations required')
    # Integer rational arithmetic avoids endpoint drift at .2s and1/3s.
    absolute=start*OUTPUT_HZ+np.arange(1,HORIZON+1,dtype=np.int64)*SOURCE_HZ[source]
    valid=absolute<=(steps-1)*OUTPUT_HZ
    left=absolute//OUTPUT_HZ;remainder=absolute%OUTPUT_HZ
    right=left+(remainder>0)
    command=(absolute+OUTPUT_HZ-1)//OUTPUT_HZ-1
    return {'source':source,'input_row':start,'steps':steps,'valid_targets':int(valid.sum()),
        'left_rows':[int(x) if v else None for x,v in zip(left,valid)],
        'right_rows':[int(x) if v else None for x,v in zip(right,valid)],
        'right_weights':[int(x)/OUTPUT_HZ if v else None for x,v in zip(remainder,valid)],
        'command_rows':[int(x) if v and source!='Language Table' else None for x,v in zip(command,valid)],
        'target_times_seconds':[j/OUTPUT_HZ if v else None for j,v in enumerate(valid,1)],
        'measured_timestamps_used':False}


def normalize_quaternions(values):
    q=np.asarray(values,dtype=np.float64)
    if q.ndim<1 or q.shape[-1]!=4 or not np.isfinite(q).all():raise ValueError('Finite xyzw quaternions required')
    norm=np.linalg.norm(q,axis=-1,keepdims=True)
    if np.any(norm<1e-8):raise ValueError('Missing orientation cannot become identity')
    q=q/norm
    return np.where(q[...,3:4]<0,-q,q)


def slerp(left,right,weight):
    a=normalize_quaternions(left);b=normalize_quaternions(right)
    if a.shape!=b.shape:raise ValueError('Aligned SLERP endpoints required')
    t=np.asarray(weight,dtype=np.float64)
    if t.shape!=a.shape[:-1] or not np.isfinite(t).all() or np.any((t<0)|(t>1)):
        raise ValueError('Finite aligned SLERP weights in [0,1] required')
    dot=np.sum(a*b,axis=-1,keepdims=True)
    b=np.where(dot<0,-b,b);dot=np.clip(np.abs(dot),0,1)
    theta=np.arccos(dot);denominator=np.sin(theta)
    near=dot>.9995
    safe=np.where(near,1.,denominator)
    t=t[...,None]
    curved=(np.sin((1-t)*theta)*a+np.sin(t*theta)*b)/safe
    linear=(1-t)*a+t*b
    return normalize_quaternions(np.where(near,linear,curved))


def relative_left(target,reference):
    a=normalize_quaternions(target);b=normalize_quaternions(reference).copy()
    b[:3]*=-1
    xyz=a[...,3:4]*b[:3]+b[3]*a[...,:3]+np.cross(a[...,:3],b[:3])
    w=a[...,3]*b[3]-np.sum(a[...,:3]*b[:3],axis=-1)
    return normalize_quaternions(np.concatenate((xyz,w[...,None]),axis=-1))


def chunk(source,positions,quaternions,interval_open_commands,current_features,start,
          position_divisor=POSITION_DIVISOR):
    p=np.asarray(positions,dtype=np.float64)
    planar=source=='Language Table';width=2 if planar else 3
    if p.ndim!=2 or p.shape[1]!=width or not np.isfinite(p).all():
        raise ValueError('Finite recorded positions with explicit observed dimensions required')
    if not math.isfinite(position_divisor) or position_divisor<=0:raise ValueError('Positive recorded-unit divisor required')
    meta=layout(source,start,len(p));count=meta['valid_targets']
    if not count:raise ValueError('No valid common-grid future target')
    features=np.asarray(current_features,dtype=np.float32)
    if features.shape!=(len(p),2) or not np.isfinite(features).all():raise ValueError('Finite per-row command/availability features required')
    allowed=((features[:,1]==0)&(features[:,0]==0))|((features[:,1]==1)&np.isin(features[:,0],[-1.,1.]))
    if not allowed.all():raise ValueError('Invalid current command/availability features')
    left=np.asarray(meta['left_rows'][:count]);right=np.asarray(meta['right_rows'][:count])
    alpha=np.asarray(meta['right_weights'][:count])
    target_position=(1-alpha[:,None])*p[left]+alpha[:,None]*p[right]
    target=np.zeros((HORIZON,8),np.float32);target[:,6]=1.;target[:,7]=1.
    mask=np.zeros((HORIZON,8),np.float32)
    target[:count,:width]=((target_position-p[start])/position_divisor).astype(np.float32)
    if planar:
        if quaternions is not None or interval_open_commands is not None or np.any(features):
            raise ValueError('Planar source has no measured quaternion/gripper; do not invent them')
        mask[:count,:2]=1.
    else:
        q=np.asarray(quaternions,dtype=np.float64)
        if q.shape!=(len(p),4):raise ValueError('Aligned recorded orientations required')
        command=np.asarray(interval_open_commands,dtype=np.float32)
        if command.shape!=(len(p)-1,) or not np.isin(command,[-1.,1.]).all():
            raise ValueError('One binary absolute open-positive command per valid source interval required')
        target_q=slerp(q[left],q[right],alpha)
        target[:count,3:7]=relative_left(target_q,q[start]).astype(np.float32)
        target[:count,7]=command[np.asarray(meta['command_rows'][:count])]
        mask[:count]=1.
    target[count:]=target[count-1]
    if not np.isfinite(target).all():raise ValueError('Nonfinite resampled target; no repair or clipping')
    return {'actions':target,'supervision_mask':mask,'current_features':features[start].copy(),'layout':meta}

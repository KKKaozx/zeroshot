"""Current-row measured pose only; unavailable LT dimensions stay masked."""
import hashlib
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
SOURCES=['Bridge','Language Table','BC-Z','Fractal']


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def array_sha(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def raw_features(pose,sources):
    p=np.asarray(pose,dtype=np.float64);assert p.shape==(len(sources),7)
    complete=np.asarray([s!='Language Table' for s in sources]);assert set(sources)<=set(SOURCES)
    values=np.zeros((len(p),9),np.float64);mask=np.zeros_like(values)
    values[:,:2]=p[:,:2];mask[:,:2]=1
    values[complete,2]=p[complete,2];mask[complete,2:]=1
    q=p[complete,3:7];norm=np.linalg.norm(q,axis=1,keepdims=True)
    assert np.isfinite(q).all() and (norm>1e-8).all()
    q=q/norm;x,y,z,w=q.T
    # Concatenated first two rotation columns, same geometric encoding as the old9D branch.
    columns=np.stack([1-2*(y*y+z*z),2*(x*y+z*w),2*(x*z-y*w),
        2*(x*y-z*w),1-2*(x*x+z*z),2*(y*z+x*w)],axis=1)
    values[complete,3:]=columns
    assert np.isfinite(values).all()
    return values,mask


def current_rows(store,indices):
    indices=np.asarray(indices,dtype=np.int64);windows=store.windows[indices]
    pose_rows=store.offsets[windows[:,0]]+windows[:,1]
    sources=[store.episodes[int(i)]['source'] for i in windows[:,0]]
    return np.asarray(store.pose[pose_rows],dtype=np.float64),sources


def fit_training_stats(store):
    assert store.partition=='train','Validation must never fit state normalization'
    pose,sources=current_rows(store,np.arange(len(store)));values,mask=raw_features(pose,sources)
    count=mask[:,:3].sum(0);assert (count>0).all()
    mean=(values[:,:3]*mask[:,:3]).sum(0)/count
    variance=(((values[:,:3]-mean)**2)*mask[:,:3]).sum(0)/count
    std=np.maximum(np.sqrt(variance),1e-6)
    return {'xyz_mean':mean.tolist(),'xyz_std':std.tolist(),'observed_xyz_counts':count.astype(int).tolist(),
        'fit_partition':'train','fit_window_count':len(store),'fit_weighting':'one current observation per eligible training window, no validation',
        'raw_training_current_rows_sha256':array_sha(np.asarray(pose,dtype='<f8')),
        'source_window_counts':{s:sources.count(s) for s in SOURCES},'position_units':'unchanged recorded source units',
        'orientation':'unitxyzw to concatenated first two rotation columns; no normalization of rotation columns'}


def encode(pose,sources,stats,role):
    assert role in ['mask_control','real_pose']
    values,mask=raw_features(pose,sources)
    values[:,:3]=(values[:,:3]-np.asarray(stats['xyz_mean']))/np.asarray(stats['xyz_std'])
    values*=mask
    if role=='mask_control':values[:]=0
    out=np.concatenate([values,mask],axis=1).astype('<f4')
    assert out.shape==(len(pose),18) and np.isfinite(out).all()
    return out


def batch_features(store,indices,stats,role):return encode(*current_rows(store,indices),stats,role)


def validate_stats(store,expected):
    actual=fit_training_stats(store)
    for key in actual:
        if key in ['xyz_mean','xyz_std']:assert np.allclose(actual[key],expected[key],rtol=0,atol=1e-12),key
        else:assert actual[key]==expected[key],key
    return actual

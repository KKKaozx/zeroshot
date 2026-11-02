"""Current observation state only; train-only XYZ scaling, continuous rotation columns."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from dataset import UnifiedRobotDataset

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def audit_index(i):return i if i<32 else (i+40 if i<128 else i-96)
def state_features(observed,stats):
    v=np.asarray(observed,dtype=np.float64)
    xyz=(v[:,:3]-np.asarray(stats['xyz_mean']))/np.asarray(stats['xyz_std'])
    r,p,y=v[:,3],v[:,4],v[:,5]
    cr,cp,cy=np.cos(r),np.cos(p),np.cos(y);sr,sp,sy=np.sin(r),np.sin(p),np.sin(y)
    columns=np.stack([cy*cp,sy*cp,-sp,cy*sp*sr-sy*cr,sy*sp*sr+cy*cr,cp*sr],axis=1)
    out=np.concatenate([xyz,columns],axis=1).astype('<f4')
    assert out.shape==(len(v),9) and np.isfinite(out).all()
    return out

def add_zero_port(model):
    # Linear's temporary initialization is discarded; protect both global RNG streams.
    before=[torch.get_rng_state().clone(),torch.cuda.get_rng_state().clone()]
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        port=torch.nn.Linear(9,1024,bias=False).cuda()
        torch.nn.init.zeros_(port.weight)
    assert torch.equal(before[0],torch.get_rng_state()) and torch.equal(before[1],torch.cuda.get_rng_state())
    assert torch.count_nonzero(port.weight)==0
    model.state_projection=port

def clipping_parameters(model,trainables,arm):
    # A mathematically zero gradient still changes float32 reduction ordering.
    # Exclude only that identically zero block to preserve original clipping bits.
    if arm=='zero':
        assert torch.count_nonzero(model.state_projection.weight.grad)==0
        return [p for p in trainables if p is not model.state_projection.weight]
    return trainables

def load_current_features(protocol_path,rows,data_root,data_checks):
    protocol=json.loads(protocol_path.read_text(encoding='utf-8'));directory=protocol_path.parent
    assert rows==protocol['rows'] and protocol['state_features']['fit_rows']==list(range(128))
    assert sha(directory/'current_pose.npz')==protocol['current_pose_sha256']
    with np.load(directory/'current_pose.npz',allow_pickle=False) as z:
        assert z.files==['current_pose']
        expected=z['current_pose'].copy()
    actual=[];cache={}
    from bridge_input_contract import raw_identity
    for i,row in enumerate(rows):
        key=(row['shard'],row['record_index'])
        if key not in cache:cache[key]=UnifiedRobotDataset._load_tfrecord_example(str(Path(data_root)/row['shard']),row['record_index'])
        example=cache[key];t=row['start_index']
        assert raw_identity(example,t)==data_checks[audit_index(i)]['raw_identity']
        # Do not slice t+1 or future actions here. Those remain supervised targets only.
        states=np.asarray(example.features.feature['steps/observation/state'].float_list.value,dtype='<f4').reshape(-1,7)
        actual.append(states[t,:6])
    observed=np.stack(actual)
    assert np.array_equal(observed,expected), 'Current observation pose changed'
    stats=protocol['state_features'];training=observed[:128,:3].astype(float)
    assert np.array_equal(training.mean(axis=0),stats['xyz_mean'])
    assert np.array_equal(training.std(axis=0),stats['xyz_std'])
    features=state_features(observed,stats)
    return features,dict(raw_current_pose_exact=True,raw_windows=168,train_only_statistics=True,
        observation_index='start_index',target_or_future_state_used_as_context=False,
        features_sha256=hashlib.sha256(features.tobytes()).hexdigest(),
        feature_min=features.min(axis=0).tolist(),feature_max=features.max(axis=0).tolist(),
        zero_port_creation_preserves_rng=True)

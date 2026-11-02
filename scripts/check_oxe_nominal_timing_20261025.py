"""Focused interpolation/command phase tests; no robot model, real data or fit."""
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from oxe_nominal_timing import layout,chunk,slerp,relative_left,protocol,protocol_sha256
from cartesian_action_contract import encode_source_pose


def rejected(fn):
    try:fn()
    except ValueError:return
    raise AssertionError('Invalid interpolation input accepted')


def main():
    checks=[]
    b=layout('BC-Z',3,50)
    assert b['left_rows'][:3]==[5,7,9] and b['right_weights'][:3]==[0,0,0]
    assert b['command_rows'][:3]==[4,6,8]
    f=layout('Fractal',3,30)
    assert f['left_rows'][:2]==[3,4] and f['right_rows'][:2]==[4,5]
    assert np.allclose(f['right_weights'][:2],[.6,.2])
    assert f['left_rows'][4]==f['right_rows'][4]==6 and f['command_rows'][4]==5
    checks.append('integer-exact nominal row and right-closed command mapping')
    identity=np.array([0.,0,0,1]);q90=np.array([0.,0,np.sin(np.pi/4),np.cos(np.pi/4)])
    midpoint=slerp(identity[None],q90[None],np.array([.5]))[0]
    assert np.allclose(midpoint,[0,0,np.sin(np.pi/8),np.cos(np.pi/8)],rtol=0,atol=1e-12)
    assert np.allclose(midpoint,slerp(-identity[None],-q90[None],np.array([.5]))[0],rtol=0,atol=1e-12)
    assert np.allclose(slerp(identity[None],-identity[None],np.array([.5]))[0],identity)
    rejected(lambda:slerp(np.zeros((1,4)),q90[None],np.array([.5])))
    checks.append('shortest-path xyzw SLERP, sign equivalence and missing-orientation rejection')
    n=40;pos=np.c_[np.arange(n)/100,np.zeros(n),np.zeros(n)]
    angles=np.arange(n)/100
    quats=np.c_[np.zeros(n),np.sin(angles/2),np.zeros(n),np.cos(angles/2)]
    commands=np.where(np.arange(n-1)%2==0,1.,-1.).astype(np.float32)
    features=np.c_[np.ones(n),np.ones(n)].astype(np.float32);features[0]=0
    result=chunk('Bridge',pos,quats,commands,features,0)
    expected=np.stack([encode_source_pose(pos[0],quats[0],pos[i],quats[i],commands[i-1]) for i in range(1,17)])
    assert np.allclose(result['actions'],expected,rtol=0,atol=1e-7)
    assert not np.any(result['current_features'])
    checks.append('5Hz Bridge preserves old fixed-row target and causal unknown input')
    bcz=chunk('BC-Z',pos,quats,commands,features,0)
    expected=np.stack([encode_source_pose(pos[0],quats[0],pos[i],quats[i],commands[i-1]) for i in range(2,33,2)])
    assert np.allclose(bcz['actions'],expected,rtol=0,atol=1e-7)
    checks.append('10Hz BC-Z uses future reached even rows and matching preceding first command')
    frac=chunk('Fractal',pos,quats,commands,features,0)
    assert np.isclose(frac['actions'][0,0],.06,rtol=0,atol=1e-7)
    assert np.array_equal(frac['actions'][:5,7],commands[[0,1,1,2,2]])
    assert np.allclose(frac['actions'][4],encode_source_pose(pos[0],quats[0],pos[3],quats[3],commands[2]),atol=1e-7)
    checks.append('3Hz Fractal interpolates positions/rotations but never binary gripper values')
    planar=chunk('Language Table',pos[:,:2],None,None,np.zeros((n,2),np.float32),0)
    assert np.allclose(planar['actions'][:,:2],result['actions'][:,:2])
    assert planar['supervision_mask'][:,:2].sum()==32 and not np.any(planar['supervision_mask'][:,2:])
    rejected(lambda:chunk('Language Table',pos[:,:2],quats,None,np.zeros((n,2),np.float32),0))
    tail=chunk('BC-Z',pos[:10],quats[:10],commands[:9],features[:10],4)
    assert tail['layout']['valid_targets']==2 and not np.any(tail['supervision_mask'][2:])
    assert np.array_equal(tail['actions'][2:],np.repeat(tail['actions'][1:2],14,axis=0))
    assert layout('BC-Z',8,10)['valid_targets']==0
    rejected(lambda:chunk('BC-Z',pos[:10],quats[:10],commands[:9],features[:10],8))
    checks.append('planar missing dimensions, partial tails, zero-target windows and no extrapolation')
    rng=np.random.default_rng(20261025)
    for _ in range(32):
        a=rng.normal(size=4);b=rng.normal(size=4)
        reference=encode_source_pose([0,0,0],a,[0,0,0],b,1)[3:7]
        assert np.allclose(relative_left(b[None],a)[0],reference,rtol=0,atol=1e-6)
    checks.append('relative rotation remains target times inverse-reference in the recorded frame')
    r={'passed':True,'checks':checks,'protocol_sha256':protocol_sha256(),
        'nominal_not_measured_clock':not protocol()['measured_timestamps_used'],
        'real_source_data_read':False,'images_generated':False,'models_loaded':False,'training_updates':0}
    out=ROOT/'reports/oxe_nominal_timing_component_20261025';out.mkdir(exist_ok=False)
    (out/'protocol.json').write_text(json.dumps(protocol(),indent=2)+'\n')
    (out/'report.json').write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(r,indent=2))


if __name__=='__main__':main()

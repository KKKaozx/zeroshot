"""Freeze a small Panda split and verify physical collection/control on CPU.

Reuses the existing scene and motor controller; never loads a learned policy.
Test episodes are reserved and neither collected nor rolled out in this stage.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

os.environ.update(CUDA_VISIBLE_DEVICES='-1', USE_TF='0', HF_HUB_OFFLINE='1')
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'reports/panda_minimal_generalization_preparation_20261015'
CODE = ROOT/'code'
BASE = ROOT/'cluster_packages/bridge_growth128_v1'
sys.path.insert(0,str(CODE))
import pybullet_fixed_grasp as sim


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def write(p,v): p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')


def prepare():
    OUT.mkdir(exist_ok=False)
    dev={(1,1),(1,3),(3,1),(3,3)}
    test={(0,0),(0,4),(4,0),(4,4),(2,2)}
    grid=[-.04,-.02,0.,.02,.04]
    rows=[dict(episode=f'xy_{i}_{j}',split='development' if (i,j) in dev else 'test' if (i,j) in test else 'train',
               cube_xy_m=[.5+x,y],grid_index=[i,j]) for i,x in enumerate(grid) for j,y in enumerate(grid)]
    assert len({tuple(r['cube_xy_m']) for r in rows})==25
    assert {s:sum(r['split']==s for r in rows) for s in ['train','development','test']}==dict(train=16,development=4,test=5)
    protocol=dict(date='2026-10-15',purpose='Minimal same-camera/same-robot object-position generalization; not a Bridge root-cause proof',
        instruction=sim.INSTRUCTION,episodes=rows,split_unit='Whole physical episode; initial XY positions disjoint',
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),CODE/'pybullet_fixed_grasp.py',BASE/'dataset.py']},
        environment=dict(robot='franka_panda/panda.urdf',ee_link=11,finger_joints=[9,10],
            simulation_hz=240,ticks_per_command=24,image_shape=[224,224,3],
            camera='Exactly existing pybullet_fixed_grasp.rgb; fixed view/projection in hashed source',
            object='Same 4cm black cube, mass0.06kg; XY changed at scene initialization only'),
        phases=[dict(name='approach_xy',commands=12),dict(name='descend',commands=30),
            dict(name='close',commands=10),dict(name='lift',commands=35),dict(name='remain_above_table',commands=15)],
        starts=[0,12,24,36,44,52,68,86],chunk_size=16,
        targets='state[t+1:t+17] relative to state[t], paired with command_gripper[t:t+16]; fixed0.10m scale,xyzw',
        raw_logs='T+1 observations with timestamps, T requested world poses and gripper commands; no ambiguous final dummy command',
        policy_inputs='Current RGB, instruction, measured finger width; any EE-state port must be declared before model training; object coordinates prohibited',
        expert_only='Actual cube XY is available to the collection expert; never supplied as a learned-policy feature',
        physical_gate='Last10 commands cube >initial+0.10m, final bilateral finger contacts; same existing criterion; not precise-height holding',
        training_recipe=dict(status='Planned, not executed; resource preflight required',fresh_initialization=43,
            backbone='Frozen CLIP ViT-L/14',adapter='Existing eight-layer cross-attention',
            decoder='Existing multiscale x0 DDPM100; no regression replacement',
            optimize='Adapter, pose decoder and gripper together',batch_size=8,
            updates=8192,learning_rate='4096 at1e-4 then4096 at1e-5',
            sampling_seeds=[1101,1102,1103],checkpoint='Final fixed update only; no best-dev selection'),
        evaluation_plan=dict(offline='All preregistered windows; report position,rotation,gripper and event metrics against no-motion',
            closed_loop='Learned policy at each held-out XY, three sampling seeds; execute first4 predictions then observe; no expert handoff',
            development_success_screen='At least3/4 positions succeed in >=2/3 draws; report full counts, not a significance claim',
            baseline='Blind center expert trajectory as a stronger spatial control, plus no-motion policy',
            interpretation='If blind-center replay also succeeds everywhere, success alone cannot establish visual localization; do not claim it',
            test='Reserve5 XY positions until recipe/checkpoint frozen; do not collect or execute them in preparation'),
        stopping_rule='One fixed-budget model recipe after interface/resource gates. Failure does not authorize extra epochs, alternate seeds or dropping positions.',
        disk_limits=dict(preparation_bytes=160*1024**2,local_new_weights=False),
        current_stage='Freeze protocol before collecting new episodes; no model training')
    write(OUT/'protocol.json',protocol)
    print('PROTOCOL FROZEN',OUT/'protocol.json',flush=True)


def scene_at(xy):
    robot,cube=sim.scene()
    # Initial-condition construction, before the first recorded observation.
    sim.p.resetBasePositionAndOrientation(cube,[float(xy[0]),float(xy[1]),.66],[0,0,0,1])
    sim.p.resetBaseVelocity(cube,[0,0,0],[0,0,0])
    sim.p.setTimeStep(1/240)
    sim.command(robot,[.5,0,.85],1,ticks=120)
    return robot,cube


def expert_commands(xy):
    x,y=map(float,xy);commands=[]
    for a in np.linspace(0,1,13)[1:]:commands.append(([.5+a*(x-.5),a*y,.85],1.))
    commands += [([x,y,float(z)],1.) for z in np.linspace(.85,.655,31)[1:]]
    commands += [([x,y,.655],-1.)]*10
    commands += [([x,y,float(z)],-1.) for z in np.linspace(.655,.86,36)[1:]]
    commands += [([x,y,.86],-1.)]*15
    assert len(commands)==102
    return np.asarray([list(p)+list(sim.ORIENTATION)+[g] for p,g in commands])


def execute(row,requested,save_images):
    robot,cube=scene_at(row['cube_xy_m']);initial=sim.p.getBasePositionAndOrientation(cube)[0][2]
    positions,rotations,currents,heights,contacts,images=[],[],[],[],[],[]
    def observe():
        p,q,g=sim.observation(robot);positions.append(p);rotations.append(q);currents.append(g)
        heights.append(float(sim.p.getBasePositionAndOrientation(cube)[0][2]))
        contacts.append(sorted({c[3] for c in sim.p.getContactPoints(robot,cube) if c[3] in (9,10) and c[9]>.01}))
        if save_images:images.append(sim.rgb())
    observe()
    for command in requested:
        sim.command(robot,command[:3].tolist(),float(command[7]),command[3:7].tolist());observe()
    result=sim.outcome(robot,cube,float(initial),heights[1:],contacts[1:])
    assert sim.p.getNumConstraints()==0, 'Unexpected simulator attachment'
    result.update(episode=row['episode'],split=row['split'],requested_commands=len(requested),
        learned_policy_used=False,object_reset_after_initial_observation=False)
    data=dict(positions=np.asarray(positions),rotations=np.asarray(rotations),currents=np.asarray(currents,dtype=np.float32),
        commanded_poses=requested,commands=requested[:,7],object_heights=np.asarray(heights),
        contacts=np.asarray([[j in c for j in [9,10]] for c in contacts]),
        times_seconds=np.arange(len(positions))*.1)
    if save_images:data['images']=np.stack(images)
    return result,data


def collect():
    protocol=read(OUT/'protocol.json')
    assert all(sha(ROOT/n)==h for n,h in protocol['source_sha256'].items())
    assert not (OUT/'collection.json').exists()
    chosen=[r for r in protocol['episodes'] if r['split']!='test']
    # Fixed first four training placements: inspect interfaces before full collection.
    train=[r for r in chosen if r['split']=='train']
    first=[train[i] for i in [0,5,10,15]]
    ordered=first+[r for r in chosen if r not in first]
    folder=OUT/'episodes';folder.mkdir(exist_ok=False)
    summary=dict(protocol_sha256=sha(OUT/'protocol.json'),passed=False,trained=False,model_loaded=False,
        test_episodes_collected=False,expert=[],replays=[],error=None,
        versions={k:importlib.metadata.version(k) for k in ['pybullet','numpy']})
    sys.path.insert(0,str(BASE))
    from dataset import relative_pose_action,decode_relative_pose
    sim.p.connect(sim.p.DIRECT)
    try:
        for row in ordered:
            expert,data=execute(row,expert_commands(row['cube_xy_m']),True)
            summary['expert'].append(expert)
            assert expert['success'], 'Expert failed at '+row['episode']+'; do not train or silently discard'
            assert data['images'].shape==(103,224,224,3) and len(data['commands'])==102
            assert all(t+16<len(data['positions']) for t in protocol['starts'])
            path=folder/(row['episode']+'.npz');np.savez_compressed(path,**data)
            expert.update(npz_sha256=sha(path),bytes=path.stat().st_size)
            if row in first:
                native,again=execute(row,data['commanded_poses'],False)
                delta=float(np.abs(again['positions']-data['positions']).max())
                assert native['success'] and delta<1e-6,'Native controller replay mismatch'
                requested=[];roundtrip=0.
                for t in range(102):
                    action=relative_pose_action(data['positions'][t],data['rotations'][t],
                        data['positions'][t+1],data['rotations'][t+1],data['commands'][t])
                    p,q=decode_relative_pose(data['positions'][t],data['rotations'][t],action)
                    roundtrip=max(roundtrip,float(np.abs(p-data['positions'][t+1]).max()))
                    requested.append(np.r_[p,q,data['commands'][t]])
                reached,_=execute(row,np.asarray(requested),False)
                summary['replays'].append(dict(episode=row['episode'],native=native,
                    native_replay_max_position_difference_m=delta,reached_pose_replay=reached,
                    label_roundtrip_max_position_difference_m=roundtrip))
                assert reached['success'] and roundtrip<2e-7,'Measured future pose replay failed; control contract not established'
            assert sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())<protocol['disk_limits']['preparation_bytes']
            write(OUT/'collection.partial.json',summary)
            print('PHYSICAL COLLECTION',row['episode'],row['split'],'success',expert['success'],flush=True)
        assert len(summary['expert'])==20 and len(summary['replays'])==4
        assert all(sha(ROOT/n)==h for n,h in protocol['source_sha256'].items())
        summary['passed']=True
    except Exception as e:
        summary['error']=type(e).__name__+': '+str(e)
        raise
    finally:
        sim.p.disconnect();write(OUT/'collection.json',summary)
    print('PHYSICAL INPUT/CONTROL PREPARATION PASSED; no model trained',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['prepare','collect'])
    if parser.parse_args().stage=='prepare':prepare()
    else:collect()

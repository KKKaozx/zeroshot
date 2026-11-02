"""One new oracle reached-pose execution path; reuse old native success reference.

No learned model, new fit, primary40 task/init state, or runtime parameter sweep.
Known engineering demo targets deliberately use privileged future poses to test
the representation/executor boundary, not to measure learned policy competence.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
AUTHOR=ROOT/'training_cache/libero_author_8f1084e'
TASK='KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet_and_put_the_black_bowl_on_top_of_it'
DATA=ROOT.parent/'dataset/LIBERO/libero_90'/(TASK+'_demo.hdf5')
sys.path[:0]=[str(ROOT/'code'),str(ROOT/'diagnostics')]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ReachedPoseOracle:
    def __init__(self,positions,quaternions,actions,contract):
        import numpy as np
        self.positions=np.asarray(positions)
        self.quaternions=np.asarray(quaternions)
        self.actions=np.asarray(actions)
        self.contract=contract
        self.native_tick=0
        self.calls=[]
        if (self.positions.shape!=(197,3) or self.quaternions.shape!=(197,4)
                or self.actions.shape!=(197,7) or not np.isin(self.actions[:,6],[-1.,1.]).all()):
            raise ValueError('Only the predeclared197step engineering trace is accepted')
    def __call__(self,request):
        import numpy as np
        from cartesian_action_contract import encode_source_pose
        ticks=self.contract.ticks()
        rows=np.minimum(self.native_tick+ticks,len(self.positions))-1
        result=np.stack([encode_source_pose(request['reference_position_m'],request['reference_quaternion_xyzw'],
            self.positions[i],self.quaternions[i],float(-self.actions[i,6]),self.contract.position_scale_meters)
            for i in rows])
        self.calls.append({'start_native_tick':self.native_tick,'future_expert_rows':rows.tolist(),
            'first_target_native_tick':int(rows[0])+1,'first_gripper_open_positive':float(result[0,7]),
            'input_last_open_command':request['current_open_command']})
        return result


class ObservedEnvironment:
    def __init__(self,env,exact_gripper=None):self._wrapped=env;self.rows=[];self.exact_gripper=exact_gripper
    def __getattr__(self,name):return getattr(self._wrapped,name)
    def step(self,action):
        import numpy as np
        requested=np.asarray(action).copy()
        action=requested.copy()
        if self.exact_gripper is not None:action[6]=self.exact_gripper[min(len(self.rows),196),6]
        obs,reward,done,info=self._wrapped.step(action)
        row={'native_tick':len(self.rows)+1,'native_action':np.asarray(action).tolist(),
             'requested_native_action':requested.tolist(),
             'measured_position_m':np.asarray(obs['robot0_eef_pos']).tolist(),
             'measured_quaternion_xyzw':np.asarray(obs['robot0_eef_quat']).tolist(),
             'native_done':bool(done),'native_goal':bool(self._wrapped.check_success())}
        row['native_goal_components']=[{'predicate':s,'value':bool(self._wrapped.env._eval_predicate(s))}
                                      for s in self._wrapped.env.parsed_problem['goal_state']]
        if 'robot0_gripper_qpos' in obs:row['measured_finger_positions']=np.asarray(obs['robot0_gripper_qpos']).tolist()
        self.rows.append(row)
        return obs,reward,done,info


def main():
    p=argparse.ArgumentParser();p.add_argument('--attempt',default='attempt1')
    p.add_argument('--control',choices=['baseline5hz','exactgrip5hz','exactgrip20hz'],default='baseline5hz')
    args=p.parse_args()
    if not args.attempt.isalnum():raise ValueError('Fresh simple attempt label required')
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()[0]
    used,busy=map(int,gpu.split(','))
    if used>1024 or busy>20:raise RuntimeError('Existing GPU workload; no simulator started')
    output=ROOT/'reports/libero_oracle_execution_20261026'/args.attempt
    output.mkdir(parents=True,exist_ok=False)
    os.environ.update(CUDA_VISIBLE_DEVICES='-1',MUJOCO_GL='glfw',
        LIBERO_CONFIG_PATH=str(output/'config'),MPLCONFIGDIR=str(ROOT/'.tmp/libero_matplotlib'),
        NUMBA_CACHE_DIR=str(ROOT/'.tmp/libero_numba'))
    import h5py
    import numpy as np
    from PIL import Image,ImageDraw
    from scipy.spatial.transform import Rotation
    from libero_waypoint_execution import WaypointExecutionContract,execute_policy_chunk
    from libero_action_interface import camera_rgb_for_policy
    from run_libero_native_interface_20261020 import relocate_xml
    import yaml
    cfg=output/'config';cfg.mkdir()
    package=AUTHOR/'libero/libero'
    (cfg/'config.yaml').write_text(yaml.safe_dump({'benchmark_root':str(package),
        'bddl_files':str(package/'bddl_files'),'init_states':str(package/'init_files'),
        'assets':str(package/'assets'),'datasets':str(DATA.parent.parent)}))
    source=json.loads((ROOT/'reports/libero_native_replay_20261020/report.json').read_text())
    if not source['native_check_success_final'] or source['steps']!=197:
        raise ValueError('Existing successful native197step reference required; do not rerun it')
    with h5py.File(DATA,'r') as f:
        d=f['data/demo_0'];actions=d['actions'][:];positions=d['obs/ee_pos'][:];oris=d['obs/ee_ori'][:]
        initial=d['states'][0];xml=d.attrs['model_file'];settings=json.loads(f['data'].attrs['env_args'])
        if (not np.array_equal(d['obs/ee_states'][:,:3],positions)
                or not np.array_equal(d['obs/ee_states'][:,3:6],oris)):
            raise ValueError('Reached-pose observation fields differ')
        if settings['env_kwargs']['control_freq']!=20:raise ValueError('Native demo clock must be20Hz')
    if hashlib.sha256(actions.tobytes()).hexdigest()!=source['action_sha256']:
        raise ValueError('Expert command identity differs from existing reference')
    execution=json.loads((ROOT/'configs/libero_nominal_execution_candidate_20261025.json').read_text())
    kwargs=execution['constructor_kwargs'].copy()
    if args.control=='exactgrip20hz':
        kwargs['target_times_seconds']=[i/20 for i in range(1,17)]
        kwargs['timing_evidence']='Diagnostic oracle native20Hz reached-pose target update; never used with a trained checkpoint'
    for n in ('target_times_seconds','output_scales'):kwargs[n]=tuple(kwargs[n])
    contract=WaypointExecutionContract(**kwargs)
    prefix_ticks=1 if args.control=='exactgrip20hz' else 4
    if contract.ticks()[0]!=prefix_ticks or contract.execute_waypoints!=1:raise ValueError('Declared diagnostic prefix required')
    quats=Rotation.from_rotvec(oris).as_quat()
    oracle=ReachedPoseOracle(positions,quats,actions,contract)
    protocol={'task':'libero_90/'+TASK,'demo':'demo_0','expert_steps':197,'native_hz':20,
        'control':args.control,'oracle_hz':20/prefix_ticks,'native_step_budget':200,'tail_hold_steps':3,'settle_steps':0,
        'settle_reason':'Match original expert replay start/time; outer formal5settle protocol is not tested here',
        'observation_phase':'HDF5 obs rowj is after actionj; future native tickk maps to rowk-1',
        'pose':'Privileged future recorded reached world pose encoded relative to current measured EEF',
        'gripper':'Negative of recorded native command at the future endpoint, applied from interval start as original executor',
        'exact_native_gripper_control':args.control!='baseline5hz',
        'requested_vs_executed_gripper_logged':True,
        'clamped_tail':'After expert197 endpoint, hold its final pose/command through tick200; no adaptive extension',
        'state_corrections_after_initial':0,'model_instances':0,'training_updates':0,'formal40_tasks_accessed':0,
        'native_success_reference':'reports/libero_native_replay_20261020/report.json',
        'executor_sha256':sha(ROOT/'code/libero_waypoint_execution.py'),
        'geometry_sha256':sha(ROOT/'code/libero_action_interface.py'),
        'target_encoding_sha256':sha(ROOT/'code/cartesian_action_contract.py')}
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    report={'execution_complete':False,'protocol':protocol,'errors':[],'scope':
        'One excluded engineering oracle representation/executor test; not learned policy or cross-source calibration',
        'old_native_success_replayed':False,'new_path_only':True,'native_goal_any':False,'native_goal_final':False}
    env=None;chunks=[];frames=[];start=time.monotonic()
    try:
        import robosuite
        from libero.libero.envs import OffScreenRenderEnv
        env=OffScreenRenderEnv(bddl_file_name=str(package/'bddl_files/libero_90'/(TASK+'.bddl')),
            camera_heights=128,camera_widths=128,ignore_done=True,horizon=1000,control_freq=20)
        env.seed(20261026);env.reset()
        relocated,_=relocate_xml(xml,Path(robosuite.__file__).parent,package/'assets')
        env.reset_from_xml_string(relocated);env.sim.reset();obs=env.set_init_state(initial)
        if np.max(np.abs(env.get_sim_state()-initial))>=1e-12:raise ValueError('Expert initial state did not restore exactly')
        frames.append(Image.fromarray(camera_rgb_for_policy(obs['agentview_image'],'opengl')))
        observed=ObservedEnvironment(env,actions if args.control!='baseline5hz' else None);command=None;count=0
        while count<200:
            oracle.native_tick=count
            part=execute_policy_chunk(observed,oracle,obs,'engineering future reached-pose oracle',command,contract,native_step_budget=200-count)
            obs=part.pop('observation');count+=part['native_steps']
            command=-float(observed.rows[-1]['native_action'][6])
            part['last_executed_open_command']=command
            part['end_native_tick']=count
            part['measured_end_quaternion_xyzw']=np.asarray(obs['robot0_eef_quat']).tolist()
            targetq=part['trace'][-1]['target_world_quaternion_xyzw']
            part['deadline_rotation_error_deg']=float(np.degrees((Rotation.from_quat(targetq)*Rotation.from_quat(obs['robot0_eef_quat']).inv()).magnitude()))
            chunks.append(part)
            if count%4==0 or part['native_done']:
                frames.append(Image.fromarray(camera_rgb_for_policy(obs['agentview_image'],'opengl')))
            if len(chunks)%10==0:print('ORACLE_EXECUTION',count,'/200 ticks',flush=True)
            if part['native_done']:break
        rows=observed.rows
        indices=np.minimum(np.arange(1,len(rows)+1),197)-1
        pos_error=np.linalg.norm(np.asarray([x['measured_position_m'] for x in rows])-positions[indices],axis=1)
        orientation=Rotation.from_quat(np.asarray([x['measured_quaternion_xyzw'] for x in rows]))
        ori_error=np.degrees((Rotation.from_quat(quats[indices])*orientation.inv()).magnitude())
        deadlines=[x['trace'][-1]['target_position_error_m'] for x in chunks]
        report.update(execution_complete=True,native_steps=len(rows),oracle_calls=len(chunks),
            native_goal_any=any(x['native_goal'] for x in rows),native_goal_final=rows[-1]['native_goal'],
            native_goal_components_final=rows[-1]['native_goal_components'],
            deadline_position_mean_mm=float(np.mean(deadlines)*1000),deadline_position_max_mm=float(np.max(deadlines)*1000),
            path_position_mean_mm=float(np.mean(pos_error)*1000),path_position_max_mm=float(np.max(pos_error)*1000),
            path_rotation_mean_deg=float(np.mean(ori_error)),path_rotation_max_deg=float(np.max(ori_error)),
            saturated_native_ticks=sum(int(t['native_command_saturated']) for x in chunks for t in x['trace']),
            expert_gripper_switch_native_ticks=(np.flatnonzero(np.diff(actions[:,6]))+2).tolist(),
            oracle_gripper_switch_native_ticks=[rows[i]['native_tick'] for i in range(1,len(rows)) if rows[i]['native_action'][6]!=rows[i-1]['native_action'][6]],
            state_restore_count=1,state_corrections_after_initial=0,
            native_reference_final_success=True,native_reference_max_position_error_mm=source['max_eef_position_replay_error_m']*1000)
        (output/'native_trace.json').write_text(json.dumps(rows,indent=2)+'\n')
        (output/'chunks.json').write_text(json.dumps(chunks,indent=2)+'\n')
        (output/'oracle_calls.json').write_text(json.dumps(oracle.calls,indent=2)+'\n')
        frames[0].save(output/'execution.gif',save_all=True,append_images=frames[1:],duration=200,loop=0)
        chosen=np.linspace(0,len(frames)-1,12,dtype=int)
        sheet=Image.new('RGB',(4*136,3*150),'white');draw=ImageDraw.Draw(sheet)
        for n,i in enumerate(chosen):
            x=(n%4)*136;y=(n//4)*150
            sheet.paste(frames[i],(x,y+18));draw.text((x+2,y+2),f't={min(i*4,count)/20:.2f}s',fill='black')
        sheet.save(output/'contact_sheet.png')
    except Exception as e:
        report['errors'].append(type(e).__name__+': '+str(e));raise
    finally:
        if env is not None:env.close()
        report['elapsed_seconds']=time.monotonic()-start
        (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print('ORACLE_EXECUTION_REPORT',output/'report.json',flush=True)
        print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()

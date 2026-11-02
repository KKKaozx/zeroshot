"""New bounded command mode vs installed native controller, synthetic dynamics.

No source interpolation/physical-unit assumptions or model/task scoring. The
original reject mode stays the default. This does not rerun old replay tests.
"""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

os.environ['MUJOCO_GL']='glfw'
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'));sys.path.insert(0,str(ROOT/'scripts'))
from libero_action_interface import waypoint_to_native_osc
from libero_waypoint_execution import WaypointExecutionContract,execute_policy_chunk
from cartesian_action_contract import SOURCE_REPRESENTATION
from check_libero_action_interface_20261020 import controller


class FeedbackFixture:
    def __init__(self,done_at=None):
        self.position=np.array([.3,-.2,.1])
        self.rotation=Rotation.from_euler('z',90,degrees=True)
        self.ctl=controller(self.position,self.rotation)
        self.ctl.control_dim=6
        self.env=SimpleNamespace(control_freq=20)
        self.robots=[SimpleNamespace(controller=self.ctl)]
        self.actions=[];self.done_at=done_at
    def obs(self):
        return {'robot0_eef_pos':self.position.copy(),'robot0_eef_quat':self.rotation.as_quat(),
            'agentview_image':np.zeros((4,4,3),np.uint8)}
    def step(self,action):
        self.ctl.ee_pos=self.position.copy();self.ctl.ee_ori_mat=self.rotation.as_matrix()
        self.ctl.set_goal(action[:6]);self.actions.append(action.copy())
        self.position+=.5*(self.ctl.goal_pos-self.position)
        self.rotation=Rotation.from_matrix(self.ctl.goal_ori)
        return self.obs(),0.,self.done_at is not None and len(self.actions)>=self.done_at,{}
    def reset(self):raise AssertionError('No environment reset in executor')
    def check_success(self):raise AssertionError('No task scoring')


def rejected(call):
    try:call()
    except ValueError:return
    raise AssertionError('Invalid mode or strict oversized goal accepted')


def main():
    pos=np.array([.3,-.2,.1]);rot=Rotation.from_euler('z',90,degrees=True)
    relative=Rotation.from_rotvec([.9,0,0])
    waypoint=np.r_[[1.2,-.6,0],relative.as_quat(),-1.]
    kwargs=dict(action_representation=SOURCE_REPRESENTATION)
    rejected(lambda:waypoint_to_native_osc(waypoint,pos,rot.as_quat(),pos,rot.as_quat(),**kwargs))
    bounded=waypoint_to_native_osc(waypoint,pos,rot.as_quat(),pos,rot.as_quat(),
        saturation_policy='bounded_feedback',**kwargs)
    target=pos+np.array([.12,-.06,0]);target_rot=relative*rot
    assert np.allclose(bounded['target_world_position'],target,rtol=0,atol=1e-12)
    assert (Rotation.from_quat(bounded['target_world_quaternion_xyzw']).inv()*target_rot).magnitude()<1e-12
    assert not bounded['target_was_clipped'] and bounded['native_command_saturated']
    assert np.max(np.abs(bounded['native_action'][:6]))<=1
    raw=np.r_[target-pos,(target_rot*rot.inv()).as_rotvec()]/np.array([.05,.05,.05,.5,.5,.5])
    native_raw=controller(pos,rot);native_raw.set_goal(raw)
    native_bounded=controller(pos,rot);native_bounded.set_goal(bounded['native_action'][:6])
    assert np.allclose(native_raw.goal_pos,native_bounded.goal_pos,rtol=0,atol=1e-12)
    assert np.allclose(native_raw.goal_ori,native_bounded.goal_ori,rtol=0,atol=1e-12)
    small=waypoint.copy();small[:3]/=20;small[3:7]=Rotation.from_rotvec([.01,0,0]).as_quat()
    old=waypoint_to_native_osc(small,pos,rot.as_quat(),pos,rot.as_quat(),**kwargs)
    new=waypoint_to_native_osc(small,pos,rot.as_quat(),pos,rot.as_quat(),saturation_policy='bounded_feedback',**kwargs)
    assert np.array_equal(old['native_action'],new['native_action']) and not new['native_command_saturated']
    contract=WaypointExecutionContract((.4,.8),1,.1,
        'Synthetic fixed evaluation interval, not measured source timing',
        'Synthetic identical frame fixture; no source calibration inferred',
        action_representation=SOURCE_REPRESENTATION,saturation_policy='bounded_feedback')
    calls=[]
    def policy(request):calls.append(request);return np.stack([waypoint,waypoint])
    fixture=FeedbackFixture()
    result=execute_policy_chunk(fixture,policy,fixture.obs(),'synthetic command',1,contract,native_step_budget=8)
    assert result['native_steps']==8 and len(calls)==1
    assert result['saturation_policy']=='bounded_feedback'
    trace=result['trace']
    assert all(np.allclose(t['target_world_position_m'],target,rtol=0,atol=1e-12) for t in trace)
    distances=[np.linalg.norm(pos-target)]+[t['target_position_error_m'] for t in trace]
    assert all(b<a for a,b in zip(distances,distances[1:]))
    assert distances[-1]<.003
    assert trace[0]['native_command_saturated'] and not trace[-1]['native_command_saturated']
    assert all(np.max(np.abs(a[:6]))<=1 for a in fixture.actions)
    stopped=FeedbackFixture(done_at=2)
    early=execute_policy_chunk(stopped,policy,stopped.obs(),'synthetic command',1,contract,native_step_budget=8)
    assert early['native_steps']==2 and early['native_done']
    rejected(lambda:replace(contract,saturation_policy='implicit').ticks())
    report={'passed':True,'native_controller_clipping_matches_explicit_bounded_mode':True,
        'default_reject_mode_preserved':True,'small_commands_identical_to_default':True,
        'fixed_model_goal_preserved':True,'measured_feedback_recomputed_each_tick':True,
        'synthetic_start_position_error_m':float(distances[0]),'synthetic_final_position_error_m':float(distances[-1]),
        'native_done_respected':True,'physics_steps':0,'synthetic_feedback_ticks':10,
        'learned_model_loaded':False,'source_data_read':False,'source_interpolation_implemented':False,
        'actual_source_timing_or_frame_calibration_verified':False,'benchmark_performance_verdict':None,
        'source_sha256':{f:hashlib.sha256((ROOT/'code'/f).read_bytes()).hexdigest() for f in (
            'libero_action_interface.py','libero_waypoint_execution.py')}}
    out=ROOT/'reports/libero_bounded_feedback_20261025.json'
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()

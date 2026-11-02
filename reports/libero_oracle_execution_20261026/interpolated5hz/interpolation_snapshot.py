"""Diagnostic-only interpolation of the unchanged5Hz endpoint information.

Linear world translation and shortest-arc orientation interpolation start from
the measured pose at interval start. Original OSC conversion/bounds are reused.
The learned/oracle callback is called once, not at the20Hz subtarget cadence.
"""
from dataclasses import replace

import numpy as np
from scipy.spatial.transform import Rotation,Slerp

from cartesian_action_contract import SOURCE_REPRESENTATION,encode_source_pose,decode_source_pose
from libero_action_interface import camera_rgb_for_policy
from libero_waypoint_execution import _pose,execute_policy_chunk


def interpolated_world_targets(p0,q0,p1,q1,ticks=4):
    if ticks!=4:raise ValueError('This bounded diagnostic uses four native ticks only')
    alpha=np.arange(1,ticks+1,dtype=np.float64)/ticks
    positions=(1-alpha[:,None])*np.asarray(p0)+alpha[:,None]*np.asarray(p1)
    rotations=Slerp([0.,1.],Rotation.from_quat(np.stack([q0,q1])))(alpha).as_quat()
    return positions,rotations


def execute_interpolated_chunk(env,policy,observation,instruction,current_open_command,
                               contract,*,native_step_budget):
    if (contract.action_representation!=SOURCE_REPRESENTATION or contract.execute_waypoints!=1
            or contract.native_control_hz!=20 or contract.ticks()[0]!=4 or native_step_budget<4):
        raise ValueError('Only the existing source-Cartesian5Hz first-target contract is supported')
    p0,q0=_pose(observation)
    request={'rgb_upright':camera_rgb_for_policy(observation[contract.camera_key],contract.image_convention),
             'instruction':instruction,'reference_position_m':p0.copy(),
             'reference_quaternion_xyzw':q0.copy(),'current_open_command':current_open_command,
             'action_representation':contract.action_representation}
    waypoints=np.asarray(policy(request),dtype=np.float64)
    if (waypoints.shape!=(16,8) or not np.isfinite(waypoints).all()
            or not np.isin(waypoints[:,7],[-1.,1.]).all()
            or not np.allclose(np.linalg.norm(waypoints[:,3:7],axis=1),1,rtol=0,atol=1e-3)):
        raise ValueError('Complete original16x8 prediction required')
    p1,q1=decode_source_pose(waypoints[0],p0,q0,contract.position_scale_meters)
    positions,quaternions=interpolated_world_targets(p0,q0,p1,q1)
    # The local constant callback supplies one interpolated subtarget to the
    # unchanged native conversion. It never invokes the learned/oracle callback.
    sub=replace(contract,target_times_seconds=tuple(i/20 for i in range(1,17)),
        timing_evidence='Diagnostic20Hz subtargets derived only from one unchanged5Hz endpoint')
    obs=observation;last=current_open_command;trace=[];done=False
    for k,(position,quaternion) in enumerate(zip(positions,quaternions)):
        def subtarget(req):
            action=encode_source_pose(req['reference_position_m'],req['reference_quaternion_xyzw'],
                position,quaternion,float(waypoints[0,7]),contract.position_scale_meters)
            return np.repeat(action[None,:],16,axis=0)
        part=execute_policy_chunk(env,subtarget,obs,instruction,last,sub,native_step_budget=1)
        obs=part['observation'];last=part['last_open_command'];done=part['native_done']
        if part['native_steps']!=1:raise ValueError('One native tick required per subtarget')
        row=part['trace'][0]
        row.update(native_tick=k+1,simulated_seconds=(k+1)/20,
            interpolation_fraction=(k+1)/4,original_interval_target_world_position_m=p1.tolist(),
            original_interval_target_world_quaternion_xyzw=q1.tolist())
        trace.append(row)
        if done:break
    return {'observation':obs,'last_open_command':last,'native_done':bool(done),
        'native_steps':len(trace),'trace':trace,'policy_calls':1,'subtarget_constant_callbacks':len(trace),
        'reference_position_m':p0.tolist(),'reference_quaternion_xyzw':q0.tolist(),
        'predicted_waypoints':waypoints.tolist(),'target_times_seconds':list(contract.target_times_seconds),
        'original_interval_target_world_position_m':p1.tolist(),
        'original_interval_target_world_quaternion_xyzw':q1.tolist(),
        'original_endpoint_deadline_reached':len(trace)==4,
        'execution_rule':'Measured interval-start to unchanged future endpoint, linear translation and Slerp at native ticks',
        'timing_evidence':contract.timing_evidence,'tcp_mapping_evidence':contract.tcp_mapping_evidence,
        'saturation_policy':contract.saturation_policy,'source_timing_or_tcp_calibration_automatically_proven':False,
        'task_success_scored':False,'expert_high_frequency_pose_input_used':False}

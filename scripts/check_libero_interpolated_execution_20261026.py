"""Check interval frame anchoring, callback count and early-stop boundaries."""
from pathlib import Path
import json
import sys
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'diagnostics'),str(ROOT/'code')]
from libero_interpolated_execution_20261026 import execute_interpolated_chunk,interpolated_world_targets
from libero_waypoint_execution import WaypointExecutionContract
from cartesian_action_contract import encode_source_pose


def main():
    cfg=json.loads((ROOT/'configs/libero_nominal_execution_candidate_20261025.json').read_text())['constructor_kwargs']
    for n in ('target_times_seconds','output_scales'):cfg[n]=tuple(cfg[n])
    contract=WaypointExecutionContract(**cfg)
    p0=np.array([.1,.2,1.]);q0=Rotation.from_rotvec([2.8,.1,-.1]).as_quat()
    p1=p0+np.array([.02,-.01,.01]);q1=(Rotation.from_rotvec([.04,.03,-.02])*Rotation.from_quat(q0)).as_quat()
    class Fake:
        def __init__(self,stop=None):
            self.p=p0.copy();self.q=q0.copy();self.steps=0;self.stop=stop
            scales=np.array(contract.output_scales)
            self.env=SimpleNamespace(control_freq=20)
            self.robots=[SimpleNamespace(controller=SimpleNamespace(control_dim=6,use_delta=True,
                output_max=scales,output_min=-scales,input_max=1.,input_min=-1.))]
        def obs(self):
            return {'robot0_eef_pos':self.p.copy(),'robot0_eef_quat':self.q.copy(),
                    'agentview_image':np.zeros((128,128,3),dtype=np.uint8)}
        def step(self,a):
            self.steps+=1
            self.p+=np.asarray(a[:3])*np.array(contract.output_scales[:3])*.5
            self.q=(Rotation.from_rotvec(np.asarray(a[3:6])*np.array(contract.output_scales[3:])*.5)*Rotation.from_quat(self.q)).as_quat()
            return self.obs(),0.,self.steps==self.stop,{}
    calls=[]
    def policy(r):
        calls.append(r)
        y=encode_source_pose(r['reference_position_m'],r['reference_quaternion_xyzw'],p1,q1,1.,.1)
        return np.repeat(y[None,:],16,axis=0)
    f=Fake();result=execute_interpolated_chunk(f,policy,f.obs(),'fixed',None,contract,native_step_budget=4)
    expected,_=interpolated_world_targets(p0,q0,p1,q1)
    checks={'one_high_level_request_four_native_ticks':len(calls)==1 and f.steps==4,
        'world_subtargets_do_not_drift_with_measured_tracking_lag':np.max(np.abs(np.array([t['target_world_position_m'] for t in result['trace']])-expected))<1e-7,
        'same_original_endpoint_and_deadline':np.max(np.abs(np.array(result['trace'][-1]['target_world_position_m'])-p1))<1e-7 and result['original_endpoint_deadline_reached'],
        'original5hz_target_clock_retained':result['target_times_seconds']==list(contract.target_times_seconds)}
    f=Fake(stop=2);n=len(calls)
    result=execute_interpolated_chunk(f,policy,f.obs(),'fixed',None,contract,native_step_budget=4)
    checks['native_success_stops_without_finishing_remaining_ticks']=f.steps==2 and len(calls)==n+1 and not result['original_endpoint_deadline_reached']
    n=len(calls)
    try:
        execute_interpolated_chunk(f,policy,f.obs(),'fixed',None,contract,native_step_budget=3)
        checks['short_budget_rejected_before_high_level_request']=False
    except ValueError:
        checks['short_budget_rejected_before_high_level_request']=len(calls)==n
    checks={k:bool(v) for k,v in checks.items()}
    report={'passed':all(checks.values()),'checks':checks,'synthetic_native_steps':6,
        'real_physics_steps':0,'learning_model_forwards':0,
        'initial_report_serialization_issue':'NumPy bool converted to Python bool; no real execution occurred in that check'}
    out=ROOT/'reports/libero_oracle_interpolation_20261026';out.mkdir(exist_ok=True)
    (out/'contract_checks.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2));assert report['passed']


if __name__=='__main__':main()

"""Physical held-out controls without a learned policy or test episodes."""
import json
from pathlib import Path
import numpy as np
from prepare_panda_minimal_generalization_20261015 import OUT,read,write,sha,sim,scene_at,execute,expert_commands


def main():
    protocol=read(OUT/'protocol.json');assert read(OUT/'collection.json')['passed']
    destination=OUT/'physical_baselines.json';assert not destination.exists()
    result=dict(protocol_sha256=sha(OUT/'protocol.json'),source_sha256=sha(Path(__file__)),
        model_used=False,trained=False,test_executed=False,executed_commands=112,episodes=[])
    sim.p.connect(sim.p.DIRECT)
    try:
        for row in protocol['episodes']:
            if row['split']!='development':continue
            commands=expert_commands([.5,0])
            commands=np.concatenate([commands,np.repeat(commands[-1:],10,axis=0)])
            blind,_=execute(row,commands,False)
            robot,_=scene_at(row['cube_xy_m']);p,q,_=sim.observation(robot)
            stationary=np.repeat(np.r_[p,q,1.][None],112,axis=0)
            static,_=execute(row,stationary,False)
            result['episodes'].append(dict(episode=row['episode'],blind_center=blind,no_motion=static))
            print('BASELINE',row['episode'],'blind_center',blind['success'],'no_motion',static['success'],flush=True)
    finally:sim.p.disconnect()
    result['blind_center_successes']=sum(r['blind_center']['success'] for r in result['episodes'])
    result['no_motion_successes']=sum(r['no_motion']['success'] for r in result['episodes'])
    result['positions']=len(result['episodes']);write(destination,result)


if __name__=='__main__':main()

"""Explicit optional ONE engineering episode after an accepted head; no auto run."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
TASK='KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet_and_put_the_black_bowl_on_top_of_it'


def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True)
    p.add_argument('--head',type=Path,required=True); p.add_argument('--source',type=Path,required=True)
    p.add_argument('--run',action='store_true'); p.add_argument('--child',action='store_true')
    args=p.parse_args(); root=args.root.resolve(); out=root/'reports/dino229116_sidecar_libero_engineering_20261030/run1'
    if not args.child:
        missing=[str(x) for x in (args.head,args.source) if not x.is_file()]
        if not args.run:
            print(json.dumps({'missing_files':missing,'run_already_exists':out.exists(),'simulation_started':False},indent=2)); return
        if missing: raise ValueError('Required existing policy/head files missing: '+', '.join(missing))
        used,busy=map(int,subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu',
            '--format=csv,noheader,nounits'],text=True).splitlines()[0].split(','))
        if used>1024 or busy>20: raise RuntimeError('Existing GPU workload; no model or simulation started')
        out.mkdir(parents=True,exist_ok=False)
        with (out/'console.log').open('w',encoding='utf-8') as log:
            try:
                result=subprocess.run([sys.executable,'-B','-u',str(Path(__file__).resolve()),'--child',
                    '--root',str(root),'--head',str(args.head.resolve()),'--source',str(args.source.resolve())],
                    stdout=log,stderr=subprocess.STDOUT,timeout=600,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            except subprocess.TimeoutExpired:
                existing=json.loads((out/'report.json').read_text()) if (out/'report.json').is_file() else {}
                existing.update(execution_complete=False,stop_reason='600seconds_hard_limit',automatic_retry=False)
                (out/'report.json').write_text(json.dumps(existing,indent=2)+'\n')
                raise
        if result.returncode: raise RuntimeError('Trial stopped; inspect report/console; no automatic retry')
        print('ENGINEERING_COMPLETE',out/'report.json'); return
    started=time.monotonic(); env=None; frames=[]; chunks=[]
    report={'execution_complete':False,'policy_calls':0,'native_steps':0,'errors':[],
        'formal_scoring_admitted':False,'training_updates':0,'source_pose_updated':False,
        'physical_mapping_calibrated':False,'automatic_following_action':False}
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',
        MUJOCO_GL='glfw',LIBERO_CONFIG_PATH=str(out/'libero_config'),
        MPLCONFIGDIR=str(root/'.tmp/libero_matplotlib'),NUMBA_CACHE_DIR=str(root/'.tmp/libero_numba'))
    sys.path[:0]=[str(HERE),str(HERE/'code'),str(HERE/'vendor/bridge_growth128_v1'),str(root/'diagnostics')]
    try:
        import h5py
        import numpy as np
        import torch
        import yaml
        from PIL import Image
        from load_policy import load_policy
        from train_head import read,write
        from legacy_dino_gripper_execution import bind_panda_measured_observation
        from libero_waypoint_execution import WaypointExecutionContract,execute_policy_chunk
        from libero_action_interface import camera_rgb_for_policy
        from run_libero_native_interface_20261020 import relocate_xml
        torch.set_num_threads(2)
        execution=read(HERE/'engineering_execution.json')
        kwargs=execution['constructor_kwargs'].copy()
        for key in ('target_times_seconds','output_scales'): kwargs[key]=tuple(kwargs[key])
        contract=WaypointExecutionContract(**kwargs)
        if contract.ticks().tolist()!=list(range(4,65,4)) or contract.execute_waypoints!=1:
            raise ValueError('Keep the declared5Hz/20Hz prefix unchanged')
        policy=load_policy(root,args.head,source_path=args.source,
            clip_path=root.parent/'hf_cache/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41',
            dino_path=root/'baselines/dinov2_large_frozen_assets_20261023')
        report['policy_metadata']=policy.metadata
        package=root/'training_cache/libero_author_8f1084e/libero/libero'
        cfg=out/'libero_config'; cfg.mkdir()
        (cfg/'config.yaml').write_text(yaml.safe_dump({'benchmark_root':str(package),
            'bddl_files':str(package/'bddl_files'),'init_states':str(package/'init_files'),
            'assets':str(package/'assets'),'datasets':str(root.parent/'dataset/LIBERO')}))
        data=root.parent/'dataset/LIBERO/libero_90'/(TASK+'_demo.hdf5')
        with h5py.File(data,'r') as f:
            initial=f['data/demo_0/states'][0]; xml=f['data/demo_0'].attrs['model_file']
        import robosuite
        from libero.libero.envs import OffScreenRenderEnv
        env=OffScreenRenderEnv(bddl_file_name=str(package/'bddl_files/libero_90'/(TASK+'.bddl')),
            camera_heights=128,camera_widths=128,ignore_done=True,horizon=1000,control_freq=20)
        env.seed(20261026); env.reset()
        relocated,_=relocate_xml(xml,Path(robosuite.__file__).parent,package/'assets')
        env.reset_from_xml_string(relocated); env.sim.reset(); obs=env.set_init_state(initial)
        if np.max(np.abs(env.get_sim_state()-initial))>=1e-12: raise ValueError('Engineering initial state changed')
        report['native_goal_initial']=bool(env.check_success())
        if report['native_goal_initial']: raise ValueError('Initial state already satisfies goal')
        frames.append(Image.fromarray(camera_rgb_for_policy(obs['agentview_image'],'opengl')))
        command=None; stop='200ticks_or50calls'
        while report['native_steps']<200 and policy.calls<50:
            bound=bind_panda_measured_observation(policy,obs,fully_open_width_m=.08)
            part=execute_policy_chunk(env,bound,obs,
                'close the top drawer of the cabinet and put the black bowl on top of it',command,
                contract,native_step_budget=200-report['native_steps'])
            obs,command=part.pop('observation'),part['last_open_command']
            part['inference']=policy.last_inference.copy(); part['native_goal']=bool(env.check_success()); chunks.append(part)
            report.update(native_steps=sum(c['native_steps'] for c in chunks),policy_calls=policy.calls)
            write(out/'chunks.json',chunks)
            frames.append(Image.fromarray(camera_rgb_for_policy(obs['agentview_image'],'opengl')))
            if policy.calls%5==0: print('DINO_SIDECAR_ENGINEERING',policy.calls,'/50',flush=True)
            if part['native_goal'] or part['native_done']:
                stop='native_success' if part['native_goal'] else 'native_done'; break
        report.update(execution_complete=True,stop_reason=stop,native_goal_final=bool(env.check_success()))
    except BaseException as e:
        report['errors'].append(type(e).__name__+': '+str(e)); raise
    finally:
        report['elapsed_seconds']=time.monotonic()-started
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        if frames: frames[0].save(out/'execution.gif',save_all=True,append_images=frames[1:],duration=200,loop=0)
        if env is not None: env.close()


if __name__=='__main__': main()

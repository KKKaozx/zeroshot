"""One explicit whole-policy wiring trial, default read-only readiness check.

No offline-gate override is implicit. --allow-unadmitted-engineering permits a
failed head ONLY for this labelled200tick trial; it never alters gate receipts.
No training, gain/threshold sweep, expert future actions or automatic retries.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'code'),str(ROOT/'diagnostics')]
OUT=ROOT/'reports/dino_gripper_function_closeout_20261031/run1'
TASK='KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet_and_put_the_black_bowl_on_top_of_it'


def read(path): return json.loads(path.read_text(encoding='utf-8'))
def write(path,value): path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def paths(mode):
    if mode=='source235928':
        head=ROOT/'reports/dino_gripper_sidecar_235928/cluster/gripper.pt'; pair=None
        expected=read(ROOT/'reports/dino_gripper_sidecar_235928/terminal_report.json')['gripper_file_sha256']
    else:
        folder=ROOT/'reports/gripper_event_repair_236140/cluster'
        head=folder/mode/'gripper.pt'; pair=folder/'pair_report.json'; expected=None
        audit=ROOT/'reports/gripper_event_repair_236140/audit.json'
        if audit.is_file(): expected=read(audit).get('head_local_sha256',{}).get(mode)
    return head,pair,expected


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--head-mode',choices=('state_control','event_aware','source235928'),default='event_aware')
    p.add_argument('--allow-unadmitted-engineering',action='store_true')
    p.add_argument('--run',action='store_true'); p.add_argument('--child',action='store_true')
    args=p.parse_args()
    head,pair_path,head_sha=paths(args.head_mode)
    source=ROOT/'reports/bridge_backbone_pair_a40_229116/cluster/extracted/dino/final_trainable.pt'
    needed=[source,head]+([pair_path] if pair_path is not None else [])
    missing=[str(path) for path in needed if not path.is_file()]
    if not args.child:
        readiness={'head_mode':args.head_mode,'missing_files':missing,'run_already_exists':OUT.exists(),
            'explicit_unadmitted_engineering_requested':args.allow_unadmitted_engineering,
            'simulation_started':False,'formal_scoring_admitted':False,
            'limits':{'native_ticks':200,'policy_calls':50,'wall_seconds':600,'training_updates':0}}
        if not missing:
            from preserved_dino_composite import load_versioned_head
            try:
                _,meta=load_versioned_head(head,pair_report=read(pair_path) if pair_path else None,
                    expected_head_sha256=head_sha,allow_unadmitted_engineering=args.allow_unadmitted_engineering)
                readiness['head_admission']=meta; readiness['load_permitted']=True
            except ValueError as e:
                readiness.update(load_permitted=False,reason=str(e))
        if not args.run:
            print(json.dumps(readiness,indent=2)); return
        if missing or not readiness.get('load_permitted'):
            raise ValueError('Files/admission unresolved; no source model or simulator started: '+json.dumps(readiness))
        used,busy=map(int,subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu',
            '--format=csv,noheader,nounits'],text=True).splitlines()[0].split(','))
        if used>1024 or busy>20: raise RuntimeError('Existing GPU workload; no model/simulator started')
        OUT.mkdir(parents=True,exist_ok=False); write(OUT/'launch_contract.json',readiness)
        child_args=[sys.executable,'-B','-u',str(Path(__file__).resolve()),'--child','--head-mode',args.head_mode]
        if args.allow_unadmitted_engineering: child_args.append('--allow-unadmitted-engineering')
        with (OUT/'console.log').open('w',encoding='utf-8') as log:
            try:
                result=subprocess.run(child_args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=600,
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            except subprocess.TimeoutExpired:
                report=read(OUT/'report.json') if (OUT/'report.json').is_file() else {}
                report.update(execution_complete=False,stop_reason='600seconds_hard_limit',automatic_retry=False)
                write(OUT/'report.json',report); raise
        if result.returncode: raise RuntimeError('Engineering trial stopped; inspect console/report; no automatic retry')
        print('PRESERVED_DINO_CLOSEOUT_COMPLETE',OUT/'report.json'); return
    started=time.monotonic(); env=None; chunks=[]; frames=[]
    report={'execution_complete':False,'native_steps':0,'policy_calls':0,'head_mode':args.head_mode,
        'training_updates':0,'source_pose_updates':0,'formal_scoring_admitted':False,
        'errors':[],'automatic_following_action':False}
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',
        MUJOCO_GL='glfw',LIBERO_CONFIG_PATH=str(OUT/'libero_config'),
        MPLCONFIGDIR=str(ROOT/'.tmp/libero_matplotlib'),NUMBA_CACHE_DIR=str(ROOT/'.tmp/libero_numba'))
    try:
        import h5py
        import numpy as np
        import torch
        import yaml
        from PIL import Image
        from preserved_dino_composite import load_composite_policy
        from legacy_dino_gripper_execution import bind_panda_measured_observation
        from libero_waypoint_execution import WaypointExecutionContract,execute_policy_chunk
        from libero_action_interface import camera_rgb_for_policy
        from run_libero_native_interface_20261020 import relocate_xml
        torch.set_num_threads(2)
        base=ROOT/'deployment/dino_gripper_sidecar_v1'
        policy=load_composite_policy(base,source,head,
            clip_assets=ROOT.parent/'hf_cache/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41',
            dino_assets=ROOT/'baselines/dinov2_large_frozen_assets_20261023',
            mapping=read(base/'engineering_mapping.json'),pair_report=read(pair_path) if pair_path else None,
            expected_head_sha256=head_sha,allow_unadmitted_engineering=args.allow_unadmitted_engineering)
        report['policy_metadata']=policy.metadata
        kwargs=read(base/'engineering_execution.json')['constructor_kwargs'].copy()
        for key in ('target_times_seconds','output_scales'): kwargs[key]=tuple(kwargs[key])
        contract=WaypointExecutionContract(**kwargs)
        if contract.ticks().tolist()!=list(range(4,65,4)) or contract.execute_waypoints!=1:
            raise ValueError('Existing5Hz/20Hz execution schedule changed')
        package=ROOT/'training_cache/libero_author_8f1084e/libero/libero'
        cfg=OUT/'libero_config'; cfg.mkdir()
        (cfg/'config.yaml').write_text(yaml.safe_dump({'benchmark_root':str(package),
            'bddl_files':str(package/'bddl_files'),'init_states':str(package/'init_files'),
            'assets':str(package/'assets'),'datasets':str(ROOT.parent/'dataset/LIBERO')}))
        data=ROOT.parent/'dataset/LIBERO/libero_90'/(TASK+'_demo.hdf5')
        with h5py.File(data,'r') as f:
            initial=f['data/demo_0/states'][0]; xml=f['data/demo_0'].attrs['model_file']
        import robosuite
        from libero.libero.envs import OffScreenRenderEnv
        env=OffScreenRenderEnv(bddl_file_name=str(package/'bddl_files/libero_90'/(TASK+'.bddl')),
            camera_heights=128,camera_widths=128,ignore_done=True,horizon=1000,control_freq=20)
        env.seed(20261026); env.reset()
        xml,_=relocate_xml(xml,Path(robosuite.__file__).parent,package/'assets')
        env.reset_from_xml_string(xml); env.sim.reset(); obs=env.set_init_state(initial)
        if np.max(np.abs(env.get_sim_state()-initial))>=1e-12 or env.check_success():
            raise ValueError('Invalid engineering initial state')
        frames.append(Image.fromarray(camera_rgb_for_policy(obs['agentview_image'],'opengl')))
        command=None; stop='200ticks_or50calls'
        while report['native_steps']<200 and policy.calls<50:
            callback=bind_panda_measured_observation(policy,obs,fully_open_width_m=.08)
            part=execute_policy_chunk(env,callback,obs,
                'close the top drawer of the cabinet and put the black bowl on top of it',command,
                contract,native_step_budget=200-report['native_steps'])
            obs,command=part.pop('observation'),part['last_open_command']
            part['inference']=policy.last_inference.copy(); part['native_goal']=bool(env.check_success())
            chunks.append(part); report.update(native_steps=sum(c['native_steps'] for c in chunks),policy_calls=policy.calls)
            write(OUT/'chunks.json',chunks)
            frames.append(Image.fromarray(camera_rgb_for_policy(obs['agentview_image'],'opengl')))
            if policy.calls%5==0: print('PRESERVED_DINO_CLOSEOUT',policy.calls,'/50',flush=True)
            if part['native_goal'] or part['native_done']:
                stop='native_success' if part['native_goal'] else 'native_done'; break
        report.update(execution_complete=True,stop_reason=stop,native_goal_final=bool(env.check_success()),
            native_goal_components_final=[{'predicate':s,'value':bool(env.env._eval_predicate(s))}
                for s in env.env.parsed_problem['goal_state']])
    except BaseException as e:
        report['errors'].append(type(e).__name__+': '+str(e)); raise
    finally:
        report['elapsed_seconds']=time.monotonic()-started; write(OUT/'report.json',report)
        if frames:
            frames[0].save(OUT/'execution.gif',save_all=True,append_images=frames[1:],duration=200,loop=0)
            frames[0].save(OUT/'initial.png'); frames[-1].save(OUT/'final.png')
        if env is not None: env.close()


if __name__=='__main__': main()

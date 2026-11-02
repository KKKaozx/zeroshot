"""One separately authorized command8 engineering episode; default readiness.

Fixed excluded LIBERO90 task/demo0,200ticks/50calls,600s. No automatic retry,
expert actions, goal-conditioned scripted path or midtrajectory state reset.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,v):
    import numpy as np
    def serial(x):
        if isinstance(x,np.ndarray):return x.tolist()
        if isinstance(x,np.generic):return x.item()
        raise TypeError(type(x).__name__)
    Path(p).write_text(json.dumps(v,indent=2,default=serial,allow_nan=False)+'\n',encoding='utf-8')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('root','checkpoint','report-json','mapping','clip-assets','dino-assets','output'):
        ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--run',action='store_true');ap.add_argument('--child',action='store_true')
    ap.add_argument('--allow-unadmitted-engineering',action='store_true')
    args=ap.parse_args();root=args.root.resolve();out=args.output.resolve()
    if not out.is_relative_to(root/'reports'):raise ValueError('Fresh output under root/reports required')
    p=read(HERE/'protocol.json');task=p['local_engineering_rollout_if_separately_authorized']['task']
    data=root.parent/'dataset/LIBERO/libero_90'/(task+'_demo.hdf5')
    paths=[args.checkpoint,args.report_json,args.mapping,args.clip_assets/'config.json',args.dino_assets/'model.safetensors',data]
    if not args.child:
        readiness={'missing':[str(v) for v in paths if not v.is_file()],'output_exists':out.exists(),
            'simulation_started':False,'limits':{'episodes':1,'native_ticks':200,'policy_calls':50,'seconds':600},
            'target_supervised':False,'formal_scoring_admitted':False}
        if not args.run:print(json.dumps(readiness,indent=2));return
        if readiness['missing'] or out.exists():raise ValueError('Missing inputs/existing output; no retry')
        r=read(args.report_json)
        if not r['execution_complete']:raise ValueError('Completed source report required')
        if not r['source_engineering_review_gate_passed'] and not args.allow_unadmitted_engineering:
            raise ValueError('Source gate failed; explicit unadmitted engineering authorization required, gate remains false')
        used,busy=map(int,subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()[0].split(','))
        if used>1024 or busy>20:raise RuntimeError('GPU already busy; no run started')
        out.mkdir(parents=True,exist_ok=False);write(out/'launch_contract.json',readiness)
        cmd=[sys.executable,'-B','-u',str(Path(__file__).resolve()),*sys.argv[1:],'--child']
        with (out/'console.log').open('w',encoding='utf-8') as log:
            try:result=subprocess.run(cmd,cwd=root,stdout=log,stderr=subprocess.STDOUT,timeout=600,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            except subprocess.TimeoutExpired:
                r=read(out/'report.json') if (out/'report.json').is_file() else {}
                r.update(execution_complete=False,error='600s_hard_limit',automatic_retry=False);write(out/'report.json',r);raise
        if result.returncode:raise RuntimeError('Episode stopped; retain log, no automatic retry')
        print('COMMAND8_LOCAL_ENGINEERING_COMPLETE',out/'report.json');return
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',USE_TF='0',CUBLAS_WORKSPACE_CONFIG=':4096:8',
        MUJOCO_GL='glfw',LIBERO_CONFIG_PATH=str(out/'libero_config'),
        MPLCONFIGDIR=str(root/'.tmp/libero_matplotlib'),NUMBA_CACHE_DIR=str(root/'.tmp/libero_numba'))
    sys.path[:0]=[str(HERE),str(HERE/'code'),str(HERE/'vendor/code'),str(HERE/'vendor')]
    started=time.monotonic();env=None
    r={'execution_complete':False,'task':task,'target_supervised':False,'formal_scoring_admitted':False,
        'expert_actions_used':0,'state_restores':0,'midtrajectory_state_corrections':0,'new_training_updates':0,
        'automatic_following_action':False,'errors':[]}
    trace=[];calls=[]
    try:
        import h5py,numpy as np,torch,yaml
        from PIL import Image
        from online_policy import load_policy,IncrementExecutor
        from command8_runtime import sha
        source_report=read(args.report_json)
        if sha(args.checkpoint)!=source_report['policy_sha256']:raise ValueError('Checkpoint differs from received source report')
        torch.set_num_threads(1);torch.backends.cudnn.benchmark=False
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
        torch.use_deterministic_algorithms(True,warn_only=True)
        mapping=read(args.mapping)
        policy=load_policy(args.checkpoint,clip_assets=args.clip_assets,dino_assets=args.dino_assets,mapping=mapping)
        executor=IncrementExecutor(mapping,p)
        package=root/'training_cache/libero_author_8f1084e/libero/libero'
        config=out/'libero_config';config.mkdir()
        (config/'config.yaml').write_text(yaml.safe_dump({'benchmark_root':str(package),'bddl_files':str(package/'bddl_files'),
            'init_states':str(package/'init_files'),'assets':str(package/'assets'),'datasets':str(root.parent/'dataset/LIBERO')}),encoding='utf-8')
        import robosuite
        from robosuite import macros
        from libero.libero.envs import OffScreenRenderEnv
        sys.path.insert(0,str(root/'diagnostics'))
        from run_libero_native_interface_20261020 import relocate_xml
        if macros.IMAGE_CONVENTION!='opengl':raise ValueError('Expected OpenGL observations')
        with h5py.File(data,'r') as f:initial=f['data/demo_0/states'][0];xml=f['data/demo_0'].attrs['model_file']
        env=OffScreenRenderEnv(bddl_file_name=str(package/'bddl_files/libero_90'/(task+'.bddl')),
            camera_heights=128,camera_widths=128,ignore_done=True,horizon=1000,control_freq=20)
        env.seed(20261031);env.reset();xml,_=relocate_xml(xml,Path(robosuite.__file__).parent,package/'assets')
        env.reset_from_xml_string(xml);env.sim.reset();obs=env.set_init_state(initial)
        if np.max(np.abs(env.get_sim_state()-initial))>=1e-12 or env.check_success():raise ValueError('Initial state/goal differs')
        r['state_restores']=1;r['checkpoint_metadata']=policy.metadata
        frames=[Image.fromarray(obs['agentview_image'][::-1].copy())];current=None;success=False
        instruction='close the top drawer of the cabinet and put the black bowl on top of it'
        for call in range(50):
            result=executor.execute(env,policy,obs,instruction,current,native_step_budget=200-len(trace))
            obs=result.pop('observation');current=result['current_open_command']
            for row in result.pop('trace'):
                row['native_tick']=len(trace)+1;row['policy_call']=call+1;trace.append(row)
            result['inference']=policy.last_inference;calls.append(result)
            frames.append(Image.fromarray(obs['agentview_image'][::-1].copy()))
            success=result['success']
            if success or result['done'] or len(trace)>=200:break
        write(out/'trace.json',trace);write(out/'commands_and_goals.json',calls)
        frames[0].save(out/'execution.gif',save_all=True,append_images=frames[1:],duration=200,loop=0)
        frames[0].save(out/'initial.png');frames[-1].save(out/'final.png')
        r.update(execution_complete=True,native_success=success,native_ticks=len(trace),policy_calls=len(calls),
            goal_initializations=sum(x['goal_initialized_from_measured_once'] for x in calls),
            desired_goal_updates=sum(x['desired_goal_updates_this_call'] for x in calls),
            source_gate_passed=source_report['source_engineering_review_gate_passed'],
            scope='One excluded engineering task/initial state; no formal/unseen-task/benchmark success claim')
    except BaseException as e:r['errors'].append(type(e).__name__+': '+str(e));raise
    finally:
        r['elapsed_seconds']=time.monotonic()-started;write(out/'report.json',r)
        if trace:write(out/'trace.json',trace)
        if calls:write(out/'commands_and_goals.json',calls)
        if env is not None:env.close()


if __name__=='__main__':main()

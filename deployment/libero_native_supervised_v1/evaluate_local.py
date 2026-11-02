"""Explicit five held-out same-task rollouts,200ticks each,600s whole-process cap.

Default reports readiness only. No training, expert commands, reset corrections,
parameter sweeps, final40 tasks, or automatic retries.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'code'))


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,required=True);ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--run',action='store_true');ap.add_argument('--child',action='store_true')
    args=ap.parse_args();root=args.root.resolve();output=args.output.resolve();checkpoint=args.checkpoint.resolve()
    if not output.is_relative_to(root/'reports'):raise ValueError('A fresh output under project/reports required')
    from native_data import TASK,INSTRUCTION
    data=root.parent/'dataset/LIBERO/libero_90'/(TASK+'_demo.hdf5')
    dino=root/'baselines/dinov2_large_frozen_assets_20261023'
    missing=[str(p) for p in [data,checkpoint,dino/'model.safetensors'] if not p.is_file()]
    if not args.child:
        readiness={'missing':missing,'output_exists':output.exists(),'target_supervised':True,'zero_shot':False,
            'formal_matrix_admitted':False,'simulation_started':False,'limits':{'episodes':5,'native_ticks_each':200,'policy_calls_each':50,'seconds_total':600}}
        if not args.run:print(json.dumps(readiness,indent=2));return
        if missing or output.exists():raise ValueError('Missing files or existing output; no run started')
        used,busy=map(int,subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()[0].split(','))
        if used>1024 or busy>20:raise RuntimeError('Existing GPU workload; no model/simulation started')
        output.mkdir(parents=True,exist_ok=False);write(output/'launch_contract.json',readiness)
        cmd=[sys.executable,'-B','-u',str(Path(__file__).resolve()),'--child','--root',str(root),
            '--checkpoint',str(checkpoint),'--output',str(output)]
        with (output/'console.log').open('w',encoding='utf-8') as log:
            try:result=subprocess.run(cmd,cwd=root,stdout=log,stderr=subprocess.STDOUT,timeout=600,
                                      creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            except subprocess.TimeoutExpired:
                report=read(output/'report.json') if (output/'report.json').is_file() else {}
                report.update(execution_complete=False,stop_reason='600s_hard_limit',automatic_retry=False)
                write(output/'report.json',report);raise
        if result.returncode:raise RuntimeError('Evaluation stopped; inspect existing log; no automatic retry')
        print('LIBERO_NATIVE_SUPERVISED_EVALUATION_COMPLETE',output/'report.json');return
    started=time.monotonic();env=None
    report={'execution_complete':False,'target_supervised':True,'zero_shot':False,'formal_matrix_admitted':False,
        'task':TASK,'episodes':[],'training_updates':0,'expert_actions_read':False,'automatic_following_action':False,'errors':[]}
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',USE_TF='0',CUBLAS_WORKSPACE_CONFIG=':4096:8',
        MUJOCO_GL='glfw',LIBERO_CONFIG_PATH=str(output/'libero_config'),MPLCONFIGDIR=str(root/'.tmp/libero_matplotlib'),
        NUMBA_CACHE_DIR=str(root/'.tmp/libero_numba'))
    try:
        import h5py
        import numpy as np
        import torch
        import yaml
        from PIL import Image
        from native_policy import NativeActionPolicy
        from native_data import proprio_features,prepare_pixels,decode_actions
        from frozen_visual_backbone import FrozenDinoVision,dino_config
        sys.path.insert(0,str(root/'diagnostics'))
        from run_libero_native_interface_20261020 import relocate_xml
        torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
        payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if (payload.get('format')!='libero90-native-supervised-policy-v1' or payload.get('update')!=4096
                or payload.get('action_contract')!='native_osc_pose7_close_positive_20hz_v1'
                or not payload.get('target_supervised') or payload.get('zero_shot') or payload.get('task')!=TASK):
            raise ValueError('Exact completed target-supervised native7 checkpoint required')
        model=NativeActionPolicy(payload['model_config']);model.load_state_dict(payload['state'],strict=True)
        if any(not torch.isfinite(v).all() for v in model.state_dict().values()):raise ValueError('Nonfinite endpoint')
        model=model.cuda().requires_grad_(False).eval()
        vision=FrozenDinoVision(dino_config(dino)).cuda().requires_grad_(False).eval()
        language=payload['language'].cuda();language_mask=payload['attention_mask'].cuda();normalizer=payload['normalizer']
        report['checkpoint_sha256']=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        package=root/'training_cache/libero_author_8f1084e/libero/libero'
        cfg=output/'libero_config';cfg.mkdir()
        (cfg/'config.yaml').write_text(yaml.safe_dump({'benchmark_root':str(package),'bddl_files':str(package/'bddl_files'),
            'init_states':str(package/'init_files'),'assets':str(package/'assets'),'datasets':str(root.parent/'dataset/LIBERO')}),encoding='utf-8')
        import robosuite
        from robosuite import macros
        if macros.IMAGE_CONVENTION!='opengl':raise ValueError('Declared OpenGL observation convention required')
        from libero.libero.envs import OffScreenRenderEnv
        for demo_id in range(25,30):
            folder=output/f'demo_{demo_id}';folder.mkdir()
            # Only XML and the one initial simulator state. No later states,
            # images, commands or expert targets enter a learned rollout.
            with h5py.File(data,'r') as f:
                group=f[f'data/demo_{demo_id}'];initial=group['states'][0];xml=group.attrs['model_file']
            env=OffScreenRenderEnv(bddl_file_name=str(package/'bddl_files/libero_90'/(TASK+'.bddl')),
                camera_heights=128,camera_widths=128,ignore_done=True,horizon=1000,control_freq=20)
            env.seed(20261031+demo_id);env.reset()
            xml,_=relocate_xml(xml,Path(robosuite.__file__).parent,package/'assets')
            env.reset_from_xml_string(xml);env.sim.reset();obs=env.set_init_state(initial)
            controller=env.robots[0].controller
            if (np.max(np.abs(env.get_sim_state()-initial))>=1e-12 or env.check_success()
                    or env.env.control_freq!=20 or not controller.use_delta or controller.control_dim!=6
                    or not np.array_equal(controller.output_max,[.05,.05,.05,.5,.5,.5])
                    or not np.array_equal(controller.output_min,[-.05,-.05,-.05,-.5,-.5,-.5])
                    or not np.array_equal(np.broadcast_to(controller.input_max,(6,)),np.ones(6))
                    or not np.array_equal(np.broadcast_to(controller.input_min,(6,)),-np.ones(6))):
                raise ValueError('Initial state/goal/native controller differs')
            frames=[Image.fromarray(obs['agentview_image'][::-1].copy())];trace=[];calls=0;success=False;stop='200tick_budget'
            while len(trace)<200 and calls<50:
                proprio=np.concatenate((obs['robot0_joint_pos'],obs['robot0_gripper_qpos']))
                features=torch.from_numpy(proprio_features(proprio,normalizer)[None]).cuda()
                with torch.inference_mode(),torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                    pixels=prepare_pixels(obs['agentview_image']).cuda();visual=vision(pixel_values=pixels).last_hidden_state
                    context=model.context(visual,language,language_mask,features)
                    sample_seed=1101+demo_id*1000+calls;torch.manual_seed(sample_seed);torch.cuda.manual_seed_all(sample_seed)
                    normalized=model.sample(context)[0].cpu().numpy()
                commands=decode_actions(normalized,normalizer);calls+=1
                for action in commands[:min(4,200-len(trace))]:
                    obs,reward,done,info=env.step(action.copy());success=bool(env.check_success())
                    trace.append({'native_tick':len(trace)+1,'policy_call':calls,'sampling_seed':sample_seed,
                        'native_action':action.tolist(),'eef_position_m':np.asarray(obs['robot0_eef_pos']).tolist(),'native_success':success})
                    if success or done:stop='native_success' if success else 'native_done';break
                frames.append(Image.fromarray(obs['agentview_image'][::-1].copy()))
                if success or done:break
            write(folder/'trace.json',trace)
            frames[0].save(folder/'execution.gif',save_all=True,append_images=frames[1:],duration=200,loop=0)
            frames[0].save(folder/'initial.png');frames[-1].save(folder/'final.png')
            episode={'demo_id':demo_id,'success':success,'native_ticks':len(trace),'policy_calls':calls,'stop':stop,
                'state_restores':1,'state_corrections_during_execution':0,'expert_actions_used':0}
            report['episodes'].append(episode);write(output/'report.json',report)
            print('LIBERO_NATIVE_HELDOUT_EPISODE',json.dumps(episode),flush=True);env.close();env=None
        report.update(execution_complete=True,successes=sum(e['success'] for e in report['episodes']),
            trials=5,success_fraction=sum(e['success'] for e in report['episodes'])/5,
            scope='Five fixed initial states for one target-supervised task; not zero-shot/general benchmark success')
    except BaseException as e:report['errors'].append(type(e).__name__+': '+str(e));raise
    finally:
        report['elapsed_seconds']=time.monotonic()-started;write(output/'report.json',report)
        if env is not None:env.close()


if __name__=='__main__':main()

"""One bounded paired gripper-loss repair. No pose network or automatic rollout."""
import argparse
import json
import os
from pathlib import Path
import signal
import sys
import time

HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'dino_gripper_sidecar_v1'
sys.path[:0]=[str(HERE/'code'),str(BASE/'code'),str(BASE)]
from train_head import read,write,sha,predict,summarize


def admit(root):
    from train_head import admit as old_source_admit
    p=read(HERE/'protocol.json')
    if sha(BASE/'SHA256SUMS')!=p['frozen_base_manifest_sha256']:
        raise ValueError('Required immutable235928 base bundle differs')
    for line in (HERE/'SHA256SUMS').read_text().splitlines():
        h,name=line.split('  ',1)
        if sha(HERE/name)!=h: raise ValueError('Repair bundle changed: '+name)
    checked=old_source_admit(root)
    head=root/p['source_gripper_checkpoint']
    if not head.is_file() or sha(head)!=p['source_gripper_sha256']:
        raise ValueError('Exact existing235928 head required, never overwrite it')
    return {'passed':True,'source_pose_check':checked,'source_gripper_sha256':sha(head),
            'source_gripper_checkpoint':str(head),'pose_network_loaded':False,'training_updates':0}


def decide(arms,source):
    absolute={}
    for name,report in arms.items():
        details={}
        for role in ('train','development'):
            m=report['roles'][role]['candidate']; previous=source['roles'][role]['candidate']
            details[role]={'balanced_accuracy_protected':m['balanced_accuracy']>=max(.8,previous['balanced_accuracy']-.02),
                'opening_event_gate':m['transitions']['opening']['f1'] is not None and m['transitions']['opening']['f1']>=.5,
                'closing_event_gate':m['transitions']['closing']['f1'] is not None and m['transitions']['closing']['f1']>=.5}
        absolute[name]={'criteria':details,'passed':all(all(v.values()) for v in details.values())}
    event_better=all(arms['event_aware']['roles']['development']['candidate']['transitions'][k]['f1']>
                     arms['state_control']['roles']['development']['candidate']['transitions'][k]['f1']
                     for k in ('opening','closing'))
    selected=('event_aware' if absolute['event_aware']['passed'] and event_better else
              'state_control' if absolute['state_control']['passed'] else None)
    return {'absolute_guards':absolute,'candidate_beats_control_both_development_events':event_better,
        'selected_head_for_further_engineering':selected,'automatic_rollout':False,'formal_matrix_admitted':False}


def run_arm(mode,out,arrays,plan,source_payload,frequencies,source_report):
    import numpy as np
    import torch
    from dino_gripper_sidecar import GripperSidecar
    from gripper_event_objective import gripper_event_loss
    torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    head=GripperSidecar().cuda(); head.load_state_dict(source_payload['state'],strict=True)
    opt=torch.optim.Adam(head.parameters(),lr=1e-4)
    tensors={k:torch.from_numpy(v).cuda() for k,v in arrays.items() if k.startswith('train_')}
    history=[]
    def save(step):
        payload={'format':'gripper-event-repair-training-v1','additional_updates':step,'mode':mode,
            'head':{k:v.detach().cpu().clone() for k,v in head.state_dict().items()},'optimizer':opt.state_dict(),
            'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
            'history':history,'protocol':read(HERE/'protocol.json'),'plan_sha256':sha(BASE/'plan.npz')}
        tmp=out/'latest.pt.tmp'; torch.save(payload,tmp); tmp.replace(out/'latest.pt')
    out.mkdir()
    for i in range(2048):
        head.train(); ix=torch.from_numpy(plan['indices'][i]).cuda(); draw=torch.from_numpy(plan['draws'][i]).cuda()
        opt.zero_grad(set_to_none=True)
        logits=head(tensors['train_contexts'][ix],tensors['train_generated_poses'][draw,ix],tensors['train_current'][ix])
        event_total,parts=gripper_event_loss(logits,tensors['train_command_targets'][ix],frequencies)
        loss=parts['state'] if mode=='state_control' else event_total
        if not torch.isfinite(loss): raise FloatingPointError('Nonfinite head objective')
        loss.backward(); norm=torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True); opt.step()
        history.append({'update':i+1,'loss':float(loss.detach()),'state_loss':float(parts['state'].detach()),
            'event_loss':float(parts['event'].detach()),'gradient_norm':float(norm)})
        if (i+1)%512==0: save(i+1)
        if i<2 or (i+1)%256==0: print('GRIPPER_EVENT_UPDATE',mode,i+1,'/2048',history[-1],flush=True)
    predictions={role:predict(head,arrays,role,'cuda') for role in ('train','development')}
    report=summarize(predictions,arrays)
    report.update(execution_complete=True,mode=mode,additional_updates=2048,total_head_updates=4096,
        source_pose_updates=0,source_head_sha256=read(HERE/'protocol.json')['source_gripper_sha256'])
    write(out/'report.json',report)
    np.savez_compressed(out/'predictions.npz',**{role+'_logits':v for role,v in predictions.items()})
    torch.save({'format':'gripper-event-repair-head-v1','source_pose_sha256':source_payload['source_checkpoint_sha256'],
        'source_gripper_sha256':report['source_head_sha256'],'mode':mode,'additional_updates':2048,
        'total_head_updates':4096,'state':{k:v.detach().cpu().clone() for k,v in head.state_dict().items()}},out/'gripper.pt')
    del head,opt,tensors
    torch.cuda.empty_cache()
    return report


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,default=Path('/projects/Zeroshot'))
    ap.add_argument('--output',type=Path); ap.add_argument('--check-only',action='store_true')
    a=ap.parse_args(); root=a.root.resolve(); admission=admit(root)
    if a.check_only:
        print('GRIPPER_EVENT_ADMISSION',json.dumps(admission),flush=True); return
    if not os.environ.get('SLURM_JOB_ID') or a.output is None:
        raise ValueError('Only inside the user manually submitted bounded job')
    out=a.output.resolve()
    if out.parent!=(root/'runs').resolve() or not out.name.startswith('gripper-event-repair-'):
        raise ValueError('A fresh dedicated output directory is required')
    import shutil
    import numpy as np
    import torch
    from gripper_event_objective import fit_event_frequencies
    if str(torch.__version__)!='2.5.1+cu118' or not torch.cuda.is_available() or torch.cuda.get_device_name()!='NVIDIA A40':
        raise RuntimeError('Original Torch2.5.1+cu118 and one A40 required')
    if shutil.disk_usage(root).free<256*1024**2: raise RuntimeError('Need256MiB output reserve; no cleanup')
    out.mkdir(exist_ok=False); write(out/'admission.json',admission)
    started=time.monotonic()
    def stop(signum,frame): raise TimeoutError('Stop/time budget ended; retain last complete512-step checkpoint; no auto resume')
    signal.signal(signal.SIGUSR1,stop); signal.signal(signal.SIGTERM,stop); signal.signal(signal.SIGALRM,stop); signal.alarm(480)
    try:
        torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        with np.load(BASE/'inputs.npz',allow_pickle=False) as z: arrays={k:z[k].copy() for k in z.files}
        with np.load(BASE/'plan.npz',allow_pickle=False) as z: plan={k:z[k].copy() for k in z.files}
        frequencies=fit_event_frequencies(arrays['train_command_targets'])
        if frequencies!=read(HERE/'event_frequencies.json'): raise ValueError('Training event population changed')
        payload=torch.load(admission['source_gripper_checkpoint'],map_location='cpu',weights_only=False)
        if payload['updates']!=2048 or payload['format']!='dino-gripper-sidecar-v1': raise ValueError('Wrong source head')
        old=read(HERE/'source_report.json')
        arms={mode:run_arm(mode,out/mode,arrays,plan,payload,frequencies,old) for mode in ('state_control','event_aware')}
        result=decide(arms,old)
        p=read(HERE/'protocol.json')
        if sha(root/p['source_gripper_checkpoint'])!=p['source_gripper_sha256'] or sha(root/p['source_pose_checkpoint'])!=p['source_pose_sha256']:
            raise ValueError('Protected original pose/head changed')
        result.update(execution_complete=True,total_new_head_updates=4096,source_pose_updates=0,
            original_pose_and_head_files_unchanged=True,elapsed_seconds=time.monotonic()-started,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,automatic_following_action=False)
        write(out/'pair_report.json',result)
        from verify_pair import verify
        write(out/'verification.json',verify(out))
        import zipfile
        with zipfile.ZipFile(out/'small_results.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
            for name in ('pair_report.json','admission.json','verification.json'): z.write(out/name,name)
            for mode in arms:
                for name in ('report.json','predictions.npz','gripper.pt'): z.write(out/mode/name,mode+'/'+name)
        print('GRIPPER_EVENT_COMPLETE',json.dumps(result),flush=True)
    except BaseException as e:
        write(out/'stopped.json',{'error':type(e).__name__+': '+str(e),'automatic_retry':False}); raise
    finally: signal.alarm(0)


if __name__=='__main__': main()

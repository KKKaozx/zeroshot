"""One gripper-only budget;229116 pose policy is never loaded or optimized."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE / 'code'))


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,value): Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2),b''): h.update(block)
    return h.hexdigest()


def admit(root):
    from dino_gripper_sidecar import SOURCE_SHA
    for line in (HERE / 'SHA256SUMS').read_text().splitlines():
        expected,name=line.split('  ',1)
        if sha(HERE / name) != expected: raise ValueError('Package changed: '+name)
    p=read(HERE / 'protocol.json'); receipt=read(HERE / 'inputs_receipt.json')
    if p['source_checkpoint_sha256'] != SOURCE_SHA or sha(HERE / 'inputs.npz') != receipt['input_sha256']:
        raise ValueError('Source/input identity changed')
    source=root / p['source_checkpoint']
    if not source.is_file() or sha(source) != SOURCE_SHA:
        raise ValueError('Exact existing229116 DINO checkpoint required; never replace it with234100')
    return {'passed':True,'source_checkpoint':str(source),'source_sha256':SOURCE_SHA,
            'pose_model_loaded':False,'training_updates':0}


def predict(head,arrays,role,device):
    import numpy as np
    import torch
    head.eval(); outputs=[]
    with torch.inference_mode():
        for draw in range(3):
            parts=[]
            for lo in range(0,len(arrays[role+'_contexts']),128):
                ix=slice(lo,lo+128)
                logits=head(torch.from_numpy(arrays[role+'_contexts'][ix]).to(device),
                    torch.from_numpy(arrays[role+'_generated_poses'][draw,ix]).to(device),
                    torch.from_numpy(arrays[role+'_current'][ix]).to(device))
                parts.append(logits.cpu().numpy())
            outputs.append(np.concatenate(parts))
    return np.stack(outputs)


def summarize(logits,arrays):
    import numpy as np
    from dino_gripper_sidecar import gripper_metrics
    roles={}
    for role in ('train','development'):
        truth=arrays[role+'_command_targets']>0
        constant=np.broadcast_to(arrays[role+'_current'][:,0,None]>=0,truth.shape)
        baselines={name:gripper_metrics(np.broadcast_to(value,logits[role].shape),truth) for name,value in [
            ('always_open',np.ones_like(truth)),('always_closed',np.zeros_like(truth)),
            ('threshold_current_measurement_held',constant)]}
        candidate=gripper_metrics(logits[role]>=0,truth)
        roles[role]={'candidate':candidate,'baselines':baselines}
    dev=roles['development']; candidate=dev['candidate']
    passed=(candidate['balanced_accuracy'] is not None and candidate['balanced_accuracy']>=.8
        and candidate['balanced_accuracy']>max(v['balanced_accuracy'] for v in dev['baselines'].values())
        and all(candidate['transitions'][k]['f1'] is not None and candidate['transitions'][k]['f1']>=.5
                for k in ('opening','closing')))
    return {'roles':roles,'offline_gripper_admitted':bool(passed),
        'closed_loop_success_proven':False,'formal_matrix_admitted':False,
        'measurement_threshold_baseline_is_not_a_past_command':True}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,default=Path('/projects/Zeroshot'))
    ap.add_argument('--output',type=Path); ap.add_argument('--check-only',action='store_true')
    args=ap.parse_args(); root=args.root.resolve(); admission=admit(root)
    if args.check_only:
        print('DINO_GRIPPER_ADMISSION',json.dumps(admission),flush=True); return
    if not os.environ.get('SLURM_JOB_ID') or args.output is None:
        raise ValueError('Training requires the user manually submitted allocated job')
    out=args.output.resolve()
    if out.parent != (root / 'runs').resolve() or not out.name.startswith('dino-gripper-sidecar-'):
        raise ValueError('A fresh dedicated run directory is required')
    import numpy as np
    import torch
    from dino_gripper_sidecar import GripperSidecar
    if str(torch.__version__)!='2.5.1+cu118' or not torch.cuda.is_available() or torch.cuda.get_device_name()!='NVIDIA A40':
        raise RuntimeError('Expected project Torch2.5.1+cu118 on one A40')
    used=int(subprocess.check_output(['du','-sB1',str(root)],text=True).split()[0])
    if used+256*1024**2>=200_000_000_000 or shutil.disk_usage(root).free<256*1024**2:
        raise RuntimeError('Need256MiB output/atomic-save reserve; no automatic cleanup; du is only an estimate')
    out.mkdir(exist_ok=False); write(out / 'admission.json',admission)
    started=time.monotonic(); stopped=False
    def request_stop(signum,frame):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGUSR1,request_stop); signal.signal(signal.SIGTERM,request_stop)
    def time_cap(signum,frame):
        raise TimeoutError('480s process cap reached; retain latest completed checkpoint; no auto retry')
    signal.signal(signal.SIGALRM,time_cap); signal.alarm(480)
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    head=GripperSidecar().cuda()
    optimizer=torch.optim.Adam(head.parameters(),lr=1e-4)
    with np.load(HERE / 'inputs.npz',allow_pickle=False) as z: arrays={k:z[k].copy() for k in z.files}
    with np.load(HERE / 'plan.npz',allow_pickle=False) as z: indices,draws=z['indices'],z['draws']
    if indices.shape!=(2048,64) or draws.shape!=(2048,64) or indices.max()>=1836 or draws.max()>2:
        raise ValueError('Fixed2048x64 training-only plan required')
    tensors={k:torch.from_numpy(v).cuda() for k,v in arrays.items() if k.startswith('train_')}
    history=[]; torch.cuda.reset_peak_memory_stats()
    def save(update):
        payload={'format':'dino-gripper-sidecar-training-state-v1','update':update,
            'head':{k:v.detach().cpu().clone() for k,v in head.state_dict().items()},
            'optimizer':optimizer.state_dict(),'torch_cpu_rng':torch.get_rng_state(),
            'torch_cuda_rng':torch.cuda.get_rng_state_all(),'numpy_rng':np.random.get_state(),
            'protocol':read(HERE / 'protocol.json'),'plan_sha256':sha(HERE / 'plan.npz'),'history':history}
        temporary=out / 'latest.pt.tmp'; torch.save(payload,temporary); temporary.replace(out / 'latest.pt')
    try:
        for i in range(2048):
            if stopped or time.monotonic()-started>=480:
                save(i); raise TimeoutError('Fixed480s process budget/stop signal; no automatic continuation')
            head.train(); ix=torch.from_numpy(indices[i]).cuda(); ds=torch.from_numpy(draws[i]).cuda()
            optimizer.zero_grad(set_to_none=True)
            logits=head(tensors['train_contexts'][ix],tensors['train_generated_poses'][ds,ix],tensors['train_current'][ix])
            target=(tensors['train_command_targets'][ix]>0).float()
            loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,target)
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite head loss')
            loss.backward(); norm=torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True)
            optimizer.step(); history.append({'update':i+1,'loss':float(loss.detach()),'gradient_norm':float(norm)})
            if (i+1)%512==0: save(i+1)
            if i<2 or (i+1)%256==0: print('DINO_GRIPPER_UPDATE',i+1,'/2048 loss',history[-1]['loss'],flush=True)
        logits={role:predict(head,arrays,role,'cuda') for role in ('train','development')}
        result=summarize(logits,arrays)
        if sha(admission['source_checkpoint'])!=admission['source_sha256']:
            raise RuntimeError('Source pose checkpoint changed during head-only job')
        payload={'format':'dino-gripper-sidecar-v1','source_checkpoint_sha256':admission['source_sha256'],
            'current_input_semantics':'bridge_measured_affine_unbounded_v1','updates':2048,
            'state':{k:v.detach().cpu().clone() for k,v in head.state_dict().items()},
            'offline_gripper_admitted':result['offline_gripper_admitted'],'protocol':read(HERE / 'protocol.json')}
        torch.save(payload,out / 'gripper.pt')
        restored=torch.load(out / 'gripper.pt',map_location='cpu',weights_only=False)
        if any(not torch.equal(v,restored['state'][k]) for k,v in payload['state'].items()):
            raise RuntimeError('Head save/readback differs')
        np.savez_compressed(out / 'predictions.npz',**{role+'_logits':v for role,v in logits.items()})
        result.update(execution_complete=True,updates=2048,source_pose_checkpoint_unchanged=True,
            source_pose_network_loaded=False,source_pose_optimizer_updates=0,
            equal_input_pose_preservation='Separate head only; native sampler composition has exact CPU regression check; not closed-loop observation invariance',
            elapsed_seconds=time.monotonic()-started,peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,
            gripper_file_sha256=sha(out / 'gripper.pt'),automatic_following_action=False)
        write(out / 'report.json',result)
        from verify_head import verify
        write(out / 'verification.json',verify(out))
        import zipfile
        with zipfile.ZipFile(out / 'small_results.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
            for name in ('report.json','predictions.npz','gripper.pt','verification.json','admission.json'): z.write(out / name,name)
        print('DINO_GRIPPER_COMPLETE',json.dumps(result),flush=True)
    except BaseException as e:
        write(out / 'stopped.json',{'error':type(e).__name__+': '+str(e),'automatic_retry':False})
        raise
    finally:
        signal.alarm(0)


if __name__=='__main__': main()

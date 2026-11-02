"""One explicitly target-supervised native-action control; no auto evaluation."""
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

HERE=Path(__file__).resolve().parent


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for data in iter(lambda:f.read(4*1024**2),b''):h.update(data)
    return h.hexdigest()
def state_copy(model):return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
def state_sha(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def seed(value):
    import numpy as np
    import torch
    np.random.seed(value);torch.manual_seed(value);torch.cuda.manual_seed_all(value)


def runtime_dependencies():
    import numpy
    import torch
    import PIL
    from transformers import CLIPTextModel,CLIPTextConfig,Dinov2Model
    from safetensors import safe_open
    from native_policy import NativeActionPolicy
    from native_data import causal_chunks,prepare_pixels
    from frozen_encoders import check_assets
    import verify
    return {'numpy':numpy.__version__,'torch':str(torch.__version__),'all_training_imports_checked_on_cpu':True}


def admit(root,*,assets=True,dino_assets=None,clip_assets=None):
    import numpy as np
    deps=runtime_dependencies()
    for line in (HERE/'SHA256SUMS').read_text().splitlines():
        digest,name=line.split('  ',1)
        if sha(HERE/name)!=digest:raise ValueError('Package changed: '+name)
    p=read(HERE/'protocol.json');binding=read(HERE/'inputs_receipt.json')
    for name,key in [('inputs.npz','inputs_sha256'),('plan.npz','plan_sha256'),('normalizer.json','normalizer_sha256')]:
        if sha(HERE/name)!=binding[key]:raise ValueError('Bound dataset changed: '+name)
    with np.load(HERE/'inputs.npz',allow_pickle=False) as z:
        if (len(z['images'])!=1888 or z['images'].shape[1:]!=(128,128,3) or z['proprio'].shape!=(1888,9)
                or set(np.unique(z['demo_id'][z['split']==0]))!=set(range(20))
                or set(np.unique(z['demo_id'][z['split']==1]))!=set(range(20,25))):
            raise ValueError('Fixed train/development episode partition differs')
        if (not np.isfinite(z['proprio']).all() or not np.isfinite(z['targets']).all()
                or not np.isin(z['mask'],[0.,1.]).all() or not np.all(z['mask'].sum(axis=(1,2))>0)):
            raise ValueError('Invalid causal inputs/valid-target masks')
    storage={'checked':False,'scope':'Cluster project du+filesystem space; Windows preparation has no cluster quota view'}
    if os.name=='posix':
        used=int(subprocess.check_output(['du','-sB1',str(root)],text=True).split()[0]);reserve=8*1024**3
        free=shutil.disk_usage(root).free
        storage.update(checked=True,project_bytes_estimate=used,filesystem_free_bytes=free,reserve_bytes=reserve)
        if used+reserve>=200_000_000_000 or free<reserve:raise RuntimeError('Need8GiB output reserve before submission; du is only an estimate; no automatic cleanup')
    dino=Path(dino_assets) if dino_assets is not None else root/p['dino_assets']
    clip=Path(clip_assets) if clip_assets is not None else root/p['clip_assets']
    weights=None
    if assets:
        from frozen_encoders import check_assets
        weights=check_assets(dino,clip,binding)
    return {'passed':True,'runtime_dependencies':deps,'assets_verified':assets,
        'storage':storage,
        'dino_assets':str(dino),'clip_assets':str(clip),'target_supervised':True,'zero_shot':False,
        'formal_matrix_admitted':False,'gpu_used_during_admission':False,'training_updates':0},weights


def predictions(model,visual,language,mask,state,ids):
    import torch
    import numpy as np
    model.eval();draws=[]
    with torch.no_grad(),torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        for value in (1101,1102,1103):
            seed(value);parts=[]
            for start in range(0,len(ids),16):
                ix=torch.as_tensor(ids[start:start+16],device='cuda');n=len(ix)
                ctx=model.context(visual[ix],language.expand(n,-1,-1),mask.expand(n,-1),state[ix])
                parts.append(model.sample(ctx).cpu().numpy())
            draws.append(np.concatenate(parts))
    return np.stack(draws)


def summarize(prediction,raw_target,valid,previous,normalizer):
    import numpy as np
    from native_data import decode_actions
    decoded=decode_actions(prediction,normalizer);truth=raw_target[None];m=valid[None]
    denom=m.sum()*3*6
    def motion_mse(value):return float((((value[...,:6]-truth[...,:6])**2)*m).sum()/denom)
    mean=np.broadcast_to(np.asarray(normalizer['train_action_mean'])[None,None,None,:],decoded.shape)
    held=np.broadcast_to(previous[None,:,None,:],decoded.shape)
    return {'motion6_mse_native_units':motion_mse(decoded),'train_mean_motion6_mse':motion_mse(mean),
        'previous_command_held_motion6_mse':motion_mse(held),'draws':3,'windows':len(raw_target),
        'constant_native_dimensions':np.flatnonzero(np.asarray(normalizer['action_scale'])==0).tolist(),
        'scope':'Offline command error; no task-success admission or checkpoint selection'}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path('/projects/Zeroshot'))
    ap.add_argument('--output',type=Path);ap.add_argument('--check-only',action='store_true')
    ap.add_argument('--dino-assets',type=Path);ap.add_argument('--clip-assets',type=Path);args=ap.parse_args()
    root=args.root.resolve();admission,text_state=admit(root,dino_assets=args.dino_assets,clip_assets=args.clip_assets)
    if args.check_only:print('LIBERO_NATIVE_SUPERVISED_ADMISSION',json.dumps(admission),flush=True);return
    if not os.environ.get('SLURM_JOB_ID') or args.output is None:raise ValueError('Manual allocated submission required')
    out=args.output.resolve()
    if out.parent!=(root/'runs').resolve() or not out.name.startswith('libero-native-supervised-'):raise ValueError('Fresh dedicated output required')
    import torch
    import numpy as np
    from native_policy import NativeActionPolicy
    from frozen_encoders import encode
    if str(torch.__version__)!='2.5.1+cu118' or not torch.cuda.is_available() or torch.cuda.get_device_name()!='NVIDIA A40':
        raise RuntimeError('Expected project Torch2.5.1+cu118/A40')
    out.mkdir(exist_ok=False);write(out/'admission.json',admission);started=time.monotonic();stopped=False
    def stop(signum,frame):
        nonlocal stopped
        stopped=True
    def hard_stop(signum,frame):raise TimeoutError('3300s total encoding/training/verification cap')
    signal.signal(signal.SIGUSR1,stop);signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,hard_stop);signal.alarm(3300)
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    torch.use_deterministic_algorithms(True,warn_only=True)
    p=read(HERE/'protocol.json');normalizer=read(HERE/'normalizer.json')
    with np.load(HERE/'inputs.npz',allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
    with np.load(HERE/'plan.npz',allow_pickle=False) as z:batches=z['batches'].copy()
    if batches.shape!=(4096,16) or not np.all(a['split'][batches]==0):raise ValueError('Fixed4096x16 training-only plan required')
    report={'execution_complete':False,'target_supervised':True,'zero_shot':False,'formal_matrix_admitted':False,
        'optimizer_updates':0,'encoder_updates':0,'original_oxe_weights_loaded':False,
        'task':p['task'],'train_demos':list(range(20)),'development_demos':list(range(20,25)),
        'automatic_rollout':False,'automatic_following_action':False,'errors':[]}
    model=None;optimizer=None;history=[];completed=0
    def save():
        temporary=out/'latest.pt.tmp'
        torch.save({'format':'libero90-native-supervised-training-state-v1','update':completed,
            'state':state_copy(model),'optimizer':optimizer.state_dict(),'history':history,
            'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'numpy_rng':np.random.get_state(),
            'protocol':p,'plan_sha256':sha(HERE/'plan.npz')},temporary)
        temporary.replace(out/'latest.pt')
    try:
        torch.cuda.reset_peak_memory_stats()
        def progress(done,total):
            if stopped:raise TimeoutError('Stop signal during encoding')
            if done%200==0 or done==total:print('LIBERO_NATIVE_ENCODING',done,'/',total,flush=True)
        visual,language=encode(a['images'],a['input_ids'],a['attention_mask'],admission['dino_assets'],text_state,progress=progress)
        del text_state
        report['new_frames_encoded']=len(visual);report['encoding_seconds']=time.monotonic()-started
        state=torch.from_numpy(a['proprio']).cuda();target=torch.from_numpy(a['targets']).cuda();valid=torch.from_numpy(a['mask']).cuda()
        language_mask=torch.from_numpy(a['attention_mask']).cuda()
        seed(42);model=NativeActionPolicy(p['model']).cuda();optimizer=torch.optim.Adam(model.parameters(),lr=1e-4,weight_decay=0)
        initial=state_sha(state_copy(model))
        with (out/'updates.jsonl').open('x') as log:
            for i in range(4096):
                if stopped:
                    if completed:save()
                    raise TimeoutError('Stop signal at update boundary; no automatic continuation')
                model.train();seed(420001+i);ix=torch.as_tensor(batches[i],device='cuda')
                ctx=model.context(visual[ix],language.expand(16,-1,-1),language_mask.expand(16,-1),state[ix])
                loss=model.loss(target[ix],valid[ix],ctx)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
                optimizer.zero_grad(set_to_none=True);loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step();completed=i+1
                row={'update':completed,'loss':float(loss.detach()),'gradient_norm':float(norm),'lr':1e-4};history.append(row);log.write(json.dumps(row)+'\n')
                if completed%512==0:save()
                if i<2 or completed%256==0:log.flush();print('LIBERO_NATIVE_UPDATE',json.dumps(row),flush=True)
        roles={'train':np.linspace(0,1508,128,dtype=int),'development':np.linspace(1509,1887,128,dtype=int)}
        saved={};metrics={}
        for role,ids in roles.items():
            pred=predictions(model,visual,language,language_mask,state,ids)
            saved[role+'_predictions']=pred;saved[role+'_targets']=a['raw_targets'][ids]
            saved[role+'_mask']=a['mask'][ids];saved[role+'_previous']=a['previous_action'][ids];saved[role+'_indices']=ids
            metrics[role]=summarize(pred,a['raw_targets'][ids],a['mask'][ids],a['previous_action'][ids],normalizer)
        np.savez_compressed(out/'predictions.npz',**saved)
        payload={'format':'libero90-native-supervised-policy-v1','action_contract':'native_osc_pose7_close_positive_20hz_v1',
            'target_supervised':True,'zero_shot':False,'formal_matrix_admitted':False,'update':4096,
            'state':state_copy(model),'model_config':p['model'],'normalizer':normalizer,
            'language':language.detach().cpu(),'attention_mask':language_mask.cpu(),'task':p['task'],
            'instruction':p['instruction'],'inputs_sha256':sha(HERE/'inputs.npz'),'protocol':p,
            'encoder_identity':{'text_parameters_sha256':read(HERE/'inputs_receipt.json')['text_parameters_sha256'],
                'dino_revision':'47b73eefe95e8d44ec3623f8890bd894b6ea2d6c'}}
        torch.save(payload,out/'policy.pt')
        loaded=torch.load(out/'policy.pt',map_location='cpu',weights_only=False)
        if state_sha(loaded['state'])!=state_sha(payload['state']):raise ValueError('Full policy readback differs')
        report.update(execution_complete=True,optimizer_updates=4096,initial_state_sha256=initial,
            final_state_sha256=state_sha(payload['state']),policy_sha256=sha(out/'policy.pt'),metrics=metrics,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,elapsed_seconds=time.monotonic()-started,
            elapsed_scope='After CPU admission; includes loading/encoding/training/offline sampling; before final verifier/ZIP',
            learned_policy_task_success=None)
        write(out/'report.json',report)
        from verify import verify
        write(out/'verification.json',verify(out))
        import zipfile
        with zipfile.ZipFile(out/'small_results.zip','x',zipfile.ZIP_DEFLATED) as z:
            for name in ('report.json','predictions.npz','admission.json','verification.json'):z.write(out/name,name)
        print('LIBERO_NATIVE_SUPERVISED_COMPLETE',json.dumps(report),flush=True)
    except BaseException as e:
        # An asynchronous timeout may interrupt optimizer.step; retain the last
        # atomic checkpoint, never label a partly applied step as complete.
        report.update(optimizer_updates=completed,error=type(e).__name__+': '+str(e),elapsed_seconds=time.monotonic()-started)
        write(out/'report.json',report);raise
    finally:signal.alarm(0)


if __name__=='__main__':main()

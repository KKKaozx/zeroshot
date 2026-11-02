"""One manually submitted A40 pair,2048 updates each,3300s total process cap."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE/'code'))


def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p, v): Path(p).write_text(json.dumps(v, indent=2, allow_nan=False)+'\n', encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for data in iter(lambda: f.read(4*1024**2), b''): h.update(data)
    return h.hexdigest()
def tensor_sha(v): return hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def recipe(value):
    import copy
    value=copy.deepcopy(value)
    for key in ('name','dino_assets'):value['model'].pop(key,None)
    return value
def state_sha(v):
    h=hashlib.sha256()
    for name, value in sorted(v.items()): h.update(name.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def seed(n):
    import numpy as np
    import torch
    np.random.seed(n); torch.manual_seed(n); torch.cuda.manual_seed_all(n)


def admit(root, *, load_cache=True):
    import torch
    for line in (HERE/'SHA256SUMS').read_text().splitlines():
        digest, name = line.split('  ', 1)
        if sha(HERE/name) != digest: raise ValueError('Package changed: '+name)
    p, binding = read(HERE/'protocol.json'), read(HERE/'inputs_receipt.json')
    source, head, cache = [root/p[name] for name in ('source_checkpoint', 'protected_gripper', 'feature_cache')]
    for path in (source, head, cache):
        if not path.is_file(): raise FileNotFoundError('Existing artifact required; no encoding/download fallback: '+str(path))
    if sha(source) != binding['source_checkpoint_sha256'] or sha(head) != p['protected_gripper_sha256']:
        raise ValueError('Protected source/head bytes differ')
    result = {'passed': True, 'source': str(source), 'source_sha256': sha(source), 'protected_head': str(head),
        'protected_head_sha256': sha(head), 'cache': str(cache), 'gpu_used': False, 'training_updates': 0}
    cached = None
    if load_cache:
        cached = torch.load(cache, map_location='cpu', weights_only=True)
        key = hashlib.sha256(json.dumps(cached['key'], sort_keys=True).encode()).hexdigest()
        if key != binding['source_token_cache_key_sha256']: raise ValueError('Original feature-cache identity differs')
        if cached['key']['tokenizer_inputs'] != binding['tokenizer_inputs_sha256']:
            raise ValueError('Prepared mask/tokenizer order differs from original features')
        if cached['key']['input_identity_sha256'] != binding['source_input_identity_sha256']:
            raise ValueError('Cache source rows differ')
        for name in ('visual', 'language'):
            if tensor_sha(cached[name]) != cached['hashes'][name] or len(cached[name]) != 1876:
                raise ValueError('Cached tensor corrupted or reordered')
        result['original_frozen_feature_cache_verified'] = True
    return result, cached


def predictions(model, tensors, ids):
    import numpy as np
    import torch
    model.eval(); draws=[]
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]), torch.inference_mode():
        for n in (1101,1102,1103):
            seed(n); parts=[]
            for start in range(0,len(ids),2):
                ix=torch.as_tensor(ids[start:start+2],device='cuda')
                context=model.context(tensors['visual'][ix],tensors['language'][ix],tensors['mask'][ix],tensors['state'][ix])
                parts.append(model.sample(context,tensors['current'][ix])[...,:7].cpu().numpy())
            draws.append(np.concatenate(parts))
    return np.stack(draws)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=Path('/projects/Zeroshot'))
    ap.add_argument('--output',type=Path); ap.add_argument('--check-only',action='store_true')
    args=ap.parse_args(); root=args.root.resolve()
    admission,cached=admit(root)
    if args.check_only:
        print('EPISODE_RELATIVE_ADMISSION',json.dumps(admission),flush=True); return
    if args.output is None or not os.environ.get('SLURM_JOB_ID'): raise ValueError('User-submitted allocation required')
    out=args.output.resolve()
    if out.parent != (root/'runs').resolve() or not out.name.startswith('episode-relative-pair-'):
        raise ValueError('Dedicated fresh output required')
    import numpy as np
    import torch
    from pose_core import CachedPoseCore
    from pair_metrics import summarize,decision
    from episode_relative_pose_state import ABSOLUTE_FORMAT,RELATIVE_FORMAT
    if str(torch.__version__)!='2.5.1+cu118' or not torch.cuda.is_available() or torch.cuda.get_device_name()!='NVIDIA A40':
        raise RuntimeError('Expected existing Torch2.5.1+cu118 and one A40')
    reserve=8*1024**3; used=int(subprocess.check_output(['du','-sB1',str(root)],text=True).split()[0])
    if used+reserve>=200_000_000_000 or shutil.disk_usage(root).free<reserve:
        raise RuntimeError('Need8GiB output/atomic-save reserve; du is an estimate, no automatic cleanup')
    out.mkdir(exist_ok=False); write(out/'admission.json',admission)
    started=time.monotonic(); stopped=False
    def stop(signum,frame):
        nonlocal stopped
        stopped=True
    def hard_stop(signum,frame): raise TimeoutError('3300s whole-pair cap; no continuation')
    signal.signal(signal.SIGUSR1,stop);signal.signal(signal.SIGTERM,stop)
    signal.signal(signal.SIGALRM,hard_stop);signal.alarm(3300)
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    # Match the old Torch2.5.1 recipe: matmul TF32 off, cuDNN TF32 on.
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    torch.use_deterministic_algorithms(True,warn_only=True)
    p=read(HERE/'protocol.json'); binding=read(HERE/'inputs_receipt.json')
    source=torch.load(admission['source'],map_location='cpu',weights_only=False)
    if source['update']!=8192 or recipe(source['config'])!=recipe(read(HERE/'source_config.json')):
        raise ValueError('Source architecture/update changed')
    rows=read(HERE/'rows.json'); rows=rows['train']+rows['development']
    with np.load(HERE/'inputs.npz',allow_pickle=False) as z: arrays={k:z[k].copy() for k in z.files}
    with np.load(HERE/'source_reference.npz',allow_pickle=False) as z: reference={k:z[k].copy() for k in z.files}
    with np.load(HERE/'plan.npz',allow_pickle=False) as z: batches=z['batches'].copy()
    if batches.shape!=(2048,8) or batches.min()<0 or batches.max()>=1836: raise ValueError('Training-only plan differs')
    roles={'train':np.arange(128),'development':np.arange(1836,1876)}
    baseline={role:summarize(reference[role+'_predictions'][...,:7],arrays['targets'][ids],[rows[i] for i in ids])
              for role,ids in roles.items()}
    tensors={'visual':cached['visual'].cuda(),'language':cached['language'].cuda(),
        'mask':torch.from_numpy(arrays['attention_mask']).cuda(),'current':torch.from_numpy(arrays['current']).cuda(),
        'targets':torch.from_numpy(arrays['targets']).cuda()}
    del cached
    torch.cuda.reset_peak_memory_stats(); arms={}; saved_predictions={}; initial_hash=None
    report={'execution_complete':False,'arms':arms,'baseline':baseline,'source_pose_updates_to_original':0,
        'completed_updates':{'absolute':0,'relative':0},
        'new_gripper_updates':0,'new_encoder_updates':0,'new_frames_encoded':0,'formal_matrix_admitted':False,
        'automatic_rollout':False,'automatic_following_action':False,'hypothesis':'World-frame-independent current-state inputs can retain source pose ability',
        'state_change_is_new_representation_not_proven_unique_root_cause':True}
    write(out/'report.json',report)
    try:
        for mode,fmt in [('absolute',ABSOLUTE_FORMAT),('relative',RELATIVE_FORMAT)]:
            folder=out/mode;folder.mkdir()
            model=CachedPoseCore(source['config']).restore(source['trainable_state_dict']).cuda()
            this_hash=state_sha(model.pose_state())
            if initial_hash is not None and this_hash!=initial_hash: raise ValueError('Pair initial parameters differ')
            initial_hash=this_hash
            tensors['state']=torch.from_numpy(arrays[mode+'_features']).cuda()
            optimizer=torch.optim.Adam(model.parameters(),lr=1e-5,weight_decay=0)
            if optimizer.state: raise ValueError('Both new Adam states must start empty')
            if mode=='absolute':
                model.eval()
                with torch.inference_mode(),torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                    context=model.context(tensors['visual'][:2],tensors['language'][:2],tensors['mask'][:2],tensors['state'][:2])
                    context_delta=float(np.max(np.abs(context.cpu().numpy()-reference['train_contexts'][:2])))
                    seed(1101);native=model.sample(context,tensors['current'][:2])[...,:7].cpu().numpy()
                    pose_delta=float(np.max(np.abs(native-reference['train_predictions'][0,:2,:,:7])))
                if context_delta>1e-5 or pose_delta>1e-4:
                    raise ValueError('Native source restore/cache guard failed; no tolerance changes or retries: '+str((context_delta,pose_delta)))
                report['native_restore_guard']={'rows':2,'draws':1,'max_context_delta':context_delta,'max_pose_delta':pose_delta}
            else:
                for role,ids in roles.items():
                    value=predictions(model,tensors,ids)
                    saved_predictions['relative_initial_'+role]=value
                report['relative_initial']={role:summarize(saved_predictions['relative_initial_'+role],arrays['targets'][ids],[rows[i] for i in ids]) for role,ids in roles.items()}
            history=[]; completed=0
            def save():
                checkpoint={'format':'episode-relative-pair-training-state-v1','mode':mode,'update':completed,
                    'state':model.pose_state(),'optimizer':optimizer.state_dict(),'history':history,
                    'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
                    'numpy_rng':np.random.get_state(),'plan_sha256':binding['plan_sha256'],'protocol':p,
                    'source_checkpoint_sha256':admission['source_sha256'],'state_feature_format':fmt}
                temporary=folder/'latest.pt.tmp';torch.save(checkpoint,temporary);temporary.replace(folder/'latest.pt')
            try:
                with (folder/'updates.jsonl').open('x') as log:
                    for i in range(2048):
                        if stopped: raise TimeoutError('Stop signal received; retain checkpoint, no continuation')
                        model.train();ix=torch.as_tensor(batches[i],device='cuda')
                        seed(430000+8193+i)
                        context=model.context(tensors['visual'][ix],tensors['language'][ix],tensors['mask'][ix],tensors['state'][ix])
                        seed(420000+8193+i)
                        predicted,truth,_=model.diffusion_loss(tensors['targets'][ix],context,tensors['current'][ix])
                        loss=torch.nn.functional.mse_loss(predicted,truth)
                        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
                        optimizer.zero_grad(set_to_none=True);loss.backward()
                        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                        optimizer.step();completed=i+1
                        report['completed_updates'][mode]=completed
                        row={'update':completed,'loss':float(loss.detach()),'gradient_norm':float(norm),'lr':1e-5}
                        history.append(row);log.write(json.dumps(row)+'\n')
                        if completed%512==0:save()
                        if i<2 or completed%256==0:
                            log.flush();print('EPISODE_RELATIVE_UPDATE',mode,row,flush=True)
                result={}
                for role,ids in roles.items():
                    pred=predictions(model,tensors,ids);saved_predictions[mode+'_'+role]=pred
                    result[role]=summarize(pred,arrays['targets'][ids],[rows[i] for i in ids])
                arms[mode]=result
                endpoint={'format':'episode-relative-pose-policy-v1','mode':mode,'state_feature_format':fmt,
                    'source_checkpoint_sha256':admission['source_sha256'],'additional_updates':2048,'source_update':8192,
                    'initial_parameters_sha256':initial_hash,'config':source['config'],'state':model.pose_state(),
                    'state_statistics':binding['state_statistics'],'initial_reference':'first observed pose of episode, index0',
                    'action_contract':source['config']['action'],'protocol':p,'formal_matrix_admitted':False,
                    'gripper_trained':False,'original_gripper_context_compatibility_not_proven':True}
                torch.save(endpoint,folder/'policy.pt')
                loaded=torch.load(folder/'policy.pt',map_location='cpu',weights_only=False)
                if state_sha(loaded['state'])!=state_sha(endpoint['state']):raise ValueError('Endpoint readback differs')
                write(folder/'report.json',{'metrics':result,'updates':2048,'initial_parameters_sha256':initial_hash,
                    'endpoint_sha256':sha(folder/'policy.pt'),'state_feature_format':fmt})
                report['arms']=arms;write(out/'report.json',report)
            except BaseException:
                if completed:save()
                raise
            del optimizer,model,context,predicted,truth,loss,endpoint,loaded
            torch.cuda.empty_cache()
        if sha(admission['source'])!=admission['source_sha256'] or sha(admission['protected_head'])!=admission['protected_head_sha256']:
            raise RuntimeError('Protected original artifact changed')
        for role,ids in roles.items():
            saved_predictions[role+'_targets']=arrays['targets'][ids]
            saved_predictions[role+'_indices']=ids
        np.savez_compressed(out/'predictions.npz',**saved_predictions)
        report.update(decision(arms,baseline));report.update(execution_complete=True,total_new_pose_updates=4096,
            original_files_unchanged=True,elapsed_seconds=time.monotonic()-started,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,
            elapsed_scope='After CPU artifact/cache admission, before final CPU endpoint verification and ZIP;3300s alarm still covers verification/packing')
        write(out/'report.json',report)
        from verify_pair import verify
        write(out/'verification.json',verify(out))
        import zipfile
        with zipfile.ZipFile(out/'small_results.zip','x',zipfile.ZIP_DEFLATED) as z:
            for name in ('report.json','predictions.npz','verification.json','admission.json','absolute/report.json','relative/report.json'):
                z.write(out/name,name)
        terminal={k:v for k,v in report.items() if k not in ('arms','baseline','relative_initial')}
        terminal['development']={mode:{k:arms[mode]['development'][k] for k in ('episode_position_cm','episode_rotation_deg')}
                                 for mode in arms}
        print('EPISODE_RELATIVE_COMPLETE',json.dumps(terminal),flush=True)
    except BaseException as e:
        report.update(error=type(e).__name__+': '+str(e),elapsed_seconds=time.monotonic()-started)
        write(out/'report.json',report);raise
    finally:signal.alarm(0)


if __name__=='__main__':main()

"""Bounded frozen endpoint action audit. No model training or environment steps."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
import numpy as np
import torch
from resource_probe import sha, state_hash, source_root
from audit_math import select_windows, summarize, summarize_epsilon
from endpoint_ops import load_endpoint, frozen_batch


def save(path, obj):
    temporary = Path(str(path)+'.writing')
    temporary.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
    os.replace(temporary,path)


def arrays_save(path, **arrays):
    temporary = Path(str(path)+'.writing')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream,**arrays)
    os.replace(temporary,path)


def array_sha(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/projects/Zeroshot'))
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args()
    root,out,bundle = args.root.resolve(),args.output.resolve(),Path(__file__).resolve().parent
    if root / 'runs' not in out.parents:
        p.error('New output must be under root/runs')
    out.mkdir(parents=True,exist_ok=False)
    started = time.monotonic()
    result = dict(complete=False,errors=[],training_updates=0,environment_steps=0,policy_rollouts=0,
        final50_accessed=False,automatic_submission=False,groups={},
        scope='Expert-window action diagnostic, reused episode roles, not policy improvement or task success')
    def interrupted(signum, frame):
        raise TimeoutError('Bounded audit interrupted; partial output is not a model verdict')
    signal.signal(signal.SIGTERM,interrupted)
    signal.signal(signal.SIGALRM,interrupted)
    signal.alarm(1700)
    client = None
    try:
        protocol = json.loads((bundle/'protocol.json').read_text())
        result['protocol_sha256'] = sha(bundle/'protocol.json')
        for name,digest in protocol['bundle_sources_sha256'].items():
            if sha(bundle/name) != digest: raise ValueError('Changed bundle: '+name)
        if not torch.cuda.is_available() or torch.__version__.split('+')[0] != '1.12.1':
            raise ValueError('Use existing official CUDA environment')
        torch.set_num_threads(4)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        base = json.loads((bundle/'base_protocol.json').read_text())
        source_root(root,base)
        import zarr
        from diffusion_policy.common.sampler import create_indices
        path = root/'runs/pusht-float-data-210690/report.json'
        if sha(path) != base['data_report_sha256']: raise ValueError('Changed data report')
        data_report = json.loads(path.read_text())
        if not data_report['data_integrity_passed']: raise ValueError('Data not qualified')
        source = zarr.open_group(data_report['dataset_path'],mode='r')
        ends,actions,state = [source[k][:] for k in ('meta/episode_ends','data/action','data/state')]
        for name,value in zip(('meta/episode_ends','data/action','data/state'),(ends,actions,state)):
            if array_sha(value) != data_report['arrays'][name]['decoded_sha256']: raise ValueError('Changed '+name)
        train_ids = data_report['author_recipe_counts']['train_episode_ids']
        if not set(protocol['selected_episode_ids']['train']) <= set(train_ids) or set(protocol['selected_episode_ids']['unseen_demo']) & set(train_ids):
            raise ValueError('Episode role leakage')
        info,records = select_windows(ends,protocol['selected_episode_ids'],create_indices)
        frames,valid,episodes,roles = [info[k] for k in ('frames','valid','episode_ids','roles')]
        obs_ids = frames[:,:2]
        raw = np.stack([source['data/img'][int(i)] for i in obs_ids.reshape(-1)]).reshape(64,2,96,96,3)
        if raw.dtype != np.float32 or not np.isfinite(raw).all() or raw.min()<0 or raw.max()>255:
            raise ValueError('Invalid raw pixels')
        targets,positions = actions[frames].copy(),state[obs_ids,:2].copy()
        frame_mask = np.zeros(len(actions),dtype=bool)
        begin = 0
        for ep,end in enumerate(ends):
            if ep in train_ids: frame_mask[begin:int(end)] = True
            begin = int(end)
        mean = actions[frame_mask].astype(np.float64).mean(0).astype(np.float32)
        arrays_save(out/'inputs.npz',**info,episode_ends=ends,raw_images=raw,targets=targets,agent_pos=positions,train_action_mean=mean)
        save(out/'window_manifest.json',dict(windows=records,source_numeric_hashes={k:data_report['arrays'][k]['decoded_sha256']
            for k in ('meta/episode_ends','data/action','data/state')},selected_raw_pixels_sha256=array_sha(raw),
            selection=protocol['selection_rule'],execution_slice=[1,9],padding_metric='Exclude padded target slots'))
        result.update(input_sha256=sha(out/'inputs.npz'),manifest_sha256=sha(out/'window_manifest.json'))
        del source,state,actions
        first_noise = None
        for name in ('native212158','project215999'):
            project = name == 'project215999'
            spec = protocol['checkpoints'][name]
            policy = load_endpoint(root,bundle,spec,base,project)
            before = state_hash(policy)
            visual = None
            if project:
                cache_dir = root/'runs/pusht-project-cache-213439'
                cache_report_path = cache_dir/'report.json'
                if sha(cache_report_path) != protocol['cache_report_sha256']: raise ValueError('Changed cache report')
                cached = json.loads(cache_report_path.read_text())
                if not cached['cache_complete'] or cached['errors'] or not cached['reference128']['passed']:
                    raise ValueError('Unqualified cache')
                for filename,item in cached['files'].items():
                    if (cache_dir/filename).stat().st_size != item['bytes']: raise ValueError('Cache length changed')
                cached_ids = np.load(cache_dir/'frame_ids.npy',allow_pickle=False)
                if sha(cache_dir/'frame_ids.npy') != cached['files']['frame_ids.npy']['sha256']: raise ValueError('Cache ids changed')
                cache = np.load(cache_dir/'visual.npy',mmap_mode='r',allow_pickle=False)
                if cache.shape != (11356,257,1024) or cache.dtype != np.float32: raise ValueError('Cache shape changed')
                from live_client import CLIPClient
                client = CLIPClient(root,out)
                if not (np.array_equal(client.text,policy.obs_encoder.text_tokens.cpu().numpy())
                        and np.array_equal(client.text_mask,policy.obs_encoder.text_mask.cpu().numpy())):
                    raise ValueError('Text differs from frozen endpoint')
                visual = np.empty((128,257,1024),dtype=np.float32)
                for i in range(0,128,8):
                    ids = obs_ids.reshape(-1)[i:i+8]
                    if (roles[i//2:i//2+4] == 'train').all():
                        where = np.searchsorted(cached_ids,ids)
                        if (where >= len(cached_ids)).any() or not np.array_equal(cached_ids[where],ids): raise ValueError('Cache row mismatch')
                        visual[i:i+8] = cache[where]
                    else:
                        visual[i:i+8] = client.encode(raw.reshape(-1,96,96,3)[i:i+8])
                # Four fixed train observations: identity check for this new selection, not another full cache qualification.
                fresh = client.encode(raw[:2].reshape(-1,96,96,3))
                delta = float(np.max(np.abs(fresh-visual[:4])))
                if delta>1e-4: raise ValueError('Selected cached/live tokens differ')
                close = client.close(); client = None
                if not close['all_frozen'] or not close['weights_unchanged']: raise ValueError('CLIP changed')
                if not np.isfinite(visual).all(): raise ValueError('Invalid selected tokens')
                result['clip'] = dict(worker=close,selected_reference_max_abs=delta,selected_tokens_sha256=array_sha(visual),
                    cache_revalidation_scope='Pinned qualified report, sizes, exact frame-id hash and four selected live references; no full11GiB rehash')
                del cache
            predictions = np.full((2,64,16,2),np.nan,dtype=np.float32)
            epsilon = np.full((2,3,64),np.nan,dtype=np.float32)
            traces = []
            for seed_index,seed in enumerate(protocol['sampling_seeds']):
                for batch_index,start in enumerate(range(0,64,4)):
                    stop = start+4
                    obs = dict(image=torch.from_numpy(raw[start:stop].transpose(0,1,4,2,3)/np.float32(255)).cuda(),
                               agent_pos=torch.from_numpy(positions[start:stop]).cuda())
                    tok = torch.from_numpy(visual[start*2:stop*2].copy()).cuda() if project else None
                    pred,err,trace = frozen_batch(policy,obs,torch.from_numpy(targets[start:stop]).cuda(),tok,seed,batch_index)
                    predictions[seed_index,start:stop],epsilon[seed_index,:,start:stop] = pred,err
                    traces.append(dict(sampling_seed=seed,batch_index=batch_index,**trace))
                    arrays_save(out/(name+'_predictions.npz'),predictions=predictions,epsilon_mse=epsilon)
                    save(out/'progress.json',dict(group=name,sampling_seed=seed,completed_windows=stop,complete=False))
                    print('ENDPOINT_ACTION_PROGRESS',name,seed,stop,'/64',flush=True)
            if first_noise is None: first_noise = traces
            elif traces != first_noise: raise ValueError('Native/project sampling or auxiliary noise streams differ')
            if state_hash(policy) != before or any(p.requires_grad or p.grad is not None for p in policy.parameters()):
                raise ValueError('Frozen endpoint changed')
            row = dict(checkpoint_sha256=spec['sha256'],state_sha256=before,state_unchanged=True,all_frozen=True,
                common_noise_traces=traces,predictions_sha256=sha(out/(name+'_predictions.npz')),
                action_metrics=summarize(predictions,targets,valid,episodes,roles),
                epsilon_mse_all16=summarize_epsilon(epsilon,episodes,roles))
            result['groups'][name] = row
            save(out/'report.json',result)
            del policy,visual,obs,tok,predictions,epsilon
            gc.collect(); torch.cuda.empty_cache()
        static = np.broadcast_to(positions[:,-1:, :],(64,16,2)).copy()[None]
        average = np.broadcast_to(mean,(64,16,2)).copy()[None]
        result['baselines'] = {name:summarize(p,targets,valid,episodes,roles) for name,p in (('no_motion',static),('train_mean',average))}
        result['complete'] = True
        print('ENDPOINT_ACTION_AUDIT_COMPLETE; diagnostic only, no automatic training',flush=True)
    except Exception as exc:
        result['errors'].append(type(exc).__name__+': '+str(exc))
        raise
    finally:
        signal.alarm(0)
        if client is not None: client.abort()
        result['elapsed_seconds'] = time.monotonic()-started
        save(out/'report.json',result)
        print('ENDPOINT_ACTION_REPORT',out/'report.json',flush=True)


if __name__ == '__main__':
    main()

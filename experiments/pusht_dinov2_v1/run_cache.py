"""Train-only DINO cache and actual cross-runtime cached/live identity; zero updates."""
import argparse
import gc
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import time

import numpy as np
from cache_tools import sha, encode_frames, compare_reference
from dino_binding import assert_sources, read, expected_binding, validate_output


def save(path, value):
    temporary = path.with_name(path.name+'.writing')
    with temporary.open('w') as f:
        json.dump(value,f,indent=2,allow_nan=False)
        f.write('\n')
    os.replace(temporary,path)


def check_space_values(used, free, p):
    need = p['cache_array_bytes'] + p['reserved_after_cache_bytes']
    if used < 0 or free < 0 or free < need or used + need > p['conservative_project_budget_bytes']:
        raise ValueError('Insufficient space for11.13GiB DINO cache plus24GiB reserve; '
                         'run the reviewed cleanup first, retain reports/data/final checkpoints')
    return dict(project_allocated_bytes=used,filesystem_free_bytes=free,
        required_before_cache_bytes=need,reserved_after_cache_bytes=p['reserved_after_cache_bytes'],
        conservative_project_budget_bytes=p['conservative_project_budget_bytes'],
        passed=True,scope='du project allocation and filesystem availability; not an authoritative account quota query')


def storage_guard(root,p):
    # Cluster project is quota-limited even when the shared filesystem has space.
    result = subprocess.run(['du','-sx','-B1',str(root)],check=True,capture_output=True,text=True,timeout=180)
    used = int(result.stdout.split()[0])
    return check_space_values(used,shutil.disk_usage(root).free,p)


def live_check(root,out,p,data,visual,ids,text,mask):
    import torch
    import hydra
    from omegaconf import OmegaConf
    from diffusion_policy.model.common.normalizer import LinearNormalizer
    from diffusion_policy.common.normalize_util import get_image_range_normalizer
    from project_frontend import ProjectObservationEncoder
    from resource_probe import state_hash
    from dino_client import DINOClient, LiveProjectPolicy
    from wire import raw_from_author_observations
    bundle = Path(__file__).resolve().parent
    base = read(bundle/'base_protocol.json')
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    policy = hydra.utils.instantiate(OmegaConf.load(bundle/'checkpoint_config.yaml').policy)
    initial = state_hash(policy.model)
    if initial != p['initial_backend_sha256']:
        raise ValueError('Fresh author backend identity differs')
    torch.manual_seed(43)
    policy.obs_encoder = ProjectObservationEncoder(torch.from_numpy(text.copy()),torch.from_numpy(mask.copy()),(64,65))
    if sum(x.numel() for x in policy.parameters() if x.requires_grad) != 305159362:
        raise ValueError('Original8 Adapter/backend structure differs')
    normalizer = LinearNormalizer()
    normalizer.fit({'action':data.train_actions,'agent_pos':data.train_positions},last_n_dims=1,mode='limits')
    normalizer['image'] = get_image_range_normalizer()
    policy.set_normalizer(normalizer)
    normalizer_sha = state_hash(policy.normalizer)
    if normalizer_sha != base['normalizer_sha256']:
        raise ValueError('Original train-only normalizer differs')
    policy.cuda().eval()
    before = state_hash(policy)
    chosen = read(root/'runs/pusht-front-resource-210763/report.json')['records']['prepare']['sampled_windows']
    from token_binding import observation_ids
    worker_dir = out/'live_worker'
    worker_dir.mkdir()
    client = DINOClient(root,worker_dir)
    context_delta, token_delta, action_delta = 0.,0.,0.
    try:
        if not np.array_equal(client.text,text) or not np.array_equal(client.text_mask,mask):
            raise ValueError('Live CLIP text differs from cached text')
        for size in (1,2,4):
            indices = np.asarray(chosen[:size])
            batch = data.batch(indices)
            frame_ids = observation_ids(data.rows[indices]).reshape(-1)
            local = np.searchsorted(ids,frame_ids)
            if not np.array_equal(ids[local],frame_ids):
                raise ValueError('Live identity attempted a nontraining frame')
            cached = np.asarray(visual[local]).copy()
            raw = raw_from_author_observations(batch['obs']['image'].numpy())
            if not np.array_equal(raw,data.images[data.lookup[frame_ids]]):
                raise ValueError('Author observation round trip changes original raw pixels')
            online = client.encode(raw)
            token_delta = max(token_delta,float(np.max(np.abs(cached-online))))
            if not np.allclose(cached,online,atol=p['token_atol'],rtol=p['token_rtol']):
                raise ValueError('Actual IPC tokens differ from cache')
            obs = {k:v.cuda() for k,v in batch['obs'].items()}
            positions = policy.normalizer['agent_pos'].normalize(obs['agent_pos']).reshape(-1,2)
            contexts = []
            with torch.no_grad():
                for tokens in (cached,online):
                    policy.obs_encoder.set_tokens(torch.from_numpy(tokens).cuda())
                    try:
                        contexts.append(policy.obs_encoder({'agent_pos':positions}).detach().cpu().numpy())
                    finally:
                        policy.obs_encoder.cached_tokens = None
            delta = float(np.max(np.abs(contexts[0]-contexts[1])))
            context_delta = max(context_delta,delta)
            if delta > p['context_atol']:
                raise ValueError('Actual Adapter context differs between cached and live inputs')
            if size == 2:
                with torch.no_grad():
                    policy.obs_encoder.set_tokens(torch.from_numpy(cached).cuda())
                    try:
                        torch.manual_seed(1101)
                        direct = policy.predict_action(obs)
                    finally:
                        policy.obs_encoder.cached_tokens = None
                    torch.manual_seed(1101)
                    live = LiveProjectPolicy(policy,client).predict_action(obs)
                for key,shape in (('action',(2,8,2)),('action_pred',(2,16,2))):
                    if tuple(direct[key].shape) != shape or tuple(live[key].shape) != shape:
                        raise ValueError('Cached/live original DDPM action shape changed')
                    delta = float((direct[key]-live[key]).abs().max().cpu())
                    if not np.isfinite(delta) or delta > p['action_atol']:
                        raise ValueError('Actual DDPM100 cached/live action differs')
                    action_delta = max(action_delta,delta)
                np.savez(out/'live_action_identity.npz',cached_action=direct['action'].cpu().numpy(),
                    live_action=live['action'].cpu().numpy(),cached_full16=direct['action_pred'].cpu().numpy(),
                    live_full16=live['action_pred'].cpu().numpy())
            print('DINO_LIVE_IDENTITY batch',size,'tokens',token_delta,'context',context_delta,flush=True)
        unchanged = state_hash(policy) == before
        if not unchanged:
            raise ValueError('Fresh disposable policy changed during eval')
        worker = client.close()
        client = None
        return dict(context_and_action_passed=True,batch_sizes=[1,2,4],
            token_max_abs=token_delta,context_max_abs=context_delta,action_max_abs=action_delta,
            initial_backend_sha256=initial,normalizer_sha256=normalizer_sha,
            policy_state_unchanged=True,fixed_text_exact=True,sampling_seed=1101,
            training_updates=0,rollouts=0,scope='Fresh untrained policy, actual new worker and old callback; no task score'),worker
    finally:
        if client is not None:
            client.abort()


def run(root,out,bundle,p):
    import torch
    from native_data import TrainingData
    from resource_probe import source_root
    from dino_client import DINOClient
    if torch.__version__.split('+')[0] != '1.12.1' or not torch.cuda.is_available():
        raise ValueError('Use existing official CUDA environment; worker uses existing project runtime')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    result = dict(cache_complete=False,errors=[],training_updates=0,
        development_or_final_scored=False,formal_training_ready=False,
        automatic_training_submission=False,cache_contract_sha256=sha(bundle/'protocol.json'))
    client = None
    writer = None
    try:
        result['storage'] = storage_guard(root,p)
        for relative,digest in p['required_reference_files_sha256'].items():
            if sha(root/relative) != digest:
                raise ValueError('Qualified input/reference changed: '+relative)
        base = read(bundle/'base_protocol.json')
        source_root(root,base)
        old = root/'runs/pusht-project-cache-213439'
        old_report = read(old/'report.json')
        data = TrainingData(root,base,out)
        if (data.evidence != old_report['training_data']
                or data.evidence['raw_training_images_sha256'] != p['raw_training_images_sha256']):
            raise ValueError('Training split/images differ from original CLIP cache')
        ids = np.flatnonzero(data.lookup >= 0).astype('<i8')
        if not np.array_equal(ids,np.load(old/'frame_ids.npy',allow_pickle=False)):
            raise ValueError('Original training-frame order differs')
        np.save(out/'frame_ids.npy',ids,allow_pickle=False)
        shutil.copyfile(old/'text.npz',out/'text.npz')
        with np.load(out/'text.npz',allow_pickle=False) as f:
            text,mask = f['text'].copy(),f['text_mask'].copy()
        worker_dir = out/'cache_worker'
        worker_dir.mkdir()
        client = DINOClient(root,worker_dir)
        writing = out/'visual.writing.npy'
        writer = np.lib.format.open_memmap(writing,mode='w+',dtype='<f4',shape=(11356,257,1024))
        count = encode_frames(data.images,writer,client,
            lambda stop,total:print('PUSHT_DINO_CACHE',stop,'/',total,flush=True))
        if count != 11356:
            raise ValueError('Not all train frames encoded')
        result['cache_encode_seconds'] = sum(client.latencies)
        worker = client.close()
        client = None
        writer.flush()
        del writer
        writer = None
        gc.collect()
        os.replace(writing,out/'visual.npy')
        visual = np.load(out/'visual.npy',mmap_mode='r',allow_pickle=False)
        ref = root/'runs/pusht-dino-resource-220882'
        resource = read(root/'runs/pusht-front-resource-210763/report.json')
        reference_ids = resource['records']['prepare']['observation_frame_ids']
        if sha(ref/'tokens.npz') != read(ref/'dino.json')['tokens_sha256']:
            raise ValueError('220882 DINO128 reference bytes changed')
        with np.load(ref/'tokens.npz',allow_pickle=False) as f:
            if not np.array_equal(f['text'],text) or not np.array_equal(f['text_mask'],mask):
                raise ValueError('Original text not exact in DINO128 resource')
            comparison = compare_reference(visual,ids,reference_ids,f['visual'].reshape(128,257,1024),
                atol=p['token_atol'],rtol=p['token_rtol'])
        live,live_worker = live_check(root,out,p,data,visual,ids,text,mask)
        result.update(cache_complete=True,training_data=data.evidence,reference128=comparison,
            worker=worker,cache_worker=worker,live_worker=live_worker,live_identity=live,
            files={name:dict(bytes=(out/name).stat().st_size,sha256=sha(out/name))
                   for name in ('visual.npy','frame_ids.npy','text.npz')},
            live_action_file=dict(bytes=(out/'live_action_identity.npz').stat().st_size,
                                  sha256=sha(out/'live_action_identity.npz')),
            visual_shape=list(visual.shape),visual_dtype=str(visual.dtype),
            elapsed_seconds=time.monotonic()-started,
            scope='Frozen DINO train-only cache and cross-runtime identity; no optimization or rollout')
        save(out/'report.json',result)
        save(out/'backbone_binding.json',expected_binding(out,p))
        summary = validate_output(out,bundle)
        print('PUSHT DINO CACHE AND LIVE IPC VERIFIED; no training or generalization verdict',flush=True)
        print(json.dumps(summary,indent=2),flush=True)
    except Exception as exc:
        result['cache_complete'] = False
        result['errors'].append(type(exc).__name__+': '+str(exc))
        result['elapsed_seconds'] = time.monotonic()-started
        save(out/'report.json',result)
        raise
    finally:
        if client is not None:
            client.abort()
        if writer is not None:
            writer.flush()
        print('DINO_CACHE_REPORT',out/'report.json',flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('space','run','verify'))
    parser.add_argument('--root',type=Path,default=Path('/projects/Zeroshot'))
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    bundle = Path(__file__).resolve().parent
    p = assert_sources(bundle)
    root = args.root.resolve()
    if args.stage == 'space':
        print(json.dumps(storage_guard(root,p),indent=2))
        return
    if args.output is None:
        parser.error('--output required')
    out = args.output.resolve()
    if root/'runs' not in out.parents:
        raise ValueError('Output must be under project/runs')
    if args.stage == 'verify':
        print(json.dumps(validate_output(out,bundle),indent=2))
    else:
        run(root,out,bundle,p)


if __name__ == '__main__':
    main()

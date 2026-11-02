"""Pure numeric artifact checks; future training must bind this exact identity."""
import json
from pathlib import Path

import numpy as np
from cache_tools import sha


def read(path):
    return json.loads(Path(path).read_text())


def assert_sources(bundle):
    p = read(bundle/'protocol.json')
    for name, digest in p['sources_sha256'].items():
        if Path(name).name != name or sha(bundle/name) != digest:
            raise ValueError('Cache/IPC source changed: '+name)
    return p


def expected_binding(out, p):
    r = read(out/'report.json')
    return dict(format='pusht-frozen-dino-cache-binding-v1',
        cache_report_sha256=sha(out/'report.json'),
        cache_protocol_sha256=r['cache_contract_sha256'],
        backbone_weights_sha256=p['backbone_weights_sha256'],
        model_state_sha256=p['model_state_sha256'], revision=p['revision'],
        preprocessing=p['preprocessing'], files=r['files'],
        raw_training_images_sha256=p['raw_training_images_sha256'],
        scope='Cache/IPC identity only; no trained checkpoint or performance verdict')


def validate_output(out, bundle):
    p = assert_sources(bundle)
    r = read(out/'report.json')
    if (not r['cache_complete'] or r['errors'] or r['training_updates'] != 0
            or r['development_or_final_scored'] or r['formal_training_ready']
            or r['automatic_training_submission']
            or r['cache_contract_sha256'] != sha(bundle/'protocol.json')):
        raise ValueError('Incomplete or wrong-scope DINO cache')
    if r['visual_shape'] != [11356,257,1024] or r['visual_dtype'] != 'float32':
        raise ValueError('Wrong DINO cache shape/dtype')
    if set(r['files']) != {'visual.npy','frame_ids.npy','text.npz'}:
        raise ValueError('Cache file set differs')
    for name, info in r['files'].items():
        path = out/name
        if path.stat().st_size != info['bytes'] or sha(path) != info['sha256']:
            raise ValueError('Cache file bytes changed: '+name)
    ids = np.load(out/'frame_ids.npy', allow_pickle=False)
    visual = np.load(out/'visual.npy', mmap_mode='r', allow_pickle=False)
    if (visual.shape != (11356,257,1024) or visual.dtype != np.float32
            or ids.dtype != np.int64 or ids.shape != (11356,)
            or (np.diff(ids) <= 0).any()
            or r['files']['frame_ids.npy']['sha256'] != p['frame_ids_sha256']
            or r['files']['text.npz']['sha256'] != p['text_sha256']):
        raise ValueError('Cache indexing/text/array contract changed')
    evidence = r['training_data']
    if (evidence['raw_training_images_sha256'] != p['raw_training_images_sha256']
            or [evidence['train_episodes'],evidence['train_windows'],evidence['train_frames']] != [90,10726,11356]):
        raise ValueError('Training split/pixel identity changed')
    for name in ('cache_worker','live_worker'):
        worker = r[name]
        if (worker.get('all_frozen') is not True or worker.get('weights_unchanged') is not True
                or any(worker.get(k) != p[k] for k in ('model_state_sha256',
                        'backbone_weights_sha256','text_sha256','preprocessing'))):
            raise ValueError('DINO worker identity failed: '+name)
    if r['worker'] != r['cache_worker']:
        raise ValueError('Cache consumer worker evidence differs')
    reference = r['reference128']
    if (reference['passed'] is not True or reference['frames'] != 128
            or reference['atol'] != p['token_atol'] or reference['rtol'] != p['token_rtol']
            or not np.isfinite(reference['max_absolute_difference'])
            or reference['max_absolute_difference'] < 0):
        raise ValueError('Resource128 replay evidence invalid')
    live = r['live_identity']
    if (live['context_and_action_passed'] is not True or live['batch_sizes'] != [1,2,4]
            or live['initial_backend_sha256'] != p['initial_backend_sha256']
            or live['normalizer_sha256'] != p['normalizer_sha256']
            or live['policy_state_unchanged'] is not True or live['fixed_text_exact'] is not True
            or live['sampling_seed'] != 1101 or live['training_updates'] != 0
            or live['rollouts'] != 0):
        raise ValueError('Actual cached/live policy identity invalid')
    for key, limit in [('context_max_abs',p['context_atol']), ('action_max_abs',p['action_atol'])]:
        if not np.isfinite(live[key]) or not 0 <= live[key] <= limit:
            raise ValueError('Cached/live difference too large: '+key)
    action_file = out/'live_action_identity.npz'
    if (action_file.stat().st_size != r['live_action_file']['bytes']
            or sha(action_file) != r['live_action_file']['sha256']):
        raise ValueError('Live action identity evidence bytes changed')
    maximum = 0.
    with np.load(action_file,allow_pickle=False) as f:
        if set(f.files) != {'cached_action','live_action','cached_full16','live_full16'}:
            raise ValueError('Live action identity evidence set differs')
        for name,shape in (('action',(2,8,2)),('full16',(2,16,2))):
            a,b = f['cached_'+name],f['live_'+name]
            if a.shape != shape or b.shape != shape or not np.isfinite(a).all() or not np.isfinite(b).all():
                raise ValueError('Live action identity evidence shape/values differ')
            maximum = max(maximum,float(np.max(np.abs(a-b))))
    if maximum > p['action_atol'] or maximum != live['action_max_abs']:
        raise ValueError('Live action differences do not match the actual evidence')
    if read(out/'backbone_binding.json') != expected_binding(out,p):
        raise ValueError('Future checkpoint identity binding differs')
    return dict(cache_and_live_identity_verified=True,frames=11356,
        cache_data_gib=visual.nbytes/1024**3,reference128=reference,
        live_identity=live,formal_training_ready=False,training_updates=0,
        automatic_training_submission=False,generalization_verdict=None,
        remaining=['Review exact84168 matched budget and checkpoint identity/resume entry before formal submission'])

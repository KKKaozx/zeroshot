"""Check the new release, quota guard, IPC worker and artifact rejection paths."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import socket
import sys
import tempfile
import threading
import types
from unittest.mock import patch
import zipfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
B = ROOT/'scripts/cluster/pusht_dino_cache_v1'
sys.path.insert(0,str(B))
import dino_binding as binding
import run_cache
import dino_worker
from cache_tools import sha
from wire import send,receive

checks = []


def ok(label,condition=True):
    assert condition,label
    checks.append(label)


def reject(label,fn):
    try:
        fn()
    except (ValueError,AssertionError,KeyError):
        checks.append(label)
        return
    raise AssertionError('Bad fixture admitted: '+label)


def save(path,value):
    path.write_text(json.dumps(value,allow_nan=True),encoding='utf-8')


p = binding.assert_sources(B)
ok('Source fingerprints')
for name in p['reused_files_byte_identical']:
    assert (B/name).read_bytes() == (ROOT/'scripts/cluster/pusht_project_training_v1'/name).read_bytes()
ok('Ten qualified original modules byte-identical')
for line in (B/'SHA256SUMS').read_text().splitlines():
    digest,name = line.split(None,1)
    assert sha(B/name.strip()) == digest
ok('All package SHA256 checksums')
with zipfile.ZipFile(ROOT/'cluster_packages/pusht_dino_cache_v1.zip') as z:
    files = sorted(f for f in B.iterdir() if f.is_file())
    assert len(files) == len(z.namelist()) == 20
    assert all(z.read(B.name+'/'+f.name) == f.read_bytes() for f in files)
ok('Exact twenty-file archive; no weights/cache/training launcher')
for f in B.glob('*.py'):
    ast.parse(f.read_text(encoding='utf-8'),feature_version=(3,9))
ok('Python3.9 parsing for parent and worker code')
script = (B/'cache.sh').read_text()
ok('A40 thirty-minute hard limit and isolated runtimes',
   all(x in script for x in ('gpu:a40:1','00:30:00','--no-requeue','official-pusht-v1')))
ok('No training/rollout objective in new driver',
   all(x not in (B/'run_cache.py').read_text() for x in ('.backward(','.step(','.compute_loss(','.reset(','.render(')))
old_interface = binding.read(ROOT/'reports/dinov2_interface_20261023/report.json')
ok('Reuse actual pretrained backbone/Adapter evidence; do not repeat inference',
   old_interface['actual_pretrained_dinov2_loaded'] and old_interface['actual_eight_layer_adapter_gradients_valid'])
need = p['cache_array_bytes']+p['reserved_after_cache_bytes']
ok('Adequate project and filesystem space passes',
   run_cache.check_space_values(100_000_000_000,need,p)['passed'])
reject('Reject full filesystem',lambda:run_cache.check_space_values(100_000_000_000,need-1,p))
reject('Reject project budget exceeded despite filesystem free space',
       lambda:run_cache.check_space_values(p['conservative_project_budget_bytes']-need+1,10**12,p))
reject('Reject invalid usage reading',lambda:run_cache.check_space_values(-1,10**12,p))
ok('Reserve24GiB beyond the complete float32 cache',p['reserved_after_cache_bytes']==24*1024**3)

with tempfile.TemporaryDirectory(dir=ROOT/'reports',prefix='dino_cache_check_') as tmp:
    root = Path(tmp).resolve()
    assert root.parent == (ROOT/'reports').resolve()
    out = root/'out';out.mkdir()
    bundle = root/'bundle';bundle.mkdir()
    contract = copy.deepcopy(p)
    contract['sources_sha256'] = {}
    np.save(out/'frame_ids.npy',np.arange(11356,dtype=np.int64),allow_pickle=False)
    np.savez(out/'text.npz',text=np.zeros((1,2,768),np.float32),text_mask=np.ones((1,2),np.int64))
    (out/'visual.npy').write_bytes(b'fixture marker, not a full cache')
    np.savez(out/'live_action_identity.npz',cached_action=np.zeros((2,8,2),np.float32),
        live_action=np.zeros((2,8,2),np.float32),cached_full16=np.zeros((2,16,2),np.float32),
        live_full16=np.zeros((2,16,2),np.float32))
    contract.update(frame_ids_sha256=sha(out/'frame_ids.npy'),text_sha256=sha(out/'text.npz'))
    save(bundle/'protocol.json',contract)
    worker = dict(all_frozen=True,weights_unchanged=True,**{k:contract[k] for k in
        ('model_state_sha256','backbone_weights_sha256','text_sha256','preprocessing')})
    report = dict(cache_complete=True,errors=[],training_updates=0,development_or_final_scored=False,
        formal_training_ready=False,automatic_training_submission=False,
        cache_contract_sha256=sha(bundle/'protocol.json'),visual_shape=[11356,257,1024],visual_dtype='float32',
        files={name:dict(bytes=(out/name).stat().st_size,sha256=sha(out/name)) for name in
               ('visual.npy','frame_ids.npy','text.npz')},
        live_action_file=dict(bytes=(out/'live_action_identity.npz').stat().st_size,sha256=sha(out/'live_action_identity.npz')),
        training_data=dict(raw_training_images_sha256=p['raw_training_images_sha256'],train_episodes=90,train_windows=10726,train_frames=11356),
        worker=worker,cache_worker=worker,live_worker=worker,
        reference128=dict(passed=True,frames=128,atol=p['token_atol'],rtol=p['token_rtol'],max_absolute_difference=0.),
        live_identity=dict(context_and_action_passed=True,batch_sizes=[1,2,4],initial_backend_sha256=p['initial_backend_sha256'],
            normalizer_sha256=p['normalizer_sha256'],policy_state_unchanged=True,fixed_text_exact=True,sampling_seed=1101,
            training_updates=0,rollouts=0,context_max_abs=0.,action_max_abs=0.))
    original_load = np.load
    visual_header = types.SimpleNamespace(shape=(11356,257,1024),dtype=np.dtype('float32'),nbytes=p['cache_array_bytes'])

    def fixture_load(path,*args,**kwargs):
        return visual_header if Path(path)==out/'visual.npy' else original_load(path,*args,**kwargs)

    def sync(value):
        save(out/'report.json',value)
        save(out/'backbone_binding.json',binding.expected_binding(out,contract))

    with patch.object(binding.np,'load',fixture_load):
        sync(report)
        ok('Synthetic metadata fixture passes without allocating a12GB test cache',
           binding.validate_output(out,bundle)['cache_and_live_identity_verified'])
        mutations = [
            ('Reject training instead of cache-only',('training_updates',),1),
            ('Reject scoring in cache job',('development_or_final_scored',),True),
            ('Reject premature formal readiness',('formal_training_ready',),True),
            ('Reject wrong frame count',('training_data','train_frames'),11355),
            ('Reject changed model state',('live_worker','model_state_sha256'),'wrong'),
            ('Reject changed text evidence',('live_identity','fixed_text_exact'),False),
            ('Reject failed128 reference',('reference128','passed'),False),
            ('Reject NaN context',('live_identity','context_max_abs'),float('nan')),
            ('Reject excessive context delta',('live_identity','context_max_abs'),.01),
            ('Reject fabricated action difference',('live_identity','action_max_abs'),.0001),
            ('Reject changed state columns/normalizer identity',('live_identity','normalizer_sha256'),'wrong'),
        ]
        for label,keys,value in mutations:
            bad = copy.deepcopy(report)
            # Worker dictionaries were aliases in fixture construction; isolate them.
            bad['live_worker'] = dict(bad['live_worker'])
            obj = bad
            for key in keys[:-1]:obj = obj[key]
            obj[keys[-1]] = value
            sync(bad)
            reject(label,lambda:binding.validate_output(out,bundle))
        sync(report)
        changed = binding.read(out/'backbone_binding.json');changed['revision']='wrong'
        save(out/'backbone_binding.json',changed)
        reject('Reject tampered future-checkpoint identity binding',lambda:binding.validate_output(out,bundle))
        sync(report)
        (out/'text.npz').write_bytes(b'changed text')
        reject('Reject actual corrupted text bytes',lambda:binding.validate_output(out,bundle))

    # Exercise the actual new worker/old numeric transport concurrently. The
    # pretrained backbone was checked already; use a tiny deterministic stand-in
    # here solely to isolate worker request/lifecycle and payload validation.
    text_path = root/'text.npz'
    np.savez(text_path,text=np.zeros((1,2,768),np.float32),text_mask=np.ones((1,2),np.int64))
    contract['text_sha256'] = sha(text_path)
    save(bundle/'protocol.json',contract)
    class TinyEncoder(torch.nn.Module):
        def __init__(self,assets):
            super().__init__()
            self.model=torch.nn.Linear(1,1).requires_grad_(False).eval()
        def forward(self,raw):
            from frozen_visual import raw_images
            raw_images(raw)
            return np.broadcast_to(raw.mean(axis=(1,2,3))[:,None,None],(len(raw),257,1024)).copy()
    # Keep raw validation real while replacing only costly pretrained inference.
    import frozen_visual
    fake = types.SimpleNamespace(FrozenDinoVisual=TinyEncoder,
        model_sha=lambda model:contract['model_state_sha256'],file_sha=sha,raw_images=frozen_visual.raw_images)
    original_socket=socket.socket
    for bad_payload in (False,True):
        parent,child=socket.socketpair()
        parent.settimeout(5)
        errors=[]
        def server():
            try:dino_worker.main()
            except Exception as exc:errors.append(type(exc).__name__)
        def socket_factory(*args,**kwargs):
            return child if kwargs.get('fileno')==12345 else original_socket(*args,**kwargs)
        argv=['dino_worker.py','--fd','12345','--assets',str(root),'--text',str(text_path)]
        with patch.object(dino_worker,'__file__',str(bundle/'dino_worker.py')),patch.object(sys,'argv',argv),\
             patch.object(socket,'socket',socket_factory),patch.object(torch.cuda,'is_available',return_value=True),\
             patch.object(torch.cuda,'max_memory_reserved',return_value=1),patch.dict(sys.modules,{'frozen_visual':fake}):
            thread=threading.Thread(target=server,daemon=True);thread.start()
            metadata,arrays=receive(parent)
            assert metadata['status']=='ready' and metadata['model_state_sha256']==p['model_state_sha256']
            assert np.array_equal(arrays['text'],np.zeros((1,2,768),np.float32))
            raw=np.full((2,96,96,3),17,np.float32)
            if bad_payload:raw=raw.astype(np.float64)
            send(parent,{'operation':'encode','request_id':1},{'raw_images':raw})
            metadata,arrays=receive(parent)
            if bad_payload:
                assert metadata['status']=='error' and 'raw float32' in metadata['error']
            else:
                assert metadata=={'status':'ok','request_id':1}
                assert arrays['visual'].shape==(2,257,1024) and np.all(arrays['visual']==17)
                send(parent,{'operation':'close','request_id':2})
                metadata,_=receive(parent)
                assert metadata['weights_unchanged'] and metadata['requests']==1
            parent.close()
            thread.join(5)
            assert not thread.is_alive()
            assert bool(errors)==bad_payload
        ok('Real numeric socket worker rejects float64 payload' if bad_payload else
           'Real numeric socket worker handshake/encode/close with deterministic stand-in')

result = dict(checks_passed=len(checks),checks=checks,
    scope='New packaging/storage/binding guards and actual socket worker with a tiny stand-in; prior real pretrained interface evidence reused',
    full_dino_cache_created=False,new_policy_training=False,new_cluster_job_submitted=False,
    cross_runtime_actual_dino_identity_still_requires_cluster_job=True,
    archive_sha256=sha(ROOT/'cluster_packages/pusht_dino_cache_v1.zip'))
(ROOT/'reports/pusht_dino_cache_checks_20261023.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,indent=2))

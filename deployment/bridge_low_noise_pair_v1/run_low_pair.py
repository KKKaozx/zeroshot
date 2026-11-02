"""Six direct t0denoiser forwards, no100-step resampling and no updates."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
HERE=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    import hashlib
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def attach(root):
    parent=(root/'bridge_rotation_trace_v1').resolve();sys.path[:0]=[str(parent),str(HERE)]
    import run_trace
    assert run_trace.HERE==parent
    return run_trace


def admit(root,space=True):
    p=read(HERE/'protocol.json')
    for line in (HERE/'SHA256SUMS').read_text().splitlines():
        expected,name=line.split('  ',1);assert sha(HERE/name)==expected,name
    for name,expected in p['trace_package_sha256'].items():assert sha(root/'bridge_rotation_trace_v1'/name)==expected,name
    for name,expected in p['trace_results_sha256'].items():assert sha(root/'runs/bridge-rotation-trace-235002'/name)==expected,name
    trace=attach(root);bound=trace.admit(root,space=space)
    r=read(root/'runs/bridge-rotation-trace-235002/report.json')
    assert r['execution_complete'] and r['optimizer_updates']==0 and r['model_adam_gradients_unchanged']
    assert r['source_identity_unchanged']==p['source_final_identity']
    return bound


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path('/projects/Zeroshot'))
    ap.add_argument('--check-only',action='store_true');ap.add_argument('--output',type=Path)
    a=ap.parse_args();root=a.root.resolve();binding=admit(root)
    if a.check_only:print('LOW_NOISE_PAIR_CPU_ADMISSION',json.dumps(binding),flush=True);return
    assert os.environ.get('SLURM_JOB_ID') and a.output is not None
    out=a.output.resolve();assert out.parent==(root/'runs').resolve() and out.name.startswith('bridge-low-pair-') and not out.exists()
    out.mkdir(parents=True);started=time.monotonic();trace=attach(root);rt=trace.attach(root)
    import numpy as np
    import torch
    from bcap_core import load_state,identity
    from lowpair_core import initial_noise,require_original_noise,teacher_forward,summarize
    rt.setup();torch.cuda.reset_peak_memory_stats();b=rt.construct(root,binding['base_admission']);rt.attach_data(b)
    policy=root/'runs/bridge-full-capability-234100/final_policy.pt'
    payload=torch.load(policy,map_location='cpu',weights_only=False,mmap=True)
    trace.check_policy(payload,trace.read(trace.HERE/'protocol.json'))
    assert b['contract']==payload['contract'];load_state(b,payload['state']);del payload
    before=identity(b);assert before==read(HERE/'protocol.json')['source_final_identity']
    b['model'].eval();b['port'].eval();rows=rt.read(rt.HERE/'selection.json')['rows']
    with np.load(root/'runs/bridge-rotation-trace-235002/trace.npz',allow_pickle=False) as z:
        archived_initial=z['latent'][:,:,0].copy();native_raw=z['raw'][:,:,-1].copy();target_ref=z['targets'].copy();mask_ref=z['masks'].copy()
    with torch.inference_mode():
        cached={};targets=[];masks=[]
        for role,ids in [('train',list(range(8))),('validation',list(range(128,136)))]:
            x=b['pack'](role,[rows[i]['window_index'] for i in ids])
            c=b['model'].get_context_vector(x['images'],x['tokens']['input_ids'],x['tokens']['attention_mask'])+b['port'](x['pose'])
            cached[role]=(x,c);targets.append(x['actions'].cpu().numpy());masks.append(x['masks'].cpu().numpy())
        target=np.concatenate(targets);mask=np.concatenate(masks)
        assert np.array_equal(target,target_ref) and np.array_equal(mask,mask_ref)
        noises={}
        # Qualify all6original Gaussian reconstructions BEFORE any denoiser probe.
        for si,seed in enumerate([1101,1102,1103]):
            for role,sl in [('train',slice(0,8)),('validation',slice(8,16))]:
                eps=initial_noise(seed,role,next(b['port'].parameters()).device)
                require_original_noise(eps,archived_initial[si,sl]);noises[(seed,role)]=eps
        print('LOW_NOISE_ORIGINAL_GAUSSIANS_EXACT',6,flush=True)
        inputs=[];outputs=[];epsilon=[];calls=0;tick=time.monotonic()
        for seed in [1101,1102,1103]:
            ix=[];ox=[];ex=[]
            for role in ['train','validation']:
                assert time.monotonic()-tick<120,'Bounded direct-forward cap; no automatic extension'
                x,c=cached[role];eps=noises[(seed,role)]
                noisy,raw=teacher_forward(b['model'],c,x['actions'][...,:7],eps);calls+=1
                ix.append(noisy.cpu().numpy());ox.append(raw.cpu().numpy());ex.append(eps.cpu().numpy())
                print('LOW_NOISE_T0_FORWARD',seed,role,8,flush=True)
            inputs.append(np.concatenate(ix));outputs.append(np.concatenate(ox));epsilon.append(np.concatenate(ex))
    assert calls==6 and identity(b)==before and not b['optimizer'].state
    assert all(p.grad is None for p in list(b['model'].parameters())+list(b['port'].parameters()))
    b['check_frozen']()
    assert [policy.stat().st_size,policy.stat().st_mtime_ns]==binding['source_policy_size_mtime']
    arrays={'teacher_input':np.stack(inputs),'teacher_raw':np.stack(outputs),'epsilon':np.stack(epsilon),
        'native_raw_t0':native_raw,'targets':target,'masks':mask}
    np.savez_compressed(out/'paired_inputs_outputs.npz',**arrays);calculated=summarize(arrays)
    with np.load(out/'paired_inputs_outputs.npz',allow_pickle=False) as z:assert summarize({k:z[k] for k in z.files})==calculated
    report={'execution_complete':True,'protocol_sha256':sha(HERE/'protocol.json'),'source_identity_unchanged':before,
        'original_initial_gaussians_replayed_exactly':True,'source_policy_and_gradients_adam_unchanged':True,
        'direct_denoiser_forwards':calls,'new_full_sampling_runs':0,'optimizer_updates':0,
        'analysis':calculated,'elapsed_seconds':time.monotonic()-started,'peak_reserved_gib':torch.cuda.max_memory_reserved()/1024**3,
        'arrays_sha256':sha(out/'paired_inputs_outputs.npz'),'automatic_following_action':False}
    write(out/'report.json',report)
    import zipfile
    with zipfile.ZipFile(out/'small_results.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
        for name in ['report.json','paired_inputs_outputs.npz']:archive.write(out/name,name)
    print('LOW_NOISE_PAIR_RESULT',json.dumps({'optimizer_updates':0,'direct_denoiser_forwards':6,'new_full_sampling_runs':0,'source_unchanged':True}),flush=True)
    print('LOW_NOISE_PAIR_SMALL_RESULTS',out/'small_results.zip',sha(out/'small_results.zip'),flush=True)
    print('LOW_NOISE_PAIR_COMPLETE',out/'report.json',flush=True)


if __name__=='__main__':main()

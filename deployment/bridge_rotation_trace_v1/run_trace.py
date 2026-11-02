"""Replay16fixed endpoint windows while observing raw native sampling tensors. Zero updates."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
HERE=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def attach(root):
    base=(root/'bridge_full_capability_v1').resolve()
    sys.path[:0]=[str(base/'code'),str(base),str(HERE)]
    import bcap_runtime as rt
    assert rt.HERE==base
    return rt


def check_policy(payload,p):
    from completed_policy_export import model_sha
    from step_training_state import _digest
    assert payload['format']=='fresh-bridge-capability-policy-v1' and payload['completed_updates']==65536
    assert payload['contract']['protocol_sha256']==p['base_protocol_sha256']
    assert payload['contract']['plan_sha256']==p['base_plan_sha256']
    assert payload['contract']['training_seed']==42 and payload['contract']['single_source_supervision']=='Bridge'
    assert {k:model_sha(v) for k,v in payload['state'].items()}==payload['final_identity']==p['final_identity']
    assert payload['contract']['config']['model']['diffusion_prediction_type']=='sample'
    return {'source_policy_parameter_identity_checked_on_cpu':True,'completed_updates':65536,
        'source_contract_sha256':_digest(payload['contract'])}


def admit(root,space=True):
    import torch
    p=read(HERE/'protocol.json')
    for line in (HERE/'SHA256SUMS').read_text().splitlines():
        expected,name=line.split('  ',1);assert sha(HERE/name)==expected,name
    base=root/'bridge_full_capability_v1'
    for name,expected in p['base_files_sha256'].items():assert sha(base/name)==expected,name
    rt=attach(root);binding=rt.admit(root,space=False)
    source=root/'runs/bridge-full-capability-234100'
    for name,expected in p['source_small_files_sha256'].items():assert sha(source/name)==expected,name
    from bcap_verify import verify
    assert verify(source,full=False)['verified']
    policy=source/'final_policy.pt';payload=torch.load(policy,map_location='cpu',weights_only=False,mmap=True)
    checked=check_policy(payload,p);del payload
    assert not torch.cuda.is_initialized()
    result={'passed':True,'base_admission':binding,'policy_cpu_check':checked,
        'source_policy_size_mtime':[policy.stat().st_size,policy.stat().st_mtime_ns],
        'model_constructed':False,'optimizer_updates':0,'cuda_initialized':False}
    if space:
        used=int(subprocess.check_output(['du','-sB1',str(root)],text=True).split()[0]);reserve=256*1024**2
        assert used+reserve<200_000_000_000 and shutil.disk_usage(root).free>reserve,'Need256MiB trace-output reserve'
        result.update(reserve_bytes=reserve,project_du_bytes=used,quota_scope='conservative project du; not authoritative user quota')
    return result


def replay(b,reference,rows,recorder):
    import numpy as np
    import torch
    from pose_runtime import digest
    from step_training_state import capture_rng,restore_rng
    from rtrace_engine import advance_sample_rng
    model=b['model'];port=b['port'];model.eval();port.eval();draws=[];targets=[];masks=[]
    sourceids=list(range(8))+list(range(128,136));rng_matches=[]
    with torch.inference_mode():
        for sample_index,seed in enumerate([1101,1102,1103]):
            torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);per=[]
            for role,ids in [('train',list(range(8))),('validation',list(range(128,136)))]:
                if role=='validation':advance_sample_rng(next(port.parameters()).device,batches=15)
                selected=[rows[i] for i in ids];x=b['pack'](role,[r['window_index'] for r in selected])
                before=capture_rng(include_cuda=b['cuda'])
                c=model.get_context_vector(x['images'],x['tokens']['input_ids'],x['tokens']['attention_mask'])+port(x['pose'])
                assert digest(capture_rng(include_cuda=b['cuda']))==digest(before),'Evaluation conditioning consumed RNG'
                recorder.begin(seed,role);pred=model.sample(c,x['current'])
                after=capture_rng(include_cuda=b['cuda'])
                if role=='train':
                    restore_rng(before);advance_sample_rng(next(port.parameters()).device,1)
                    assert digest(capture_rng(include_cuda=b['cuda']))==digest(after),'Skipped native RNG pattern differs'
                    restore_rng(after);rng_matches.append(seed)
                actual=pred.cpu().numpy();expected=reference[sample_index,ids]
                assert np.array_equal(actual,expected),('Source native predictions failed exact replay',seed,role,float(abs(actual-expected).max()))
                per.append(actual)
                if sample_index==0:targets.append(x['actions'].cpu().numpy());masks.append(x['masks'].cpu().numpy())
                print('ROTATION_TRACE_REPLAY_EXACT',seed,role,8,flush=True)
            draws.append(np.concatenate(per))
    return np.stack(draws),np.concatenate(targets),np.concatenate(masks),{'skipped_rng_primitive_matches_actual_sample_for_seeds':rng_matches,'source_selection_indices':sourceids}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path('/projects/Zeroshot'))
    ap.add_argument('--check-only',action='store_true');ap.add_argument('--output',type=Path)
    a=ap.parse_args();root=a.root.resolve();binding=admit(root)
    if a.check_only:print('ROTATION_TRACE_CPU_ADMISSION',json.dumps(binding),flush=True);return
    assert os.environ.get('SLURM_JOB_ID') and a.output is not None
    out=a.output.resolve();assert out.parent==(root/'runs').resolve() and out.name.startswith('bridge-rotation-trace-') and not out.exists()
    out.mkdir(parents=True);start=time.monotonic();rt=attach(root)
    import numpy as np
    import torch
    import models
    from bcap_core import identity,load_state
    from rtrace_engine import Recorder,instrument,analyze
    rt.setup();torch.cuda.reset_peak_memory_stats();b=rt.construct(root,binding['base_admission']);rt.attach_data(b)
    source=root/'runs/bridge-full-capability-234100';policy_path=source/'final_policy.pt'
    payload=torch.load(policy_path,map_location='cpu',weights_only=False,mmap=True);check_policy(payload,read(HERE/'protocol.json'))
    assert b['contract']==payload['contract'],'Actual reconstruction config/statistics differ from original training contract'
    load_state(b,payload['state']);del payload
    initial=identity(b);assert initial==read(HERE/'protocol.json')['final_identity']
    assert not b['optimizer'].state
    with np.load(source/'predictions_and_targets.npz',allow_pickle=False) as z:
        reference=z['after'].copy();target_ref=z['targets'].copy();mask_ref=z['masks'].copy()
    rows=rt.read(rt.HERE/'selection.json')['rows'];effective=instrument(rt.HERE/'code/models.py')
    assert hashlib.sha256(effective.encode()).hexdigest()==read(HERE/'protocol.json')['effective_sample_sha256']
    (out/'effective_sample.py').write_text(effective,encoding='utf-8')
    recorder=Recorder(cap=300);original=models.RobotAdapterModel.sample
    models.__dict__['RTRACE_HOOK']=recorder
    exec(compile(effective,'<read-only-original-native-sample-trace>','exec'),models.__dict__)
    models.RobotAdapterModel.sample=models.__dict__['sample']
    try:final,target,mask,rng_checks=replay(b,reference,rows,recorder)
    finally:models.RobotAdapterModel.sample=original
    selected=list(range(8))+list(range(128,136))
    assert np.array_equal(target,target_ref[selected]) and np.array_equal(mask,mask_ref[selected])
    arrays=recorder.arrays();b['check_frozen']()
    assert identity(b)==initial and not b['optimizer'].state
    assert all(p.grad is None for p in list(b['model'].parameters())+list(b['port'].parameters()))
    assert [policy_path.stat().st_size,policy_path.stat().st_mtime_ns]==binding['source_policy_size_mtime']
    np.savez_compressed(out/'trace.npz',targets=target,masks=mask,final=final,reference=reference[:,selected],**arrays)
    calculated=analyze(arrays,target,mask,final,reference[:,selected])
    with np.load(out/'trace.npz',allow_pickle=False) as z:
        replayed=analyze({k:z[k] for k in arrays},z['targets'],z['masks'],z['final'],z['reference'])
    assert replayed==calculated
    report={'execution_complete':True,'protocol_sha256':sha(HERE/'protocol.json'),'source_binding':binding,
        'source_identity_unchanged':initial,'model_adam_gradients_unchanged':True,'rng_checks':rng_checks,
        'analysis':calculated,'elapsed_seconds':time.monotonic()-start,'peak_reserved_gib':torch.cuda.max_memory_reserved()/1024**3,
        'trace_sha256':sha(out/'trace.npz'),'optimizer_updates':0,'automatic_following_action':False,'main_model_replaced':False}
    write(out/'report.json',report)
    import zipfile
    with zipfile.ZipFile(out/'small_results.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
        for name in ['report.json','trace.npz','effective_sample.py']:archive.write(out/name,name)
    print('ROTATION_TRACE_RESULT',json.dumps({'all16windows3seeds_exact':True,'optimizer_updates':0,'source_unchanged':True}),flush=True)
    print('ROTATION_TRACE_SMALL_RESULTS',out/'small_results.zip',sha(out/'small_results.zip'),flush=True)
    print('ROTATION_TRACE_COMPLETE',out/'report.json',flush=True)


if __name__=='__main__':main()

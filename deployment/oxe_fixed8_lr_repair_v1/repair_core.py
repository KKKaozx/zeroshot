"""Use archived actual cluster LR bytes; retain exact guards and all source weights."""
import hashlib
import inspect
import json
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def modules(root):
    base=root/'oxe_fixed8_learnability_v1';sys.path.insert(0,str(base))
    import qualification as q
    assert q.HERE==base.resolve(),'A different diagnostic qualification module was imported'
    return q


def fixed_sources(q,engine,verifier):
    contract=read(HERE/'repair_contract.json');sources={}
    for name,fn in [('load_source',q.load_source),('run',engine.run),('verify',verifier.verify)]:
        source=inspect.getsource(fn)
        assert hashlib.sha256(source.encode()).hexdigest()==contract['original_function_sha256'][name],name
        if name in ['load_source','verify']:
            old="with np.load(HERE/'reference_lrs.npz') as z:"
            assert source.count(old)==1;source=source.replace(old,'with np.load(REPAIRED_LRS) as z:')
        else:
            old="outputs={};report={'execution_complete':False,"
            assert source.count(old)==1
            source=source.replace(old,"outputs={};report={'lr_reference_repair':LR_REFERENCE_REPAIR,'execution_complete':False,")
            old="policy={'format':'current-oxe-fixed8-diagnostic-policy-v1',"
            assert source.count(old)==1
            source=source.replace(old,"policy={'lr_reference_repair':LR_REFERENCE_REPAIR,'format':'current-oxe-fixed8-diagnostic-policy-v1',")
        compile(source,'<fixed8-archived-lr-'+name+'>','exec');sources[name]=source
        if 'repaired_function_sha256' in contract:
            assert hashlib.sha256(source.encode()).hexdigest()==contract['repaired_function_sha256'][name],name
    return sources


def admit(root,*,space=True,source_schedule=True):
    contract=read(HERE/'repair_contract.json');base=root/'oxe_fixed8_learnability_v1'
    for line in (HERE/'SHA256SUMS').read_text().splitlines():
        digest,name=line.split('  ',1);assert sha(HERE/name)==digest,name
    for name,digest in contract['original_bundle_files'].items():assert sha(base/name)==digest,name
    q=modules(root);bound=q.admit(root,space=space)
    log=root/'logs/oxe-fixed8-233339.out';text=log.read_text(errors='replace')
    assert 'load_source' in text and 'schedule.rates(10240+i)' in text and 'AssertionError' in text
    assert 'OXE_FIXED8_UPDATE ' not in text and 'OXE_FIXED8_GENERATION ' not in text,'Do not repeat a run that already trained'
    result={'format':'fixed8-lr-archive-repair-binding-v1','repair_contract_sha256':sha(HERE/'repair_contract.json'),
        'failed_job':233339,'failed_log_sha256':sha(log),'original_source_binding':bound,
        'replacement_reference_sha256':sha(HERE/'reference_lrs_from_cluster.npz'),
        'optimizer_updates_before_failure':0,'weights_and_optimizer_changed_by_repair':False,
        'identity_tolerances_changed':False,'automatic_submission':False}
    with np.load(base/'reference_lrs.npz') as old,np.load(HERE/'reference_lrs_from_cluster.npz') as new:
        a=old['lrs'];b=new['lrs'];different=np.flatnonzero((a!=b).any(1))
        assert a.shape==b.shape==(512,3)
        assert (10241+different).tolist()==contract['different_update_numbers']
        assert float(np.abs(a-b).max())==contract['max_absolute_difference']
    if source_schedule:
        # Read only existing checkpoint metadata on CPU before ANY model reconstruction/GPU allocation.
        import torch
        q.attach(root)
        from completed_policy_export import contract_sha
        from step_training_state import UpdateSchedule
        payload=torch.load(root/'runs/oxe-pose-pair-232587/real_pose/latest_state.pt',
            map_location='cpu',weights_only=False,mmap=True)
        assert payload['format']=='current-pose-bounded-formal-state-v1' and payload['plan']['completed_updates']==10240
        assert payload['contract_sha256']==contract_sha(payload['contract'])==q.read(base/'protocol.json')['source_identity']['contract_sha256']
        schedule=UpdateSchedule(**payload['schedule']);assert schedule.contract==contract['expected_schedule']
        with np.load(HERE/'reference_lrs_from_cluster.npz') as z:
            differences=[{'update':10241+i,'actual':schedule.rates(10240+i),'reference':z['lrs'][i].tolist()}
                for i in range(512) if schedule.rates(10240+i)!=z['lrs'][i].tolist()]
        assert not differences,('Actual CPU checkpoint schedule differs; stop before submission',differences[:8])
        assert [g['lr'] for g in payload['optimizer']['param_groups']]==schedule.rates(10240)
        result.update(actual_checkpoint_schedule_checked_on_cpu=True,all512_steps_exact=True,
            schedule=payload['schedule'],cuda_initialized_by_check=torch.cuda.is_initialized())
        assert not torch.cuda.is_initialized(),'Run this metadata check before CUDA initialization'
    path=HERE/'source_repair_binding.json'
    if path.exists():assert read(path)==result,'Immutable actual repair binding differs'
    else:path.write_bytes((json.dumps(result,indent=2)+'\n').encode())
    return result


def install(root,binding):
    q=modules(root);q.attach(root)
    import engine
    import verify_output
    sources=fixed_sources(q,engine,verify_output)
    replacement=HERE/'reference_lrs_from_cluster.npz'
    for name,module in [('load_source',q),('run',engine),('verify',verify_output)]:
        module.__dict__.update(REPAIRED_LRS=replacement,LR_REFERENCE_REPAIR=binding)
        exec(compile(sources[name],'<fixed8-archived-lr-'+name+'>','exec'),module.__dict__)
    return q,engine,verify_output,sources


def verify_extra(output,binding):
    import torch
    report=read(output/'report.json');assert report['execution_complete'] and report['lr_reference_repair']==binding
    policy=torch.load(output/'diagnostic_policy.pt',map_location='cpu',weights_only=False,mmap=True)
    assert policy['lr_reference_repair']==binding
    return {'repair_binding_verified':True,'optimizer_budget':512,'extra_training_or_seeds':False}

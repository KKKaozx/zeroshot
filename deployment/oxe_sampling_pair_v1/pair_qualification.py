"""Actual source checkpoint/schedule/data-order checks BEFORE allocating a GPU."""
import hashlib
import inspect
import json
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def attach(root):
    sys.path.insert(0,str(root/'oxe_fixed8_learnability_v1'))
    import qualification as q
    assert q.HERE==(root/'oxe_fixed8_learnability_v1').resolve()
    _,pg=q.attach(root)
    return q,pg


def install_loader(q):
    p=read(HERE/'protocol.json');s=inspect.getsource(q.load_source)
    assert hashlib.sha256(s.encode()).hexdigest()==p['original_loader_sha256']
    old="with np.load(HERE/'reference_lrs.npz') as z:";assert s.count(old)==1
    s=s.replace(old,'with np.load(REPAIRED_LRS) as z:')
    assert hashlib.sha256(s.encode()).hexdigest()==p['qualified_corrected_loader_sha256']
    q.__dict__['REPAIRED_LRS']=HERE/'plans.npz'
    exec(compile(s,'<previously-qualified-corrected-source-loader>','exec'),q.__dict__)
    return q.load_source,s


def check_checkpoint(payload,protocol,plans):
    import torch
    from completed_policy_export import contract_sha,model_sha
    from pose_runtime import digest
    from step_training_state import UpdateSchedule,_digest
    assert payload['format']=='current-pose-bounded-formal-state-v1'
    assert payload['contract_sha256']==contract_sha(payload['contract'])==protocol['source_identity']['contract_sha256']
    assert model_sha(payload['base'])==protocol['source_identity']['base_learned_sha256']
    assert digest(payload['pose_port'])==protocol['source_identity']['port_sha256']
    rng=payload['rng'];runtime=protocol['expected_runtime']
    assert str(torch.__version__)==runtime['torch'] and torch.version.cuda==runtime['cuda_runtime']
    assert rng['include_cuda'] is True and len(rng['torch_cuda'])==1
    assert rng['environment']['torch']==runtime['torch'] and rng['environment']['cuda_version']==runtime['cuda_runtime']
    assert rng['environment']['cuda_devices']==1
    plan=payload['plan'];assert plan['completed_updates']==10240
    assert plan['contract']==protocol['original_full_plan_contract']
    assert plan['contract']['indices_sha256']==_digest(list(range(145812))),'Retained order is positional; require identity training-index mapping'
    assert plan['cursor']==81920 and len(plan['order'])==145812 and plan['pass_index']==0
    # This4096suffix remains inside the retained first pass: no new sampling RNG call is needed.
    stop=plan['cursor']+8*protocol['updates_each'];assert stop<=len(plan['order'])
    actual=plan['order'][plan['cursor']:stop].cpu().numpy().reshape(protocol['updates_each'],8)
    assert np.array_equal(actual,plans['replacement']),'Archived control suffix does not match ACTUAL source plan'
    schedule=UpdateSchedule(**payload['schedule']);assert schedule.contract==protocol['schedule_contract']
    mismatch=[{'update':10241+i,'actual':schedule.rates(10240+i),'reference':plans['lrs'][i].tolist()}
        for i in range(protocol['updates_each']) if schedule.rates(10240+i)!=plans['lrs'][i].tolist()]
    assert not mismatch,('Source runtime LR differs before submission',mismatch[:8])
    assert [g['lr'] for g in payload['optimizer']['param_groups']]==schedule.rates(10240)
    assert not torch.cuda.is_initialized(),'Metadata check must precede CUDA initialization'
    return {'source_weights_port_contract_checked_on_cpu':True,'all4096_lr_steps_exact':True,
        'actual_source_plan_control_suffix_exact':True,'cuda_initialized':False}


def admit(root,space=True):
    p=read(HERE/'protocol.json')
    for line in (HERE/'SHA256SUMS').read_text().splitlines():
        expected,name=line.split('  ',1);assert sha(HERE/name)==expected,name
    for relative,expected in p['inherited_files'].items():assert sha(root/relative)==expected,relative
    q,pg=attach(root);source_binding=q.admit(root,space=False)
    import torch
    from oxe_nominal_target_store import NominalTargetStore
    store=NominalTargetStore(root/'oxe_dino_main_preflight_v1/inputs/train')
    assert store.binding==p['target_binding'] and len(store)==145812
    codes=np.asarray([p['sources'].index(store.episodes[int(e)]['source']) for e in store.windows[:,0]],dtype=np.int8)
    with np.load(HERE/'plans.npz') as z:plans={k:z[k] for k in z.files}
    assert np.array_equal(codes,plans['source_codes'])
    assert plans['replacement'].shape==plans['cycle'].shape==(p['updates_each'],8)
    for role in ['replacement','cycle']:
        assert plans[role].dtype==np.int64 and plans[role].min()>=0 and plans[role].max()<len(store)
    assert np.array_equal(codes[plans['replacement']],codes[plans['cycle']])
    for code in range(4):
        chosen=plans['cycle'].reshape(-1)[codes[plans['cycle'].reshape(-1)]==code]
        assert len(chosen)<=int((codes==code).sum()) and len(np.unique(chosen))==len(chosen)
    for role in ['replacement','cycle']:
        assert np.array_equal(store.masks[plans[role]].sum((1,2)),plans[role+'_valid_dimensions'])
    source=root/'runs/oxe-pose-pair-232587/real_pose/latest_state.pt'
    payload=torch.load(source,map_location='cpu',weights_only=False,mmap=True)
    checks=check_checkpoint(payload,p,plans)
    bound={'format':'sampling-pair-actual-source-binding-v1','protocol_sha256':sha(HERE/'protocol.json'),
        'source_binding':source_binding,'plans_sha256':sha(HERE/'plans.npz'),'cpu_checks':checks,
        'source_file_size_mtime':[source.stat().st_size,source.stat().st_mtime_ns],
        'source_weights_are_fixed8_memorized_policy':False,'training_submitted':False}
    path=HERE/'source_pair_binding.json'
    if path.exists():assert read(path)==bound,'Original source checkpoint or binding changed'
    else:path.write_bytes((json.dumps(bound,indent=2)+'\n').encode())
    if space:
        used=int(subprocess.check_output(['du','-sB1',str(root)],text=True).split()[0]);reserve=6*1024**3
        assert used+reserve<200_000_000_000 and shutil.disk_usage(root).free>reserve,'Need6GiB full-state/atomic-save reserve'
    return bound

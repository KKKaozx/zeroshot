"""Own explicit augmentation checkpoint; never relabel a native or preflight file."""
import os
from pathlib import Path
import tempfile
import torch

FORMAT='current-pose-bounded-formal-state-v1'


def save(path,base,port,optimizer,plan,schedule,contract,learned_state,extra,include_cuda=True):
    from step_training_state import _snapshot,capture_rng
    from completed_policy_export import contract_sha
    assert 8192<plan.completed_updates<=10240 and schedule.contract['total_updates']==65536
    assert [g['lr'] for g in optimizer.param_groups]==schedule.rates(plan.completed_updates)
    p=_snapshot({'format':FORMAT,'contract':contract,'contract_sha256':contract_sha(contract),
        'base':learned_state(base),'pose_port':port.state_dict(),'optimizer':optimizer.state_dict(),
        'plan':plan.state_dict(),'schedule':schedule.contract,'rng':capture_rng(include_cuda=include_cuda),'extra':extra})
    path=Path(path);fd,tmp=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as f:torch.save(p,f);f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if Path(tmp).exists():Path(tmp).unlink()


def restore(path,base,port,optimizer,plan,schedule,contract,learned_state):
    from step_training_state import _validate_rng,restore_rng
    from completed_policy_export import contract_sha
    from pose_runtime import check_moments
    p=torch.load(path,map_location='cpu',weights_only=False)
    assert p['format']==FORMAT and p['contract']==contract and p['contract_sha256']==contract_sha(contract)
    assert p['schedule']==schedule.contract and 8192<p['plan']['completed_updates']<=10240
    plan.validate_state(p['plan']);_validate_rng(p['rng'])
    expected=learned_state(base)
    assert set(expected)==set(p['base']) and all(expected[k].shape==p['base'][k].shape and expected[k].dtype==p['base'][k].dtype for k in expected)
    assert set(p['pose_port'])=={'weight'} and p['pose_port']['weight'].shape==port.weight.shape
    loaded=base.load_state_dict(p['base'],strict=False)
    assert not loaded.unexpected_keys and not set(expected).intersection(loaded.missing_keys)
    port.load_state_dict(p['pose_port'],strict=True);optimizer.load_state_dict(p['optimizer']);check_moments(optimizer)
    plan.load_state_dict(p['plan'])
    assert [g['lr'] for g in optimizer.param_groups]==schedule.rates(plan.completed_updates)
    restore_rng(p['rng']);return p['extra']

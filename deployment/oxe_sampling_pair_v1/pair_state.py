"""Explicit paired-sampling checkpoint; never impersonates the old source plan."""
import os
from pathlib import Path
import tempfile
import torch

FORMAT='oxe-sampling-intervention-state-v1'


def validate_trace(payload,role,plans,control=None):
    from pair_steps import check_paired_stochastic
    import numpy as np
    assert len(payload['history'])==len(payload['noise'])==payload['counter']
    for i,(row,event) in enumerate(zip(payload['history'],payload['noise'])):
        assert row['local_update']==i+1 and row['source_schedule_update']==10241+i
        assert row['indices']==plans[role][i].tolist() and row['lr_used']==plans['lrs'][i].tolist()
        assert row['valid_dimensions']==plans[role+'_valid_dimensions'][i].tolist() and row['token_shape']==[8,77]
        assert event['step']==10241+i and np.isfinite([row[k] for k in ['loss','pose_loss','gripper_loss','gradient_norm']]).all()
        if control is not None:check_paired_stochastic(row,event,control['history'][i],control['noise'][i])


def save(path,bundle,role,done,history,noise,contract):
    from step_training_state import _snapshot,capture_rng
    from completed_policy_export import contract_sha
    assert len(history)==len(noise)==done and 0<done<=contract['updates_each']
    payload=_snapshot({'format':FORMAT,'contract':contract,'contract_sha256':contract_sha(contract),
        'role':role,'counter':done,'base':bundle['api'].trainable_state_dict(bundle['model']),
        'pose_port':bundle['port'].state_dict(),'optimizer':bundle['optimizer'].state_dict(),
        'schedule':bundle['schedule'].contract,'rng':capture_rng(include_cuda=bundle['source']['rng']['include_cuda']),
        'history':history,'noise':noise,'training_job_ids':bundle.get('training_job_ids',[])})
    path=Path(path);fd,tmp=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as f:torch.save(payload,f);f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if Path(tmp).exists():Path(tmp).unlink()


def restore(path,bundle,role,contract,plans,control=None):
    from step_training_state import _validate_rng,restore_rng
    from completed_policy_export import contract_sha
    from pose_runtime import check_moments
    p=torch.load(path,map_location='cpu',weights_only=False)
    assert p['format']==FORMAT and p['contract']==contract and p['contract_sha256']==contract_sha(contract)
    assert p['role']==role and 0<p['counter']<=contract['updates_each']
    assert p['schedule']==bundle['schedule'].contract and len(p['history'])==len(p['noise'])==p['counter']
    validate_trace(p,role,plans,control)
    expected=bundle['api'].trainable_state_dict(bundle['model']);assert set(p['base'])==set(expected)
    assert all(p['base'][k].shape==expected[k].shape and p['base'][k].dtype==expected[k].dtype for k in expected)
    _validate_rng(p['rng'])
    loaded=bundle['model'].load_state_dict(p['base'],strict=False)
    assert not loaded.unexpected_keys and not set(expected).intersection(loaded.missing_keys)
    bundle['port'].load_state_dict(p['pose_port'],strict=True);bundle['optimizer'].load_state_dict(p['optimizer']);check_moments(bundle['optimizer'])
    assert [g['lr'] for g in bundle['optimizer'].param_groups]==bundle['schedule'].rates(10240+p['counter'])
    bundle['training_job_ids']=p.get('training_job_ids',[])+[j for j in bundle.get('training_job_ids',[]) if j not in p.get('training_job_ids',[])]
    restore_rng(p['rng']);return p['counter'],p['history'],p['noise']

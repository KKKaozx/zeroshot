"""Explicit pose-port augmentation state, separate from the original model contract."""
import copy
import hashlib
import io
import json
import torch
from torch import nn


def digest(value):
    h=hashlib.sha256()
    def visit(v):
        if isinstance(v,torch.Tensor):
            x=v.detach().cpu().contiguous();h.update(json.dumps([str(x.dtype),list(x.shape)]).encode())
            h.update(x.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(v,dict):
            for k in sorted(v,key=lambda k:(type(k).__name__,str(k))):h.update(repr(k).encode());visit(v[k])
        elif isinstance(v,(list,tuple)):
            h.update(type(v).__name__.encode())
            for x in v:visit(x)
        elif hasattr(v,'dtype') and hasattr(v,'tobytes'):
            h.update(str(v.dtype).encode());h.update(str(v.shape).encode());h.update(v.tobytes())
        else:h.update(repr(v).encode())
    visit(value);return h.hexdigest()


def zero_port(device,width=1024):
    devices=[torch.cuda.current_device()] if torch.device(device).type=='cuda' else []
    cpu=torch.get_rng_state().clone();cuda=torch.cuda.get_rng_state_all() if devices else []
    with torch.random.fork_rng(devices=devices):
        port=nn.Linear(18,width,bias=False).to(device);nn.init.zeros_(port.weight)
    assert torch.equal(cpu,torch.get_rng_state())
    if devices:assert all(torch.equal(x,y) for x,y in zip(cuda,torch.cuda.get_rng_state_all()))
    assert not torch.count_nonzero(port.weight)
    return port


def check_moments(optimizer):
    for group in optimizer.param_groups:
        for p in group['params']:
            for name,value in optimizer.state.get(p,{}).items():
                if isinstance(value,torch.Tensor) and name!='step':assert value.device==p.device,(name,value.device,p.device)


def snapshot(base,port,optimizer,plan,schedule,contract,learned_state,include_cuda):
    from step_training_state import _snapshot,capture_rng
    from completed_policy_export import contract_sha
    payload=_snapshot({'format':'current-pose-disposable-preflight-state-v1','contract':contract,
        'contract_sha256':contract_sha(contract),'base':learned_state(base),'pose_port':port.state_dict(),
        'optimizer':optimizer.state_dict(),'plan':plan.state_dict(),'schedule':schedule.contract,
        'rng':capture_rng(include_cuda=include_cuda)})
    stream=io.BytesIO();torch.save(payload,stream);return stream


def restore(stream,base,port,optimizer,plan,schedule,contract,learned_state):
    from step_training_state import restore_rng
    from completed_policy_export import contract_sha
    stream.seek(0);payload=torch.load(stream,map_location='cpu',weights_only=False)
    assert payload['format']=='current-pose-disposable-preflight-state-v1'
    assert payload['contract']==contract and payload['contract_sha256']==contract_sha(contract)
    assert payload['schedule']==schedule.contract
    plan.validate_state(payload['plan'])
    expected=learned_state(base)
    assert set(expected)==set(payload['base'])
    assert all(expected[k].shape==payload['base'][k].shape and expected[k].dtype==payload['base'][k].dtype for k in expected)
    loaded=base.load_state_dict(payload['base'],strict=False)
    assert not loaded.unexpected_keys and not set(expected).intersection(loaded.missing_keys)
    port.load_state_dict(payload['pose_port'],strict=True);optimizer.load_state_dict(payload['optimizer'])
    check_moments(optimizer);plan.load_state_dict(payload['plan'])
    assert [g['lr'] for g in optimizer.param_groups]==schedule.rates(plan.completed_updates)
    restore_rng(payload['rng'])


def current_state(base,port,optimizer,plan,schedule,learned_state,include_cuda):
    from step_training_state import capture_rng
    return {'base':digest(learned_state(base)),'port':digest(port.state_dict()),'optimizer':digest(optimizer.state_dict()),
        'plan':digest(plan.state_dict()),'schedule':digest(schedule.contract),'rng':digest(capture_rng(include_cuda=include_cuda))}


class NoiseLog:
    def __init__(self):self.events=[];self.current=None
    def capture_noise(self,model,noise,timesteps,noisy_actions):
        assert self.current is not None and model.diffusion_prediction_type=='sample'
        self.events.append({'step':self.current,'noise':digest(noise),'timesteps':digest(timesteps),'noisy_actions':digest(noisy_actions)})

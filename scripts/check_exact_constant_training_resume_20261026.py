"""New training-resume boundary: synthetic checkpoint, exact continuation."""
import copy
import hashlib
import json
from pathlib import Path
import random
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
REPAIR=ROOT/'deployment/oxe_resume_and_libero_dependency_repair_v1'
sys.path[:0]=[str(REPAIR),str(ROOT/'code')]
import numpy as np
import torch
from torch import nn
from resume_constants import training_constant_constructor
from frozen_profile_restore import metadata
from step_training_runner import frozen_digest
from step_training_state import ResumableBatchPlan,UpdateSchedule,save_checkpoint,restore_checkpoint


class Tiny(nn.Module):
    perturb=False
    def __init__(self, config):
        super().__init__()
        self.encoder=nn.Linear(2,2)
        self.encoder.requires_grad_(False)
        self.adapter=nn.Linear(2,1)
        v=torch.tensor([.125,.25])
        self.register_buffer('posterior_mean_x0',v.clone())
        self.register_buffer('posterior_mean_xt',v.clone())
        if self.perturb:
            self.posterior_mean_x0.add_(2**-24)
            self.posterior_mean_xt.add_(2**-24)
    def forward(self,x):
        return self.adapter(self.encoder(x)+self.posterior_mean_x0+self.posterior_mean_xt)


def learned(m):return {n:v for n,v in m.state_dict().items() if n.startswith('adapter.')}
def seed():random.seed(43);np.random.seed(43);torch.manual_seed(43)
def plan():return ResumableBatchPlan(list(range(8)),batch_size=2,total_updates=6,seed=43,dataset_identity='synthetic')
def schedule():return UpdateSchedule(total_updates=6,base_lrs=[.002],kind='cosine',warmup_updates=1)
def advance(m,opt,p,s):
    indices=p.begin_batch()
    x=torch.randn(len(indices),2)+random.random()+float(np.random.rand())
    loss=m(x).square().mean();opt.zero_grad();loss.backward();opt.step()
    p.commit_batch();s.apply(opt,p.completed_updates)
def equal(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def main():
    config={'model':{'name':'synthetic'},'action':{}}
    seed();m=Tiny(config);s=schedule();p=plan();opt=torch.optim.AdamW(m.adapter.parameters(),lr=.002);s.apply(opt,0)
    frozen={n:metadata(v) for n,v in m.state_dict().items() if n not in learned(m)}
    contract={'format':'synthetic-original-contract','config':config,'frozen_state_sha256':frozen_digest(m,learned(m)),
        'tokenizer':{'fixture':True},'source_code_sha256':{},'total_updates':6}
    profile={'legacy_frozen_sha256':contract['frozen_state_sha256'],'config':config,
        'tokenizer_before_model':contract['tokenizer'],'tokenizer_after_model':contract['tokenizer'],
        'expected_tokenizer':contract['tokenizer'],'core_source_sha256':{},
        'tensors':frozen,'key_order':list(frozen),'schedule_values':{
            'posterior_mean_x0':m.posterior_mean_x0.tolist(),'posterior_mean_xt':m.posterior_mean_xt.tolist()}}
    checks={}
    with tempfile.TemporaryDirectory(dir=ROOT/'.tmp',prefix='constant_resume_') as tmp:
        out=Path(tmp);profile_path=out/'profile.json';profile_path.write_text(json.dumps(profile))
        for _ in range(3):advance(m,opt,p,s)
        save_checkpoint(out/'state.pt',model_state=learned(m),optimizer=opt,plan=p,schedule=s,contract=contract,
            include_cuda=False,extra={'completed':3})
        saved_before=hashlib.sha256((out/'state.pt').read_bytes()).hexdigest()
        for _ in range(3):advance(m,opt,p,s)
        expected_model=copy.deepcopy(m.state_dict());expected_opt=copy.deepcopy(opt.state_dict())
        expected_next=(torch.rand(3),random.random(),float(np.random.rand()))
        original=Tiny.__init__;Tiny.perturb=True;seed();events=[]
        with training_constant_constructor(Tiny,contract=contract,profile_path=profile_path,learned_state=learned,on_restored=events.append):
            restored=Tiny(config)
        checks['original_constructor_restored_after_scope']=Tiny.__init__ is original
        checks['only_two_constants_written_and_exact_original_guard_passed']=(
            events[0]['restored_buffer_names']==['posterior_mean_x0','posterior_mean_xt']
            and frozen_digest(restored,learned(restored))==contract['frozen_state_sha256']
            and not events[0]['encoder_parameters_written'] and not events[0]['learned_parameters_written'])
        rp=plan();rs=schedule();ro=torch.optim.AdamW(restored.adapter.parameters(),lr=.002);rs.apply(ro,0)
        extra=restore_checkpoint(out/'state.pt',model_state_loader=lambda state:restored.load_state_dict(state,strict=False),
            optimizer=ro,plan=rp,schedule=rs,contract=contract)
        checks['original_resume_keeps_completed_counter_and_learning_rate']=rp.completed_updates==3 and extra=={'completed':3} and [g['lr'] for g in ro.param_groups]==rs.rates(3)
        for _ in range(3):advance(restored,ro,rp,rs)
        checks['exact_next_training_model_and_optimizer_match_uninterrupted']=(equal(expected_model,restored.state_dict()) and equal(expected_opt,ro.state_dict()))
        checks['exact_rng_continuation_match_uninterrupted']=equal(expected_next,(torch.rand(3),random.random(),float(np.random.rand())))
        checks['input_checkpoint_bytes_unchanged']=hashlib.sha256((out/'state.pt').read_bytes()).hexdigest()==saved_before
        called=[]
        try:
            restore_checkpoint(out/'state.pt',model_state_loader=called.append,optimizer=ro,plan=rp,schedule=rs,
                contract=dict(contract,total_updates=7))
            checks['original_contract_guard_still_rejects_changed_budget']=False
        except ValueError:
            checks['original_contract_guard_still_rejects_changed_budget']=not called
        Tiny.perturb=False
    report={'passed':all(checks.values()),'checks':checks,'synthetic_optimizer_updates':9,
        'actual_model_forwards':0,'actual_training_updates':0,'actual_libero_episodes':0,
        'original_training_files_changed':False}
    dest=ROOT/'reports/oxe_resume_and_libero_repair_20261026';dest.mkdir(exist_ok=True)
    (dest/'checks.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
    assert report['passed']


if __name__=='__main__':main()

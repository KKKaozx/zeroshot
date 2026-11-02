"""Checkpoint integration fixture: synthetic CPU optimization, no robot model/data."""
import copy
import os
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from step_training_state import (ResumableBatchPlan, UpdateSchedule, capture_rng,
                                 restore_rng, save_checkpoint, restore_checkpoint)

USE_CUDA = sys.argv[1:] == ['--cuda']
if sys.argv[1:] and not USE_CUDA:raise ValueError('Only --cuda is supported')
DEVICE = torch.device('cuda:0' if USE_CUDA else 'cpu')
OUT=ROOT/('reports/step_training_state_cuda_20261020' if USE_CUDA else 'reports/step_training_state_20261020')
IDS=[3,7,11,18,23,30,41]
TOTAL=11
CONTRACT={'purpose':'synthetic_cpu_resume_fixture_not_robot_training',
          'dataset':'seven_fixed_synthetic_indices_v1','backbone':'fixed_tiny_linear_v1',
          'optimizer':'AdamW','weight_decay':.01,'condition':'no_augmentation',
          'workers':0,'gradient_accumulation':1}


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)


def identical(a,b):
    if isinstance(a,torch.Tensor):
        return isinstance(b,torch.Tensor) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(identical(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(identical(x,y) for x,y in zip(a,b))
    return a==b


class Fixture(nn.Module):
    def __init__(self):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(104)
            self.encoder=nn.Linear(4,4).requires_grad_(False)
        self.head=nn.Sequential(nn.Linear(4,5),nn.Tanh(),nn.Dropout(.25),nn.Linear(5,3))

    def forward(self,x):return self.head(self.encoder(x))


def factory(mode,kind,seed,total=TOTAL,indices=None,batch=3):
    model=Fixture().to(DEVICE)
    optimizer=torch.optim.AdamW([
        {'params':[p for n,p in model.head.named_parameters() if n.endswith('weight')],'lr':.01},
        {'params':[p for n,p in model.head.named_parameters() if n.endswith('bias')],'lr':.004}],weight_decay=.01)
    plan=ResumableBatchPlan(IDS if indices is None else indices,batch_size=batch,total_updates=total,
            seed=seed,dataset_identity='synthetic-seven-index-identity',mode=mode,
            weights=[1.,2.,1.,4.,2.,1.,3.] if mode=='weighted_replacement' else None)
    schedule=UpdateSchedule(total_updates=total,base_lrs=[.01,.004],kind=kind,
                            warmup_updates=2 if kind=='cosine' else 0)
    schedule.apply(optimizer,0)
    return model,optimizer,plan,schedule


def run(state,until,*,isolated_metrics=False):
    model,opt,plan,schedule=state
    logs=[]
    model.train()
    while plan.completed_updates<until:
        if isolated_metrics:
            saved=capture_rng(include_cuda=USE_CUDA)
            random.random();np.random.rand(7);torch.randn(13);torch.randn(13,device=DEVICE)
            restore_rng(saved)
        ids=plan.begin_batch()
        t=torch.randint(0,100,(len(ids),),device=DEVICE)
        noise=torch.randn(len(ids),3,device=DEVICE)
        python_draw=random.random();numpy_draw=float(np.random.rand())
        x=torch.tensor([[i/50.,(i%3)/3.,(i%7)/7.,1.] for i in ids],device=DEVICE)
        x=x+python_draw*.03+numpy_draw*.02+t[:,None]/1000.
        pred=model(x)
        target=noise*.2+x[:,:3]
        loss=(pred-target).square().mean()
        logs.append({'ids':ids,'pass_index':plan.pass_index,'timesteps':t.cpu().clone(),
                     'noise':noise.cpu().clone(),'python':python_draw,'numpy':numpy_draw,
                     'learning_rates':[g['lr'] for g in opt.param_groups],'loss':float(loss.detach())})
        opt.zero_grad(set_to_none=True)
        loss.backward();opt.step();plan.commit_batch()
        schedule.apply(opt,plan.completed_updates)
    return logs


def final_state(state):
    model,opt,plan,schedule=state
    return copy.deepcopy({'head':model.head.state_dict(),'encoder':model.encoder.state_dict(),
                          'optimizer':opt.state_dict(),'plan':plan.state_dict(),
                          'rng':capture_rng(include_cuda=USE_CUDA),'rates':schedule.rates(plan.completed_updates)})


def save(path,state):
    model,opt,plan,schedule=state
    before=capture_rng(include_cuda=USE_CUDA)
    save_checkpoint(path,model_state=model.head.state_dict(),optimizer=opt,plan=plan,schedule=schedule,
                    contract=CONTRACT,include_cuda=USE_CUDA,extra={'evaluation_cursor':0})
    assert identical(before,capture_rng(include_cuda=USE_CUDA)), 'Checkpoint writing consumed training RNG'


def restore(path,state,contract=CONTRACT):
    model,opt,plan,schedule=state
    return restore_checkpoint(path,model_state_loader=model.head.load_state_dict,
                              optimizer=opt,plan=plan,schedule=schedule,contract=contract)


def rejected(action):
    try:action()
    except (ValueError,RuntimeError,StopIteration):return
    raise AssertionError('Invalid contract or transition was accepted')


def main():
    if USE_CUDA and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no passing evidence written')
    if USE_CUDA:
        torch.cuda.set_device(DEVICE)
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(DEVICE)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if (OUT/'report.json').exists():raise FileExistsError('Completed evidence must not be overwritten')
    OUT.mkdir(exist_ok=True)
    results=[]
    for mode in ['shuffle','weighted_replacement']:
        for kind in ['constant','cosine']:
            for seed in [42,43]:
                seed_all(seed)
                baseline=factory(mode,kind,seed)
                frozen=copy.deepcopy(baseline[0].encoder.state_dict())
                expected_log=run(baseline,TOTAL)
                expected_state=final_state(baseline)
                assert identical(frozen,expected_state['encoder'])
                assert all(p.grad is None for p in baseline[0].encoder.parameters())
                rejected(baseline[2].begin_batch)
                # Cuts include a short-tail boundary, a complete pass and several passes.
                for cuts in [[1],[2],[3],[4],[8],[2,3,7,10]]:
                    seed_all(seed)
                    state=factory(mode,kind,seed)
                    logs=[]
                    filename=f'{mode}_{kind}_{seed}_'+('_'.join(map(str,cuts)))+'.pt'
                    path=OUT/filename
                    for cut in cuts:
                        logs.extend(run(state,cut))
                        save(path,state)
                        seed_all(999)  # Construction and unrelated work must not leak into resumed draws.
                        replacement=factory(mode,kind,seed)
                        random.random();np.random.rand(11);torch.randn(9);torch.randn(9,device=DEVICE)
                        assert restore(path,replacement)=={'evaluation_cursor':0}
                        state=replacement
                    logs.extend(run(state,TOTAL))
                    assert identical(logs,expected_log),(mode,kind,seed,cuts,'training trace differs')
                    assert identical(final_state(state),expected_state),(mode,kind,seed,cuts,'final state differs')
                    # Synthetic head only: the frozen encoder is excluded from the saved model state.
                    stored=torch.load(path,map_location='cpu',weights_only=False)
                    assert set(stored['model'])==set(state[0].head.state_dict())
                    results.append({'mode':mode,'schedule':kind,'seed':seed,'cuts':cuts,
                                    'sample_noise_loss_optimizer_parameters_rng_exact':True})
                seed_all(seed)
                monitored=factory(mode,kind,seed)
                assert identical(run(monitored,TOTAL,isolated_metrics=True),expected_log)
                assert identical(final_state(monitored),expected_state)

    seed_all(42)
    state=factory('shuffle','cosine',42)
    run(state,2)
    path=OUT/'contract_rejection.pt'
    save(path,state)
    rejection_checks=[]
    for label,replacement,contract in [
        ('total_budget',factory('shuffle','cosine',42,total=12),CONTRACT),
        ('batch_size',factory('shuffle','cosine',42,batch=2),CONTRACT),
        ('index_order',factory('shuffle','cosine',42,indices=list(reversed(IDS))),CONTRACT),
        ('sampling_weights',factory('weighted_replacement','cosine',42),CONTRACT),
        ('schedule',factory('shuffle','constant',42),CONTRACT),
        ('language_condition',factory('shuffle','cosine',42),{**CONTRACT,'condition':'combined'}),
    ]:
        model_before=copy.deepcopy(replacement[0].head.state_dict())
        rng_before=capture_rng(include_cuda=USE_CUDA)
        rejected(lambda:restore(path,replacement,contract))
        assert identical(model_before,replacement[0].head.state_dict())
        assert identical(rng_before,capture_rng(include_cuda=USE_CUDA))
        rejection_checks.append(label)
    state[2].begin_batch()
    rejected(lambda:save(OUT/'pending_must_not_exist.pt',state))
    assert not (OUT/'pending_must_not_exist.pt').exists()
    rejected(state[2].begin_batch)
    state[2].abort_batch()
    before=state[2].state_dict()
    ids=state[2].begin_batch();state[2].abort_batch()
    assert ids==state[2].begin_batch()
    state[2].abort_batch()
    assert identical(before,state[2].state_dict())
    bad=copy.deepcopy(before);bad['cursor']=1
    rejected(lambda:state[2].load_state_dict(bad))

    # Sensitivity control: deliberately discard restored RNG; the fixture must detect it.
    seed_all(42)
    good=factory('shuffle','cosine',42);restore(path,good)
    good_trace=run(good,3)
    other=factory('shuffle','cosine',42);restore(path,other)
    seed_all(999)
    bad_trace=run(other,3)
    assert not identical(good_trace,bad_trace)

    previous=json.loads((ROOT/'reports/oxe_pose_source_support_20261020/report.json').read_text())
    for name,digest in previous['production_source_sha256'].items():
        assert hashlib.sha256((ROOT/'code'/name).read_bytes()).hexdigest()==digest
    report={'passed':True,'scope':f'Isolated {DEVICE} synthetic-network optimizer/checkpoint fixture',
            'resumed_paths':len(results),'path_results':results,
            'exact_items':['batch indices including short tails','pass/cursor','Python/NumPy draws',
                           'diffusion-like noise and timestep draws','learning rates','losses',
                           'head parameters','AdamW state','sampling RNG','training RNG'],
            'rejected_before_model_or_rng_mutation':rejection_checks,
            'pending_update_checkpoint_rejected':True,'fixed_budget_exhaustion_rejected':True,
            'checkpoint_save_rng_neutral':True,'isolated_metric_draws_rng_neutral':True,
            'discarded_rng_negative_control_detected':True,
            'synthetic_frozen_encoder_unchanged_and_omitted_from_model_payload':True,
            'cuda_rng_runtime_tested':USE_CUDA,
            'cuda_device':torch.cuda.get_device_name(DEVICE) if USE_CUDA else None,
            'peak_reserved_gib':torch.cuda.max_memory_reserved(DEVICE)/2**30 if USE_CUDA else None,'real_clip_or_robot_model_loaded':False,
            'source_data_read':False,'main_training_entrypoint_integrated':False,
            'original_production_sources_unchanged':True,'training_ready':False,
            'generalization_verdict':None,
            'component_sha256':hashlib.sha256((ROOT/'code/step_training_state.py').read_bytes()).hexdigest(),
            'verification_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['passed','resumed_paths','discarded_rng_negative_control_detected',
        'cuda_rng_runtime_tested','main_training_entrypoint_integrated','training_ready']},indent=2))


if __name__=='__main__':main()

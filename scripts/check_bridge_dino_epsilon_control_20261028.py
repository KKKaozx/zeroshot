"""Test new wrapper hooks/guards with CPU fixtures, no real policy execution."""
import copy
from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/'deployment/bridge_dino_epsilon_control_v1'
BASE=ROOT/'cluster_packages/bridge_backbone_pair_a40_v1'
sys.path.insert(0,str(HERE))
from checks import read,sha
from control import Control,initial_digest
spec=importlib.util.spec_from_file_location('epsilon_candidate_entry',HERE/'run.py')
entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)


def tsha(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def statesha(s):return initial_digest(s,tsha)
def refuses(fn):
    try:fn()
    except (AssertionError,ValueError,RuntimeError):return
    raise AssertionError('Expected a guard refusal')


def main():
    checks=[]
    original,source,_=entry.effective_source(BASE)
    compile(source,str(original),'exec')
    assert source.count('CONTROL.audit_first(')==1 and source.count('CONTROL.replay_and_activate(')==1
    assert source.count('CONTROL.choose_constants(')==1
    assert 'guess,truth,_=model.diffusion_loss(targets[ids],ctx,current[ids])' in source
    assert 'loss=torch.nn.functional.mse_loss(guess,truth)' in source
    assert 'optimizer=torch.optim.Adam(trainables,lr=1e-4)' in source
    assert 'seed(430000+update)' in source and 'seed(420000+update)' in source
    assert "for group in optimizer.param_groups:group['lr']=1e-5" in source
    checks.append('Downloaded actual positive effective runner hash exact; four declared hooks compile, training/noise/optimizer/evaluator retained')
    archived=read(ROOT/'reports/bridge_backbone_pair_a40_229116/cluster/extracted/dino/report.json')
    protocol=read(HERE/'protocol.json')
    assert archived['augmented_initial_sha256']==protocol['baseline_augmented_initial_sha256']
    assert archived['checkpoints'][0]['sha256']==protocol['baseline_checkpoint_sha256']
    assert read(ROOT/'reports/bridge_backbone_pair_a40_229116/download_verification.json')['passed']
    for name,digest in protocol['baseline_files'].items():
        assert sha(ROOT/'reports/bridge_backbone_pair_a40_229116/cluster/extracted'/name)==digest
    checks.append('All locally received baseline report/history/prediction hashes bound; existing independent baseline verification reused')

    constants=read(HERE/'canonical_constants.json')['constants']
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__();self.linear=nn.Linear(2,2);self.diffusion_prediction_type='sample'
            for name,value in constants.items():self.register_buffer(name,torch.tensor(value['values'],dtype=torch.float32))
    torch.manual_seed(42);m=Tiny()
    state=lambda net:{k:v.detach().clone() for k,v in net.state_dict().items()}
    expected=state(m);scratch=ROOT/'.tmp';scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='dino-epsilon-control-',dir=scratch) as tmp:
        temp=Path(tmp).resolve();assert temp.is_relative_to(scratch.resolve())
        variant=temp/protocol['known_variant_profile_relative'];variant.parent.mkdir(parents=True)
        variant.write_text(json.dumps({'schedule_values':{k:v['values'] for k,v in constants.items()},
            'tensors':{k:v['metadata'] for k,v in constants.items()}}))
        c=Control(temp);c.protocol=dict(c.protocol,baseline_initial_shared_sha256=initial_digest(expected,tsha))
        with torch.no_grad():m.posterior_mean_x0[10]+=1e-5;m.posterior_mean_xt[11]+=1e-5
        initial=state(m);rng=torch.get_rng_state().clone();r={}
        fixed=c.choose_constants(m,initial,tsha,r)
        assert initial_digest(fixed,tsha)==initial_digest(expected,tsha)
        assert all(torch.equal(m.state_dict()[k],v) for k,v in expected.items())
        assert torch.equal(rng,torch.get_rng_state())
        assert r['constant_identity_admission']['buffer_names']==['posterior_mean_x0','posterior_mean_xt']
        with torch.no_grad():m.linear.weight.add_(1.)
        refuses(lambda:c.choose_constants(m,state(m),tsha,{}))
        with torch.no_grad():m.load_state_dict(expected)
        bad=state(m);bad['betas'][0]+=.01
        refuses(lambda:c.choose_constants(m,bad,tsha,{}))
        checks.append('Only two fixed-buffer restoration matches entire initial state; learned/other-buffer changes refused; RNG preserved')

        # Saved old positive-prefix replay and complete initial restoration.
        learned=lambda net:{k:v.detach().clone() for k,v in net.named_parameters()}
        c=Control(temp);c.baseline=temp/'fixture-positive';c.baseline.mkdir()
        c.protocol=dict(c.protocol,baseline_augmented_initial_sha256=statesha(learned(m)))
        initial=learned(m);trained={k:torch.full_like(v,.25) for k,v in initial.items()}
        torch.save({'update':8192,'config':{'model':{'diffusion_prediction_type':'sample'}},'trainable_state_dict':trained},c.baseline/'final_trainable.pt')
        targets=torch.zeros(4,16,8);current=torch.zeros(4,1)
        def context(ids):return m.linear(torch.stack([ids.float(),ids.float()+1],dim=1))
        def evaluate(net,ctx,cur,truth,seeds,batch):
            assert batch==2
            values=np.repeat(ctx[:,0].detach().numpy()[:,None,None],16,axis=1)
            values=np.repeat(values,8,axis=2)
            return np.repeat(values[None],len(seeds),axis=0),{}
        @contextmanager
        def observer(net):
            modes=[(v,v.training) for v in net.modules()];cpu=torch.get_rng_state().clone();nr=np.random.get_state()
            try:
                net.eval()
                with torch.no_grad():yield
            finally:
                np.random.set_state(nr)
                for v,mode in modes:v.training=mode
                assert torch.equal(cpu,torch.get_rng_state())
        m.load_state_dict(trained,strict=False)
        with torch.no_grad():ctx=context(torch.arange(4));pred,_=evaluate(m,ctx,current,targets.numpy(),c.protocol['sampling_seeds'],2)
        np.savez(c.baseline/'reduced_u8192_train.npz',contexts=ctx.numpy(),predictions=pred,targets=targets.numpy())
        m.load_state_dict(initial,strict=False);cfg={'model':{'diffusion_prediction_type':'sample'}};report={}
        c.replay_and_activate(m,context,current,targets,observer,evaluate,learned,statesha,cfg,report)
        assert c.baseline_replayed and all(torch.equal(learned(m)[k],v) for k,v in initial.items())
        assert cfg['model']['diffusion_prediction_type']==m.diffusion_prediction_type=='epsilon'
        checks.append('Toy saved-positive replay exact and fresh parameter values restored before epsilon activation; no optimizer update')

        # The audited x0 reference backward must clear all gradients before the
        # epsilon backward on the same graph; first reference is never optimized.
        c.report={};opt=torch.optim.Adam(m.parameters(),lr=.0001);x=torch.ones(2,2)
        y=m.linear(x).reshape(2,1,2).repeat(1,2,4)[:,:,:7]
        clean=torch.zeros(2,2,8)
        ref=nn.functional.mse_loss(y,clean[:,:,:7]);ref.backward(retain_graph=True)
        grads=lambda net:{'fixture.':float(torch.linalg.vector_norm(torch.cat([p.grad.flatten() for p in net.parameters()])))}
        norms=grads(m);norm=nn.utils.clip_grad_norm_(list(m.parameters()),1.)
        c.first={'loss':float(ref.detach()),'global_unclipped_gradient_norm':float(norm),'unclipped_module_gradient_norms':norms}
        opt.zero_grad(set_to_none=True)
        c.audit_first(1,m,opt,y,clean,grads,list(m.parameters()))
        assert c.reference_passed and not opt.state and all(p.grad is None for p in m.parameters())
        epsilon=(y-torch.ones_like(y)).square().mean();epsilon.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
        checks.append('Reference x0 backward clears gradients, preserves empty Adam state and retains same graph for epsilon backward')
    for path in HERE.glob('*.py'):compile(path.read_text(encoding='utf-8'),str(path),'exec')
    for path in HERE.glob('*.sh'):assert b'\r' not in path.read_bytes()
    shell=(HERE/'job.sh').read_text();compile(shell.split("<<'HEALTH'\n",1)[1].split('\nHEALTH',1)[0],'health','exec')
    assert '--gres=gpu:a40:1' in shell
    checks.append('New sources and embedded Python compile; typed A40/LF shell static checks, no bash execution')
    report={'passed':True,'checks':checks,'scope':'CPU synthetic integration/source/hash checks only',
        'actual_policy_loaded':False,'actual_cuda_calls':0,'actual_policy_training_updates':0,
        'cluster_jobs_submitted':0,'actual_positive_replay_passed':False,
        'existing_math_and_sampler_checks_reused_not_rerun':['197845','code/check_bridge_prediction_targets.py'],
        'baseline_raw_download_verification':'reports/bridge_backbone_pair_a40_229116/download_verification.json'}
    (ROOT/'reports/bridge_dino_epsilon_control_validation_20261028.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()

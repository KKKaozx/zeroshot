"""CPU fixtures only; never loads a real policy/encoder or claims cluster replay."""
import ast
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
BUNDLE=ROOT/'deployment/oxe_bridge_supervision_v1'
sys.path[:0]=[str(BUNDLE),str(BUNDLE/'code')]
from supervision import bridge_loss,Supervisor,read,sha
from run_experiment import effective_source,options_for
from evaluate import metrics
from step_training_state import ResumableBatchPlan,UpdateSchedule,save_checkpoint,restore_checkpoint

CHECKS=[]


def passed(name):
    CHECKS.append(name);print('PASS',name,flush=True)


def refuses(fn):
    try:fn()
    except (AssertionError,ValueError,RuntimeError):return
    raise AssertionError('Expected rejection')


def original_policy_loss():
    tree=ast.parse((BUNDLE/'code/train.py').read_text(encoding='utf-8'))
    fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='policy_loss')
    future=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
    module=ast.fix_missing_locations(ast.Module(body=[future,fn],type_ignores=[]))
    scope={'torch':torch,'nn':nn}
    exec(compile(module,'original_policy_loss_ast','exec'),scope)
    return scope['policy_loss']


def loss_checks():
    torch.manual_seed(231)
    model=SimpleNamespace(diffusion_prediction_type='epsilon',decoder_type='diffusion',
        gripper_target_mode='state',balanced_gripper_loss=False,gripper_change_weight=1.,gripper_loss_weight=.25)
    actions=torch.randn(4,5,8)
    mask=torch.ones_like(actions);mask[1,:,2:]=0;mask[2,3:]=0;mask[3,:,7]=0
    pred=torch.randn(4,5,7,requires_grad=True);noise=torch.randn_like(pred)
    logits=torch.randn(4,5,requires_grad=True)
    old=original_policy_loss()(model,(pred,noise,logits),actions,nn.MSELoss(),supervision_masks=mask)
    rng=torch.get_rng_state().clone()
    new=bridge_loss(model,(pred,noise,logits),actions,mask,[1,1,1,1])
    for a,b in zip(old,new):assert torch.equal(a,b)
    og=torch.autograd.grad(old[0],(pred,logits),retain_graph=True)
    ng=torch.autograd.grad(new[0],(pred,logits),retain_graph=True)
    assert all(torch.equal(a,b) for a,b in zip(og,ng))
    assert torch.equal(rng,torch.get_rng_state())
    passed('all-kept objective and gradients exactly match archived policy_loss; loss RNG unchanged')
    keep=[1,0,1,0]
    val=bridge_loss(model,(pred,noise,logits),actions,mask,keep)
    # Independently sum selected rows, retaining complete original denominators.
    pose=sum(((pred[i]-noise[i]).square()*mask[i,:,:7]).sum() for i in (0,2))/mask[:,:,:7].sum()
    bce=nn.functional.binary_cross_entropy_with_logits(logits,(actions[:,:,7]>0).float(),reduction='none')
    grip=sum(bce[i][mask[i,:,7]>.5].sum() for i in (0,2))/(mask[:,:,7]>.5).sum()
    torch.testing.assert_close(val[0],pose+.25*grip,rtol=1e-7,atol=1e-7)
    gp,gg=torch.autograd.grad(val[0],(pred,logits),retain_graph=True)
    assert not gp[[1,3]].any() and not gg[[1,3]].any()
    assert gp[[0,2]].abs().sum()>0 and gg[[0,2]].abs().sum()>0
    passed('mixed keep matches independent numerator oracle; other-row direct gradients zero')
    zero=bridge_loss(model,(pred,noise,logits),actions,mask,[0]*4)
    grads=torch.autograd.grad(zero[0],(pred,logits),retain_graph=True)
    assert zero[0].item()==0 and all(x is not None and not x.any() for x in grads)
    lt=mask.clone();lt[:,:,2:]=0
    ltval=bridge_loss(model,(pred,noise,logits),actions,lt,[0]*4)
    g=torch.autograd.grad(ltval[0],(pred,logits),allow_unused=True)
    assert g[0] is not None and not g[0].any() and g[1] is None
    passed('zero-Bridge graph and original missing-gripper None behavior preserved')
    p=nn.Parameter(torch.tensor([1.,2.]));opt=torch.optim.AdamW([p],lr=.1,weight_decay=.2)
    p.sum().backward();opt.step();opt.zero_grad(set_to_none=True)
    previous=p.detach().clone();state=copy.deepcopy(opt.state_dict());q=nn.Parameter(previous.clone())
    oracle=torch.optim.AdamW([q],lr=.1,weight_decay=.2);oracle.load_state_dict(state)
    (p.sum()*0).backward();q.grad=torch.zeros_like(q)
    opt.step();oracle.step()
    assert torch.equal(p,q) and not torch.equal(p,previous)
    passed('zero gradients still apply original AdamW momentum/decay and global step')


def source_and_schedule_checks():
    reference=read(BUNDLE/'reference_contract.json')
    for name,digest in reference['source_code_sha256'].items():assert sha(BUNDLE/'code'/name)==digest
    assert sha(BUNDLE/'source_binding.json')==reference['offline_training_protocol']['cluster_preflight_binding_sha256']
    path,effective=effective_source();compile(effective,str(path),'exec')
    assert effective.count('SUPERVISION.loss(')==1
    assert 'value = api.evaluate_loss(model, loader, tokenizer, device, criterion,' in effective
    tree=ast.parse((BUNDLE/'code/train.py').read_text(encoding='utf-8'))
    evaluation=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='evaluate_loss')
    assert any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='policy_loss' for n in ast.walk(evaluation))
    passed('all archived contract source bytes exact; only training loss call replaced, validation unmodified')
    fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='parse_args')
    parsed=ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),fn],type_ignores=[]))
    namespace={'argparse':argparse,'Path':Path,'__file__':str(BUNDLE/'code/train.py')}
    exec(compile(parsed,'original_argument_parser','exec'),namespace)
    options=options_for(SimpleNamespace(parse_args=namespace['parse_args']),ROOT,ROOT/'CPU_FIXTURE',32768,None)
    operational={'output_dir','step_resume','stop_after_updates','checkpoint_every_updates','log_interval','cache_dir',
        'dataset_dir','split_manifest','language_bank','language_table_groups','original_instruction_catalog',
        'oxe_nominal_targets','oxe_nominal_source_map','oxe_nominal_validation_targets','oxe_formal_protocol','oxe_preflight_binding'}
    actual={k:v for k,v in vars(options).items() if k not in operational}
    # Asset paths are the only nonoperational CLI fields remapped to the root.
    actual['model_name']=reference['arguments']['model_name'];actual['dino_assets']=reference['arguments']['dino_assets']
    assert actual==reference['arguments']
    passed('original parser and all nonoperational recipe arguments preserved (asset root remapped only)')
    hook=Supervisor(BUNDLE,'preflight','fixture')
    bridge=draws=empty=0
    for i in range(65536):
        n=int(hook.schedule['lengths'][i]);ids=hook.schedule['batches'][i,:n].tolist()
        hook.begin(ids,i);bridge+=int(hook.keep[ids].sum());draws+=n;empty+=int(not hook.keep[ids].any())
        hook.commit(i+1)
    assert (bridge,draws,empty)==(131244,524276,6617)
    hook.initial_sha='CPU_FIXTURE';hook.first_replay={'passed':True,'scope':'synthetic state counter fixture'}
    state=hook.state_dict();other=Supervisor(BUNDLE,'preflight','fixture');other.initial_sha='CPU_FIXTURE'
    other.restore_state(state,65536);assert other.state_dict()==state
    bad=copy.deepcopy(state);bad['bridge_draws']+=1
    refuses(lambda:other.restore_state(bad,65536))
    first=Supervisor(BUNDLE,'preflight','fixture')
    ids=first.schedule['batches'][0].tolist();ids[0]=ids[0]+1
    refuses(lambda:first.begin(ids,0))
    passed('full65536 schedule/exposure counters and restore exact; changed schedule/counters refused')


def checkpoint_fixture():
    # A small actual AdamW/dropout/RNG continuation using the exact source helper.
    def plan():return ResumableBatchPlan(list(range(5)),batch_size=2,total_updates=6,seed=42,dataset_identity='CPU_FIXTURE')
    preview=plan();batches=[]
    for _ in range(6):batches.append(preview.begin_batch());preview.commit_batch()
    padded=np.full((6,2),-1,dtype=np.int32)
    for i,row in enumerate(batches):padded[i,:len(row)]=row
    def setup():
        torch.manual_seed(92)
        model=nn.Sequential(nn.Linear(3,8),nn.Dropout(.2),nn.Linear(8,8))
        optimizer=torch.optim.AdamW(model.parameters(),lr=.01,weight_decay=.01)
        batchplan=plan();schedule=UpdateSchedule(total_updates=6,base_lrs=[.01],kind='cosine',warmup_updates=2)
        schedule.apply(optimizer,0)
        hook=Supervisor(BUNDLE,'preflight','CPU_FIXTURE');hook.keep=np.array([1,0,0,0,0],dtype=np.uint8)
        hook.schedule={'batches':padded,'lengths':np.array([len(x) for x in batches])}
        hook.initial_sha='CPU_FIXTURE';hook.first_replay={'passed':True,'scope':'CPU synthetic only'}
        return model,optimizer,batchplan,schedule,hook
    cfg=SimpleNamespace(diffusion_prediction_type='epsilon',gripper_target_mode='state',
        balanced_gripper_loss=False,gripper_change_weight=1.,gripper_loss_weight=.25)
    def advance(state,end):
        model,opt,pl,sc,hook=state;history=[]
        while pl.completed_updates<end:
            ids=pl.begin_batch();hook.begin(ids,pl.completed_updates)
            x=torch.randn(len(ids),2,3);a=torch.randn(len(ids),2,8);mask=torch.ones_like(a)
            out=model(x);noise=torch.randn_like(out[:,:,:7]);loss=bridge_loss(cfg,(out[:,:,:7],noise,out[:,:,7]),a,mask,hook.pending['flags'])[0]
            opt.zero_grad(set_to_none=True);loss.backward();opt.step();pl.commit_batch();hook.commit(pl.completed_updates);sc.apply(opt,pl.completed_updates)
            history.append(float(loss.detach()))
        return history
    state=setup();advance(state,2)
    scratch=ROOT/'.tmp';scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='bridge-supervision-check-',dir=scratch) as tmp:
        target=Path(tmp).resolve();assert target.is_relative_to(scratch.resolve())
        model,opt,pl,sc,hook=state
        path=target/'fixture.pt';contract={'scope':'CPU_FIXTURE_NOT_CLUSTER'}
        save_checkpoint(path,model_state=model.state_dict(),optimizer=opt,plan=pl,schedule=sc,contract=contract,
            include_cuda=False,extra={'supervision':hook.state_dict()})
        continuation=advance(state,6);expected=copy.deepcopy(model.state_dict());rng=torch.get_rng_state().clone()
        resumed=setup();m,o,p,s,h=resumed
        extra=restore_checkpoint(path,model_state_loader=m.load_state_dict,optimizer=o,plan=p,schedule=s,contract=contract)
        h.restore_state(extra['supervision'],p.completed_updates)
        assert advance(resumed,6)==continuation
        assert all(torch.equal(expected[k],m.state_dict()[k]) for k in expected)
        assert torch.equal(rng,torch.get_rng_state()) and h.state_dict()==hook.state_dict()
    passed('CPU dropout/AdamW/sampler/RNG/intervention save-restore continuation exact')


def binding_checks():
    model=nn.Linear(2,2)
    hook=Supervisor(BUNDLE,'preflight','CPU_FIXTURE');hook.keep=np.array([1,0,1],dtype=np.uint8)
    api=SimpleNamespace(trainable_state_dict=lambda m:m.state_dict())
    plan=SimpleNamespace(total_updates=65536,batch_size=8)
    data=SimpleNamespace(train_count=3,samples=[{'source':x} for x in
        ['tfrecord_bridge_state_action','other','tfrecord_bridge_state_action']])
    reference=copy.deepcopy(hook.reference);contract=copy.deepcopy(reference)
    hook.bind(model,api,contract,plan,data)
    assert hook.reference==reference and 'diagnostic_supervision' not in reference['config']
    assert contract['supervision_counterfactual']['initial_learned_sha256']==hook.initial_sha
    assert contract['config']['diagnostic_supervision']==contract['supervision_counterfactual']
    bad=copy.deepcopy(reference);bad['optimizer']='Adam'
    refuses(lambda:hook.bind(model,api,bad,plan,data))
    bad_data=SimpleNamespace(train_count=3,samples=[{'source':'other'}]*3)
    refuses(lambda:hook.bind(model,api,copy.deepcopy(reference),plan,bad_data))
    scratch=ROOT/'.tmp';scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='bridge-bind-check-',dir=scratch) as tmp:
        target=Path(tmp).resolve();assert target.is_relative_to(scratch.resolve())
        pre={'passed':True,'mode':'preflight','completed_updates':2,
            'first_reference_replay':{'passed':True,'scope':'CPU_FIXTURE'},
            'initial_learned_sha256':hook.initial_sha,'protocol_sha256':sha(BUNDLE/'protocol.json')}
        (target/'diagnostic_receipt.json').write_text(json.dumps(pre),encoding='utf-8')
        fresh=Supervisor(BUNDLE,'fit','CPU_FIXTURE',target);fresh.keep=hook.keep
        admitted=copy.deepcopy(reference);fresh.bind(model,api,admitted,plan,data)
        assert admitted['supervision_counterfactual']['engineering_preflight_receipt_sha256']==sha(target/'diagnostic_receipt.json')
        changed=nn.Linear(2,2)
        refuses(lambda:fresh.bind(changed,api,copy.deepcopy(reference),plan,data))
        pre['protocol_sha256']='changed';(target/'diagnostic_receipt.json').write_text(json.dumps(pre),encoding='utf-8')
        refuses(lambda:fresh.bind(model,api,copy.deepcopy(reference),plan,data))
    passed('toy binding exact-reference/initial-digest/preflight-protocol gates and refusals (not actual encoder replay)')


def metric_and_package_checks():
    pred=np.zeros((3,2,4,8));target=np.zeros((2,4,8));target[:,:,6]=1;pred[:,:,:,6]=-1
    pred[:,0,:,0]=[1,2,3,99];pred[:,1,:,1]=[4,5,99,99]
    mask=np.zeros_like(target);mask[0,:3,:7]=1;mask[1,:2,:7]=1
    result=metrics(pred,target,mask,[0,1])
    assert result['path_position_normalized']==3.25
    assert result['endpoint_position_normalized']==4.
    assert result['path_rotation_deg']==0
    passed('paired evaluation draw/window axes, variable valid tails and quaternion sign fixture')
    roster=read(BUNDLE/'evaluation_windows.json')
    assert len(roster['rows'])==64 and len({r['group_id'] for r in roster['rows']})==64
    assert all(r['source']=='Bridge' for r in roster['rows'])
    zero=pred.copy();zero[:,:,:,3:7]=0
    refuses(lambda:metrics(zero,target,mask,[0,1]))
    count=0
    for path in BUNDLE.rglob('*.py'):
        compile(path.read_text(encoding='utf-8'),str(path),'exec');count+=1
    for path in BUNDLE.glob('*.sh'):
        assert b'\r' not in path.read_bytes() and path.read_bytes().startswith(b'#!/bin/bash\n')
    protocol=read(BUNDLE/'protocol.json')
    for name,digest in protocol['bundle_inputs'].items():assert sha(BUNDLE/name)==digest
    for line in (BUNDLE/'SHA256SUMS').read_text().splitlines():
        digest,name=line.split('  ',1);assert sha(BUNDLE/name)==digest
    shell=(BUNDLE/'job.sh').read_text();health=shell.split("<<'HEALTH'\n",1)[1].split('\nHEALTH',1)[0]
    compile(health,'job_health_python','exec')
    assert '--gres=gpu:a40:1' in shell
    assert 'entry.train(options)' in (BUNDLE/'run_experiment.py').read_text()
    passed(f'{count} Python sources compile; shell LF/resource/embedded Python static checks (bash NOT executed)')


if __name__=='__main__':
    torch.set_num_threads(1)
    loss_checks();source_and_schedule_checks();checkpoint_fixture();binding_checks();metric_and_package_checks()
    report={'passed':True,'checks':CHECKS,'runtime':{'torch':str(torch.__version__),'numpy':np.__version__},
        'scope':'CPU synthetic fixtures/source provenance only; no real policy, CUDA, encoders, cluster or scientific result',
        'real_cluster_preflight_passed':False,'cluster_submitted':False,
        'development_repairs':[
            'Initial full-schedule CPU fixture exposed per-step NPZ decompression; stopped the slow fixture, materialized arrays once and full65536 checks then passed.',
            'AST-only parser fixture initially omitted Path and __file__ globals; corrected test harness, no change to archived parser or training source.'
        ]}
    output=ROOT/'reports/oxe_bridge_supervision_package_validation_20261027.json'
    output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(output)

"""New-entry CPU checks: real frontend initialization and lightweight backend."""
import ast
import copy
import gc
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile
import numpy as np
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
B=ROOT/'scripts/cluster/pusht_zero_block_preflight_v1'
sys.path.insert(0,str(B))
import training_core as core
import zero_state
from zero_blocks import remove_blocks,formula_check
from project_frontend import ProjectObservationEncoder
from resource_probe import state_hash
from token_binding import project_update
from verify_preflight import validate
from native_data import sha
sys.path.insert(0,str(ROOT/'scripts'))
import check_pusht_project_training_20261020 as fixture
torch.set_num_threads(1)
checks=[]


def ok(name,condition=True):
    assert condition,name
    checks.append(name)


def reject(name,fn):
    try:fn()
    except (ValueError,AssertionError,KeyError):checks.append(name);return
    raise AssertionError('Accepted invalid '+name)


protocol=json.loads((B/'protocol.json').read_text())
for f in B.glob('*.py'):ast.parse(f.read_text(encoding='utf-8'),feature_version=(3,9))
ok('Python3.9 syntax all modules')
for name,digest in protocol['reused_sources_sha256'].items():
    assert sha(B/name)==digest and (B/name).read_bytes()==(ROOT/'scripts/cluster/pusht_project_training_v1'/name).read_bytes()
ok('All10 reused modules/JSON byte-identical to original8 bundle')
for line in (B/'SHA256SUMS').read_text().splitlines():
    digest,name=line.split(None,1);assert sha(B/name.strip())==digest
ok('All20 bundle file hashes')
with zipfile.ZipFile(ROOT/'cluster_packages/pusht_zero_block_preflight_v1.zip') as z:
    files=[p for p in B.iterdir() if p.is_file()]
    ok('ZIP exact20 members',len(files)==len(z.namelist())==20)
    ok('ZIP bytes match checked files',all(z.read(B.name+'/'+p.name)==p.read_bytes() for p in files))
text=torch.ones(1,3,768);mask=torch.ones(1,3,dtype=torch.long)
torch.manual_seed(43)
reference=ProjectObservationEncoder(text,mask,(64,65))
ref_projection=state_hash(reference.projection);ref_norm=state_hash(reference.adapter.output_norm)
ref_rng=torch.get_rng_state().clone();ref_full=state_hash(reference)
del reference;gc.collect()
torch.manual_seed(43)
encoder=ProjectObservationEncoder(text,mask,(64,65))
norm_object,projection_object=encoder.adapter.output_norm,encoder.projection
evidence=remove_blocks(encoder)
ok('Production8 constructor reconstructed exactly',evidence['initial_eight_frontend_sha256']==ref_full)
ok('Original projection weight values preserved',state_hash(encoder.projection)==ref_projection)
ok('Original output LayerNorm preserved',state_hash(encoder.adapter.output_norm)==ref_norm)
ok('Common parameter objects preserved',encoder.projection is projection_object and encoder.adapter.output_norm is norm_object)
ok('RNG consumption matches original8 initialization',torch.equal(ref_rng,torch.get_rng_state()))
ok('Zero active blocks and67648 remaining frontend parameters',len(encoder.adapter.layers)==0 and sum(p.numel() for p in encoder.parameters())==67648)
reject('Double conversion rejected',lambda:remove_blocks(encoder))
tokens=torch.randn(4,257,1024);positions=torch.randn(4,2)
actual=formula_check(encoder,tokens,positions)
ok('Real zero-layer pooling/norm/projection matches manual reference',actual.shape==(4,66))
ok('Agent state columns copied exactly',torch.equal(actual[:,64:],positions))
before=actual.detach().clone()
with torch.no_grad():encoder.text_tokens.add_(30)
ok('Zero-block output independent of constant text',torch.equal(before,formula_check(encoder,tokens,positions)))
reject('Trainable CLIP tokens rejected',lambda:formula_check(encoder,tokens.requires_grad_(True),positions))
tokens=tokens.detach()
reject('Missing CLS or patch token rejected',lambda:formula_check(encoder,tokens[:,:1],positions))
encoder.adapter.pooling='cls'
reject('Conversion refuses wrong pooling',lambda:remove_blocks(encoder))
encoder.adapter.pooling='cls_patch_mean'

# Reuse original compute_loss/EMA code with only a lightweight substitute U-Net.
policy=fixture.Policy();policy.obs_encoder=encoder
averaged,ema=fixture.fixture.make_ema(policy)
opt=torch.optim.AdamW(policy.parameters(),lr=1e-4,betas=(.95,.999),weight_decay=1e-6)
lr=torch.optim.lr_scheduler.LambdaLR(opt,lambda step:step/500 if step<500 else .5*(1+np.cos(np.pi*(step-500)/(512400-500))))
current=0
def capture():return zero_state.capture(policy,averaged,opt,lr,ema,current,'base','plan','zero')
with tempfile.TemporaryDirectory(dir=ROOT/'reports',prefix='zero_state_check_') as tmp:
    path=Path(tmp)/'replay.pt'
    for size in (8,38,64):
        batch={'obs':{'image':torch.randn(size,2,3),'agent_pos':torch.randn(size,2,2)},'action':torch.randn(size,16,2)}
        visual=torch.randn(size*2,257,1024)
        loss,audit=project_update(core,policy,opt,lr,ema,batch,visual,current,audit=True);current+=1
        ok('Real author loss/noise/gradient audit on logical batch%d'%size,all(audit.values()))
        core.atomic_checkpoint(path,capture())
        a,_=project_update(core,policy,opt,lr,ema,batch,visual,current);current+=1
        digest=core.tree_digest(capture())
        payload=torch.load(path,map_location='cpu',weights_only=False)
        current=zero_state.restore(payload,policy,averaged,opt,lr,ema,'base','plan','zero')
        b,_=project_update(core,policy,opt,lr,ema,batch,visual,current);current+=1
        ok('Full raw/EMA/Adam/LR/counter/RNG restore on batch%d'%size,a==b and core.tree_digest(capture())==digest)
    ok('Projection/output norm learned after warmup',state_hash(encoder.projection)!=ref_projection and state_hash(encoder.adapter.output_norm)!=ref_norm)
    ok('EMA remains frozen',all(not p.requires_grad and p.grad is None for p in averaged.parameters()))
    ok('Binding clears cached tokens and passive hooks',policy.obs_encoder.cached_tokens is None and not policy.obs_encoder._forward_pre_hooks)
    bad=dict(payload,zero_block_protocol_sha256='wrong')
    reject('Wrong zero-ablation protocol checkpoint rejected',lambda:zero_state.restore(bad,policy,averaged,opt,lr,ema,'base','plan','zero'))
    bad=dict(payload,format='pusht-native-complete-v1')
    reject('Original8/native format cannot masquerade as zero control',lambda:zero_state.restore(bad,policy,averaged,opt,lr,ema,'base','plan','zero'))
    reject('Wrong sample plan rejected',lambda:zero_state.restore(payload,policy,averaged,opt,lr,ema,'base','wrong','zero'))
    encoder.adapter.layers=nn.ModuleList([nn.Identity()])
    reject('Nonzero destination architecture rejected',lambda:zero_state.restore(payload,policy,averaged,opt,lr,ema,'base','plan','zero'))
    encoder.adapter.layers=nn.ModuleList()

# Complete output fixture exercises the real server-side validator.
with tempfile.TemporaryDirectory(dir=ROOT/'reports',prefix='zero_output_check_') as tmp:
    out=Path(tmp);(out/'replay.pt').write_bytes(b'synthetic state checksum fixture')
    actions=np.zeros((2,16,2),dtype=np.float32);np.savez(out/'smoke_actions.npz',action_pred=actions,action=actions[:,1:9])
    base=json.loads((B/'base_protocol.json').read_text());old=json.loads((B/'project_contract.json').read_text())
    reference=json.loads((B/'eight_block_reference.json').read_text())
    def save(x):(out/'report.json').write_text(json.dumps(x))
    r=dict(preflight_passed=True,errors=[],protocol_sha256=sha(B/'protocol.json'),formal_training_submitted=False,
        development_scored=False,final50_accessed=False,trained_weights_loaded=False,clip_training_executed=False,
        unique_preflight_updates=8,recomputed_replay_updates=1,isolated_partial_probe_updates=1,
        initial_backend_sha256=base['initial_backend_sha256'],normalizer_sha256=base['normalizer_sha256'],
        sample_order_sha256=old['sample_order_sha256'],sample_plan_sha256=old['sample_plan_sha256'],cache_report_sha256=old['cache_report_sha256'],
        eight_block_report_sha256=protocol['eight_block_report_sha256'],data=reference['data'],initialization=evidence,
        fresh_initialization_only=True,zero_block_formula_exact=True,shared_text_unchanged=True,ema_frozen=True,
        projection_updated=True,output_norm_updated=True,ema_projection_updated=True,
        loss_and_gradient_audit=audit,disk_resume=dict(exact=True,loss_before=.2,loss_after=.2,state_before_sha256='fixture',state_after_sha256='fixture',
            replay_completed_updates=2,replay_checkpoint_sha256=sha(out/'replay.pt')),
        losses=[.2]*8,warm_durations=[.1]*5,warm_update_seconds=.1,peak_reserved_gib=7.0,
        estimated_formal_training_hours_excluding_io_and_rollout=.1*84168/3600,
        partial38_probe=dict(microbatch_sizes=[8,8,8,8,6],finite_loss=True),ema_optimization_step_after_disposable_probe=9,
        active_cross_attention_blocks=0,smoke_actions_sha256=sha(out/'smoke_actions.npz'))
    save(r);ok('Real output verifier accepts consistent fixture',validate(out,B)['preflight_verified'])
    for label,mutate in [
        ('premature performance scoring',lambda x:x.update(development_scored=True)),
        ('trained source weights',lambda x:x.update(trained_weights_loaded=True)),
        ('incorrect update budget',lambda x:x.update(unique_preflight_updates=9)),
        ('different norm initialization',lambda x:x['initialization'].update(common_parameter_objects_preserved=False)),
        ('hidden attention blocks',lambda x:x['initialization'].update(active_cross_attention_blocks=8)),
        ('different data',lambda x:x.update(data={})),
        ('missing gradient audit',lambda x:x.update(loss_and_gradient_audit={})),
        ('unmatched restore state',lambda x:x['disk_resume'].update(state_after_sha256='wrong')),
        ('bad timing estimate',lambda x:x.update(estimated_formal_training_hours_excluding_io_and_rollout=1)),
        ('missing projection updates',lambda x:x.update(projection_updated=False))]:
        bad=copy.deepcopy(r);mutate(bad);save(bad)
        reject('Verifier rejects '+label,lambda:validate(out,B))
    save(r);(out/'replay.pt').write_bytes(b'changed')
    reject('Verifier rejects altered checkpoint bytes',lambda:validate(out,B))
for name in ('preflight.sh','submit_once.sh'):
    subprocess.run(['C:/Program Files/Git/bin/bash.exe','-n',str(B/name)],check=True,capture_output=True)
ok('Both Bash entries pass syntax')
ok('A40 and15minute no-requeue bound',all(v in (B/'preflight.sh').read_text() for v in ('gpu:a40:1','00:15:00','--no-requeue')))
ok('No formal training entry included',not protocol['formal_entry_included'] and not (B/'formal.sh').exists())
for name in ('base_protocol.json','project_contract.json','project_frontend.py','frozen_adapter.py','training_core.py'):
    assert (B/name).read_bytes()==(ROOT/'scripts/cluster/pusht_project_training_v1'/name).read_bytes()
ok('Original8 source artifacts remain byte-identical')
result=dict(checks_passed=len(checks),checks=checks,scope='Production frontend random initialization and zero-block path on CPU, real author loss/EMA with small substitute backend; no real weights/data/CUDA/rollouts',
    actual_model_checkpoint_loaded=False,actual_server_resources_measured=False,formal_training_submitted=False,
    archive_sha256=sha(ROOT/'cluster_packages/pusht_zero_block_preflight_v1.zip'))
(ROOT/'reports/pusht_zero_block_preflight_checks_20261022.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))

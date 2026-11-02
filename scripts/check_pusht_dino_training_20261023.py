"""Check only the new formal entry/identity, retaining qualified training machinery."""
import ast
import copy
import importlib.util
import json
import math
from pathlib import Path
import random
import shutil
import sys
import tempfile
import types
import zipfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
B = ROOT/'scripts/cluster/pusht_dino_training_v1'
sys.path.insert(0,str(B))
import training_core as core
import dino_state
import formal_binding
import verify_dino
from compare_backbones import compare
from train_dino import validate_resume
from cache_tools import sha

checks = []


def ok(label,condition=True):
    assert condition,label
    checks.append(label)


def reject(label,fn):
    try:fn()
    except (ValueError,AssertionError,KeyError):
        checks.append(label)
        return
    raise AssertionError('Bad state accepted: '+label)


def save(path,value):
    path.parent.mkdir(exist_ok=True,parents=True)
    path.write_text(json.dumps(value),encoding='utf-8')


c = formal_binding.check_sources(B)
base = formal_binding.read(B/'base_protocol.json')
ok('Formal source/base/visual protocol fingerprints')
for name in c['unchanged_old_files_byte_identical']:
    assert (B/name).read_bytes() == (ROOT/'scripts/cluster/pusht_project_training_v1'/name).read_bytes()
ok('Thirteen original training, model, loss, cache, evaluation and transport files byte-identical')
for name in ('dino_client.py','dino_worker.py'):
    assert (B/name).read_text() == (ROOT/'scripts/cluster/pusht_dino_cache_v1'/name).read_text().replace("'protocol.json'","'visual_protocol.json'")
ok('Qualified actual DINO IPC code changes only protocol filename')
for line in (B/'SHA256SUMS').read_text().splitlines():
    digest,name=line.split(None,1)
    assert sha(B/name.strip())==digest
ok('All SHA256SUMS entries')
with zipfile.ZipFile(ROOT/'cluster_packages/pusht_dino_training_v1.zip') as z:
    files=[f for f in B.iterdir() if f.is_file()]
    assert len(files)==len(z.namelist())==34
    assert all(z.read(B.name+'/'+f.name)==f.read_bytes() for f in files)
ok('Exact34-file ZIP, no large model/cache')
for f in B.glob('*.py'):ast.parse(f.read_text(encoding='utf-8'),feature_version=(3,9))
ok('All formal sources parse on Python3.9')
spec=importlib.util.spec_from_file_location('dino_builder',ROOT/'scripts/build_pusht_dino_training_20261023.py')
builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
transformed,changes=builder.transform_entry((builder.OLD/'train_project.py').read_text(encoding='utf-8'))
ok('Minimal entry reproducible with exact counted replacements',transformed==(B/'train_dino.py').read_text(encoding='utf-8'))
ok('Original first-loss/update/sample micro machinery retained',
   all(x in transformed for x in ('project_update(core, policy, optimizer, scheduler, ema, data.batch(indices)',
      'cache.tokens(indices), index, audit=audit)','num_warmup_steps=500','core.BatchPlan()',
      'completed % 2000 == 0','plan.digest(84168)','if args.segment == 2:')))
ok('Same fixed total/segments and original sample plan',c['total_updates']==84168 and c['segment_endpoints']==[42084,84168]
   and c['sample_order_sha256']=='b6481e424f6ee5f86e263b2fca4d20d4644b5e87c5ff5a334e8b3d7bcde488ec')
for segment in (1,2):
    script=(B/('part%d.sh'%segment)).read_text()
    assert all(x in script for x in ('gpu:a40:1','20:00:00','--no-requeue','--mem=40G'))
ok('Each segment A40/40GiB/20h/no-requeue; no silent budget reduction')
ok('18GiB within-project atomic headroom',formal_binding.space_values(140_000_000_000,20*1024**3,c)['passed'])
reject('Reject full filesystem before formal start',lambda:formal_binding.space_values(140_000_000_000,1,c))
reject('Reject project quota headroom despite shared free disk',lambda:formal_binding.space_values(195_000_000_000,10**12,c))

with tempfile.TemporaryDirectory(dir=ROOT/'reports',prefix='dino_formal_fixture_') as tmp:
    root=Path(tmp).resolve();assert root.parent==(ROOT/'reports').resolve()
    for relative,digest in c['required_reports_sha256'].items():
        if 'pusht-dino-cache-221055' in relative:
            source=ROOT/'reports/pusht_dino_cache_221055'/Path(relative).name
        elif 'pusht-native84k' in relative:source=ROOT/'reports/pusht_native_212158/report.json'
        else:source=ROOT/'reports/pusht_project_215999/report.json'
        target=root/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    binding=formal_binding.verify_binding(root,B)
    ok('Downloaded real221055 metadata binds formal identity without another12GB cache pass',
       binding['files']['visual.npy']['sha256']=='a26e19da0a4a0858853e3ee302a1195b5cc81c57ad6fdf38af0b483219e5966a')
    cache_report=root/c['cache_directory']/'report.json'
    old=cache_report.read_bytes();cache_report.write_bytes(old+b' ')
    reject('Reject changed cache report bytes',lambda:formal_binding.verify_binding(root,B))
    cache_report.write_bytes(old)

    # Only the new complete-state wrapper is exercised with a tiny8-block
    # stand-in, not a second original8-model fit or another DINO resource run.
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.obs_encoder=torch.nn.Module()
            self.obs_encoder.adapter=torch.nn.Module()
            self.obs_encoder.adapter.layers=torch.nn.ModuleList([torch.nn.Linear(2,2) for _ in range(8)])
            self.model=torch.nn.Linear(2,2)
        def forward(self,x):
            for layer in self.obs_encoder.adapter.layers:x=layer(x).tanh()
            return self.model(x)
    torch.manual_seed(42);np.random.seed(42);random.seed(42)
    policy=Toy();averaged=copy.deepcopy(policy).requires_grad_(False).eval()
    optimizer=torch.optim.AdamW(policy.parameters(),lr=1e-4)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda k:1/(k+1))
    ema=types.SimpleNamespace(optimization_step=0,decay=.9)
    def update(index):
        np.random.rand();random.random()
        optimizer.zero_grad(set_to_none=True)
        loss=policy(torch.randn(3,2)).square().mean();loss.backward();optimizer.step();scheduler.step()
        with torch.no_grad():
            for a,b in zip(averaged.parameters(),policy.parameters()):a.lerp_(b,.1)
        ema.optimization_step+=1
        return float(loss.detach())
    def capture(completed):return dino_state.capture(policy,averaged,optimizer,scheduler,ema,completed,'base','plan','contract','binding')
    update(0)
    checkpoint=root/'tiny.pt';core.atomic_checkpoint(checkpoint,capture(1))
    loss_a=update(1);expected=core.tree_digest(capture(2))
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    restored=dino_state.restore(payload,policy,averaged,optimizer,scheduler,ema,'base','plan','contract','binding')
    loss_b=update(1)
    ok('New DINO complete-state format: disk restore/update2 exact incl Adam/LR/EMA/RNG',
       restored==1 and loss_a==loss_b and expected==core.tree_digest(capture(2)))
    baseline=core.tree_digest(policy.state_dict())
    for key,value in [('format','pusht-native-complete-v1'),('project_contract_sha256','wrong'),
                      ('dino_backbone_binding_sha256','wrong'),('active_cross_attention_blocks',0)]:
        bad=copy.deepcopy(payload);bad[key]=value
        reject('Reject incompatible checkpoint before model modification: '+key,
               lambda:dino_state.restore(bad,policy,averaged,optimizer,scheduler,ema,'base','plan','contract','binding'))
        assert core.tree_digest(policy.state_dict())==baseline
    side=dict(project_contract_sha256='contract',protocol_sha256='base',sample_plan_sha256='plan',
        completed_updates=42084,segment=1,startup_checks={'disk_resume_replay_exact':True},dino_backbone_binding_sha256='binding')
    ok('Segment2 accepts exactly bound completed segment1',validate_resume(side,'contract','base','plan',84168,2,'binding')==42084)
    for key,value in [('dino_backbone_binding_sha256','wrong'),('completed_updates',42083),('segment',3),('startup_checks',{})]:
        bad=copy.deepcopy(side);bad[key]=value
        reject('Reject invalid next-segment source: '+key,lambda:validate_resume(bad,'contract','base','plan',84168,2,'binding'))
    # Formal output validators use tiny marker checkpoints and only2 trace rows
    # to isolate their new identity/segment behavior, not fabricate a model run.
    audit_keys={'original_compute_loss_used','common_noise_exact','common_timesteps_exact','epsilon_objective_loss_exact',
        'representative_backend_gradient_exact','frontend_and_backend_gradients_valid','adapter_gradients_valid','projection_gradients_valid'}
    audit=dict(disk_resume_replay_exact=True,loss_audit={k:True for k in audit_keys},loss_before_restore=.1,
        loss_after_restore=.1,state_digest_before_restore='same',state_digest_after_restore='same')
    def fixture(segment):
        out=root/('segment%d'%segment);out.mkdir()
        target=c['segment_endpoints'][segment-1]
        (out/'latest.pt').write_bytes(b'tiny output-validator fixture marker')
        shutil.copyfile(B/'cache_binding_reference.json',out/'backbone_binding.json')
        r=dict(segment=segment,segment_completed=True,errors=[],completed_updates=target,target_updates=target,
            total_project_budget=84168,project_contract_sha256=sha(B/'project_contract.json'),protocol_sha256=sha(B/'base_protocol.json'),
            final_set_accessed=False,clip_training_executed=False,dino_training_executed=False,trained_native_weights_loaded=False,
            automatic_submission=False,initial_backend_sha256=base['initial_backend_sha256'],normalizer_sha256=base['normalizer_sha256'],
            data=formal_binding.read(B/'cache_report_reference.json')['training_data'],frozen_dino_cache_verified=True,
            frozen_text_unchanged=True,active_cross_attention_blocks=8,lr_horizon_updates=512400,startup_checks=audit,
            checkpoint=str(out/'latest.pt'),checkpoint_sha256=sha(out/'latest.pt'),complete_resume_state_exact=True,
            initial_completed_updates=target-2,training_completed=segment==2,development_complete=segment==2)
        for key in ('sample_plan_sha256','sample_order_sha256','cache_report_sha256','dino_backbone_binding_sha256'):r[key]=c[key]
        side={key:r[key] for key in ('completed_updates','project_contract_sha256','protocol_sha256',
             'sample_plan_sha256','dino_backbone_binding_sha256')}
        side.update(sha256=r['checkpoint_sha256'],segment=segment,startup_checks=audit)
        save(out/'latest.json',side)
        logs=[]
        for n in range(target-1,target+1):
            lr=1e-4*(n/500 if n<500 else .5*(1+math.cos(math.pi*(n-500)/(512400-500))))
            logs.append(json.dumps(dict(completed_updates=n,loss=.1,lr_for_next_update=lr)))
        (out/'updates.jsonl').write_text('\n'.join(logs)+'\n')
        if segment==2:
            dev=formal_binding.read(B/'clip_development_reference.json')
            save(out/'development.json',dev)
            for row in dev['episodes'][:2]:(out/row['video']).write_bytes(b'nonempty fixture, not actual video')
            comparison=compare(dev,dev,formal_binding.read(B/'native_development_reference.json'),base)
            save(out/'comparison.json',comparison)
            r.update(development=dev,ema_unchanged_during_evaluation=True,backbone_comparison=comparison,
                evaluation_worker=formal_binding.read(B/'cache_report_reference.json')['live_worker'])
        save(out/'report.json',r)
        return out,r
    for segment in (1,2):
        out,r=fixture(segment)
        ok('Synthetic completed segment%d metadata/trace validator'%segment,verify_dino.validate(out,B)['segment_verified'])
        bad=copy.deepcopy(r);bad['dino_backbone_binding_sha256']='wrong';save(out/'report.json',bad)
        reject('Reject wrong binding in segment%d report'%segment,lambda:verify_dino.validate(out,B))
        save(out/'report.json',r)
        if segment==1:
            save(out/'development.json',{})
            reject('Reject premature development scoring in segment1',lambda:verify_dino.validate(out,B))
        else:
            compare_path=out/'comparison.json';saved=compare_path.read_bytes();compare_path.write_text('{}')
            reject('Reject changed final comparison',lambda:verify_dino.validate(out,B))
            compare_path.write_bytes(saved)
        (out/'updates.jsonl').write_text(json.dumps(dict(completed_updates=c['segment_endpoints'][segment-1],loss=.1,lr_for_next_update=0))+'\n')
        reject('Reject missing/changed LR trace in segment%d'%segment,lambda:verify_dino.validate(out,B))

clip=formal_binding.read(B/'clip_development_reference.json')
native=formal_binding.read(B/'native_development_reference.json')
same=compare(clip,clip,native,base)
ok('Paired comparison exact zero for identical reference rewards',same['dino_minus_clip_mean']==0 and same['paired_scene_bootstrap95']==[0.,0.])
bad=copy.deepcopy(clip);bad['episodes'][0]['environment_seed']+=1
reject('Reject changed evaluation scene identity',lambda:compare(bad,clip,native,base))

result=dict(checks_passed=len(checks),checks=checks,archive_sha256=sha(ROOT/'cluster_packages/pusht_dino_training_v1.zip'),
    scope='New entry, binding, tiny complete-checkpoint replay and synthetic output validators; qualified original training machinery reused',
    full_policy_training_executed=False,dino_resource_probe_repeated=False,full_cache_regenerated=False,
    new_cluster_jobs_submitted=False,actual_DINO_first_update_disk_audit_pending=True)
(ROOT/'reports/pusht_dino_training_checks_20261023.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,indent=2))

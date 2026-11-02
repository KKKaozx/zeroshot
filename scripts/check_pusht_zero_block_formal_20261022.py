"""New formal-entry checks; no repetition of previous real-frontend fixtures."""
import ast
import contextlib
import copy
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
B=ROOT/'scripts/cluster/pusht_zero_block_formal_v1'
OLD=ROOT/'scripts/cluster/pusht_project_training_v1'
sys.path.insert(0,str(B))
from native_data import sha
from pair_metrics import compare
from bind_preflight import verify_binding
from verify_formal import validate,equal
checks=[]


def ok(name,condition=True):
    assert condition,name
    checks.append(name)


def reject(name,fn):
    try:fn()
    except (ValueError,AssertionError,KeyError):checks.append(name);return
    raise AssertionError('Invalid accepted: '+name)


c=json.loads((B/'project_contract.json').read_text());base=json.loads((B/'base_protocol.json').read_text())
for name,digest in c['bundle_sources_sha256'].items():assert sha(B/name)==digest
ok('All formal source hashes')
for name,relative in c['reused_files_byte_identical'].items():assert (B/name).read_bytes()==(ROOT/'scripts/cluster'/relative).read_bytes()
ok('All15 core/data/evaluator/live modules byte-identical')
for name in ('zero_blocks.py','zero_state.py'):assert (B/name).read_bytes()==(ROOT/'scripts/cluster/pusht_zero_block_preflight_v1'/name).read_bytes()
ok('Qualified zero-block implementation reused byte-identically')
for line in (B/'SHA256SUMS').read_text().splitlines():
    digest,name=line.split(None,1);assert sha(B/name.strip())==digest
ok('All30 file checksums')
for f in B.glob('*.py'):ast.parse(f.read_text(encoding='utf-8'),feature_version=(3,9))
ok('Python3.9 syntax')
with zipfile.ZipFile(ROOT/'cluster_packages/pusht_zero_block_formal_v1.zip') as z:
    files=[p for p in B.iterdir() if p.is_file()]
    ok('Archive exact30 members',len(files)==len(z.namelist())==30)
    ok('Archive bytes match',all(z.read(B.name+'/'+p.name)==p.read_bytes() for p in files))
old=ast.parse((OLD/'train_project.py').read_text(encoding='utf-8'))
new=ast.parse((B/'train_zero.py').read_text(encoding='utf-8'))
def get_main(tree):return next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
oldloop=next(n for n in ast.walk(get_main(old)) if isinstance(n,ast.While))
loop=next(n for n in ast.walk(get_main(new)) if isinstance(n,ast.While))
class Normalize(ast.NodeTransformer):
    def visit_Constant(self,node):
        if isinstance(node.value,str) and ('PUSHT_PROJECT' in node.value or 'ZERO_BLOCK' in node.value):node.value='label'
        return node
    def visit_Call(self,node):
        self.generic_visit(node)
        if isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name) and node.func.value.id=='zero_state' and node.func.attr=='restore':
            node.func.value.id='core';node.args=node.args[:-1]
        return node
ok('Actual training while-loop unchanged except checkpoint identity/log labels',
    ast.dump(Normalize().visit(copy.deepcopy(oldloop)))==ast.dump(Normalize().visit(copy.deepcopy(loop))))
for start in (2,42084):
    updates,saved=[],[]
    env=dict(completed=start,target=84168,result={},log=io.StringIO(),json=json,started=0.,
        scheduler=type('LR',(),{'get_last_lr':lambda _:[1e-4]})(),time=type('T',(),{'monotonic':staticmethod(lambda:1.)})())
    def update(i,audit=False):
        assert not audit;updates.append(i);return .1,None
    env['one_update']=update;env['save_checkpoint']=lambda:saved.append(env['completed'])
    with contextlib.redirect_stdout(io.StringIO()):exec(compile(ast.Module(body=[copy.deepcopy(loop)],type_ignores=[]),'<real zero loop>','exec'),env)
    assert updates==list(range(start,84168)) and env['completed']==84168 and saved[-1]==84168
    assert [json.loads(s)['completed_updates'] for s in env['log'].getvalue().splitlines()]==list(range(start+1,84169))
    ok('Real loop no extra/skipped/repeated updates from%d'%start)
call_names=[n.func.id for n in ast.walk(loop) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)]
ok('No development scoring inside training loop','evaluate' not in call_names and 'compare' not in call_names)
text=(B/'train_zero.py').read_text()
ok('Exactly one fixed-endpoint evaluation',text.count('development = evaluate(')==1 and 'if completed == 84168:' in text)
ok('Qualified initial eight model built before block deletion',text.index('sum(p.numel()')<text.index('initialization = remove_blocks')<text.index('normalizer = LinearNormalizer()'))
ok('Formal snapshots use zero-state format', 'payload = zero_state.capture(' in text and text.count('zero_state.restore(')==2)
ok('Resume checks bound preflight identity','sidecar.get("preflight_binding_sha256") != binding_sha' in text)
ok('Formal starts fresh instead of preflight weights',not c['trained_preflight_or_reference_weights_allowed'])


def development(scores):
    scores=list(map(float,scores))
    return dict(complete=True,final_set_accessed=False,checkpoint_selection='EMA at exactly84168 updates; no best-of-development selection',
        episodes=[dict(environment_seed=5000000+i,max_overlap_reward=v,steps=1,reward_series=[v],full_reward=v==1,
            video='fixed%d.mp4'%i if i<2 else None) for i,v in enumerate(scores)],
        mean_score=float(np.mean(scores)),full_reward_fraction=float(np.mean(np.array(scores)==1)),
        reference_competence_gate_passed=bool(np.mean(scores)>=.8 and np.mean(np.array(scores)==1)>=.5))


eight=development([.4]*20);zero=development([.6]*20)
positive=compare(zero,eight,base,c)
ok('Positive interpretation needs practical/interval/full criteria',positive['zero_better_criteria_passed'] and not positive['eight_better_criteria_passed'])
ok('Deterministic paired bootstrap',equal(positive,compare(zero,eight,base,c)))
negative=compare(eight,zero,base,c)
ok('Symmetric evidence can favor retaining eight',negative['eight_better_criteria_passed'] and not negative['zero_better_criteria_passed'])
ambiguous=compare(development([.8]*10+[0]*10),eight,base,c)
ok('Mixed outcomes remain inconclusive',not ambiguous['zero_better_criteria_passed'] and not ambiguous['eight_better_criteria_passed'])
full_loss=compare(development([.9]*20),development([1]*5+[0]*15),base,c)
ok('Higher mean cannot bypass lower full-reward fraction',not full_loss['zero_better_criteria_passed'])
bad=copy.deepcopy(zero);bad['episodes'][0]['environment_seed']=1
reject('Reject different environment scene plan',lambda:compare(bad,eight,base,c))
bad=copy.deepcopy(zero);bad['checkpoint_selection']='best-of-development'
reject('Reject selected intermediate checkpoint',lambda:compare(bad,eight,base,c))
bad=copy.deepcopy(zero);bad['mean_score']=.99
reject('Reject stale/incorrect reward mean',lambda:compare(bad,eight,base,c))

with tempfile.TemporaryDirectory(dir=ROOT/'reports',prefix='zero_formal_') as tmp:
    root=Path(tmp);out=root/'runs/zero';out.mkdir(parents=True)
    pre=root/c['preflight_directory'];pre.mkdir(parents=True)
    initial=dict(remaining_frontend_parameters=67648,active_text_conditioning=False)
    def save(path,x):path.write_text(json.dumps(x))
    save(pre/'report.json',dict(initialization=initial,preflight_passed=True))
    binding=dict(format='pusht-zero-preflight-binding-v1',preflight_job=219260,formal_contract_sha256=sha(B/'project_contract.json'),
        preflight_protocol_sha256=c['preflight_protocol_sha256'],preflight_report_sha256=sha(pre/'report.json'),
        initialization=initial,verified={'preflight_verified':True},preflight_weights_used=False)
    bpath=out/'preflight_binding.json';save(bpath,binding)
    ok('Immutable preflight binding accepted',verify_binding(bpath,root,B)==binding)
    for field,value in (('preflight_job',1),('preflight_weights_used',True),('formal_contract_sha256','other')):
        bad=copy.deepcopy(binding);bad[field]=value;save(bpath,bad)
        reject('Reject binding '+field,lambda:verify_binding(bpath,root,B))
    save(bpath,binding)
    source_reference=json.loads((B/'eight_block_reference.json').read_text())
    actual_eight=source_reference['development']
    new_dev=development([e['max_overlap_reward'] for e in actual_eight['episodes']])
    outcome=compare(new_dev,actual_eight,base,c)
    for i in range(2):(out/('fixed%d.mp4'%i)).write_bytes(b'video-byte fixture')
    save(out/'development.json',new_dev);save(out/'comparison.json',outcome)
    (out/'latest.pt').write_bytes(b'checkpoint-byte fixture')
    r=dict(segment_completed=True,training_completed=True,development_complete=True,formal_complete=True,completed_updates=84168,
        target_updates=84168,segment=1,errors=[],final_set_accessed=False,clip_training_executed=False,trained_native_weights_loaded=False,
        automatic_submission=False,project_contract_sha256=sha(B/'project_contract.json'),protocol_sha256=sha(B/'base_protocol.json'),
        sample_plan_sha256=c['sample_plan_sha256'],sample_order_sha256=c['sample_order_sha256'],cache_report_sha256=c['cache_report_sha256'],
        initial_backend_sha256=base['initial_backend_sha256'],normalizer_sha256=base['normalizer_sha256'],data=source_reference['data'],
        frozen_clip_cache_verified=True,frozen_text_unchanged=True,preflight_binding_sha256=sha(bpath),zero_block_initialization=initial,
        active_cross_attention_blocks=0,startup_checks=dict(disk_resume_replay_exact=True,loss_audit={'fixture':True}),
        checkpoint_sha256=sha(out/'latest.pt'),initial_completed_updates=0,development=new_dev,
        evaluation_worker=dict(all_frozen=True,weights_unchanged=True),ema_unchanged_during_evaluation=True,zero_vs_eight=outcome)
    side=dict(sha256=r['checkpoint_sha256'],completed_updates=84168,project_contract_sha256=r['project_contract_sha256'],
        preflight_binding_sha256=r['preflight_binding_sha256'],startup_checks=r['startup_checks'])
    save(out/'latest.json',side)
    with (out/'updates.jsonl').open('w') as f:
        for step in range(1,84169):
            lr=1e-4*(step/500 if step<500 else .5*(1+math.cos(math.pi*(step-500)/(512400-500))))
            f.write(json.dumps(dict(completed_updates=step,loss=.1,lr_for_next_update=lr))+'\n')
    save(out/'report.json',r)
    ok('Full verifier accepts complete84168-update fixture',validate(out,B)['output_verified'])
    for label,mutate in (
        ('not complete',lambda x:x.update(formal_complete=False)),
        ('wrong endpoint',lambda x:x.update(completed_updates=8192)),
        ('native trained weights',lambda x:x.update(trained_native_weights_loaded=True)),
        ('unfrozen CLIP',lambda x:x['evaluation_worker'].update(weights_unchanged=False)),
        ('wrong architecture',lambda x:x.update(active_cross_attention_blocks=8)),
        ('preflight identity',lambda x:x.update(preflight_binding_sha256='wrong')),
        ('changed result interpretation',lambda x:x['zero_vs_eight'].update(interpretation='improved'))):
        bad=copy.deepcopy(r);mutate(bad);save(out/'report.json',bad)
        reject('Verifier rejects '+label,lambda:validate(out,B))
    save(out/'report.json',r)
    with (out/'updates.jsonl').open('a') as f:f.write('{}\n')
    reject('Verifier rejects extra update',lambda:validate(out,B))
for name in ('formal.sh','submit_once.sh'):
    subprocess.run(['C:/Program Files/Git/bin/bash.exe','-n',str(B/name)],check=True,capture_output=True)
ok('Both Bash entries syntax checked')
ok('A40/20hour/no-requeue bound',all(s in (B/'formal.sh').read_text(encoding='utf-8') for s in ('gpu:a40:1','20:00:00','--no-requeue')))
ok('No automatic dependency/next-job submission',(B/'submit_once.sh').read_text(encoding='utf-8').count('sbatch')==1)
prior=json.loads((ROOT/'reports/pusht_zero_block_preflight_release_20261022.json').read_text())
ok('Published preflight ZIP untouched',sha(ROOT/'cluster_packages/pusht_zero_block_preflight_v1.zip')==prior['archive_sha256'])
result=dict(checks_passed=len(checks),checks=checks,
    scope='New formal entry only: real derived loop with cheap updates, pairing and binding, full-output synthetic fixtures; previous47 frontend checks reused',
    actual_policy_loaded=False,training_executed=False,rollouts_executed=False,formal_training_submitted=False,
    archive_sha256=sha(ROOT/'cluster_packages/pusht_zero_block_formal_v1.zip'))
(ROOT/'reports/pusht_zero_block_formal_checks_20261022.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))

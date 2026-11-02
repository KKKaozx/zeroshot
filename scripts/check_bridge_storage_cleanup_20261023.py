"""Check Bridge-only retirement profile using tiny owned fixtures, no cluster access."""
import ast
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
spec=importlib.util.spec_from_file_location('bridge_retire',ROOT/'scripts/cluster_bridge_state_cleanup_v1.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
checks=[]


def ok(label,condition=True):assert condition,label;checks.append(label)
def reject(label,fn):
    try:fn()
    except (ValueError,FileNotFoundError):checks.append(label);return
    raise AssertionError('Bad fixture accepted: '+label)
def put(p,b):p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
def save(p,x):put(p,json.dumps(x).encode())


ast.parse((ROOT/'scripts/cluster_bridge_state_cleanup_v1.py').read_text(),feature_version=(3,9))
ok('Python3.9 syntax')
m.configure()
ok('Matching qualified cleanup engine reused',m.base.sha(Path(m.base.__file__))==m.BASE_SHA)
ok('Only six named optimizer-state candidates',len(m.base.TARGETS)==6 and all('with_adam' in x[0] or 'training_state_' in x[0] for x in m.base.TARGETS))
ok('207558 and207746 and author Bridge never targeted',all(not any(str(j) in x[0] for j in (207558,207746,203699,207348)) for x in m.base.TARGETS))
ok('All eight standalone inference weights retained',sum(len(x['retain']) for x in m.PROFILES.values())==8)

with tempfile.TemporaryDirectory(dir=ROOT/'reports',prefix='bridge_retire_fixture_') as temp:
    root=Path(temp).resolve();assert (ROOT/'reports').resolve() in root.parents
    for folder,profile in m.PROFILES.items():
        checkpoints=[]
        for i,(name,info) in enumerate(profile['retain'].items()):
            p=root/folder/name;put(p,('tiny retained inference '+str(i)).encode())
            info.update(bytes=p.stat().st_size,sha256=m.base.sha(p))
            checkpoints.append(dict(path=name,sha256=info['sha256'],includes_optimizer=False))
        for name,info in profile['retire'].items():
            p=root/folder/name;put(p,('tiny optimizer state '+name).encode())
            info.update(bytes=p.stat().st_size,sha256=m.base.sha(p))
        p=root/folder/'report.json'
        save(p,dict(execution_checks_passed=True,clip_unchanged=True,final_update=profile['final_update'],checkpoints=checkpoints))
        profile['report_sha256']=m.base.sha(p)
    put(root/'runs/bridge-coverage-rep-fit-207558/final_trainable.pt',b'protected main reference')
    put(root/'runs/bridge-rotation-weight-fit-207746/final_trainable.pt',b'protected rotation reference')
    m.configure();entries,protected=m.base.plan(root)
    ok('Read-only Bridge profile sees six states and eight finals',len(entries)==6 and len(protected)==8)
    first=root/next(iter(m.PROFILES))/'report.json';saved=first.read_bytes();first.write_bytes(b'changed')
    reject('Changed source report stops cleanup',lambda:m.base.plan(root))
    first.write_bytes(saved)
    p=root/protected[0]['path'];saved=p.read_bytes();p.unlink()
    reject('Missing standalone final checkpoint stops cleanup',lambda:m.base.plan(root))
    p.write_bytes(saved)
    entries,protected=m.base.plan(root)
    retained={root/x['path']:(root/x['path']).read_bytes() for x in protected}
    untouched={p:p.read_bytes() for p in root.rglob('*') if p.is_file() and p.relative_to(root).as_posix() not in dict(m.base.TARGETS)}
    with contextlib.redirect_stdout(io.StringIO()):r=m.base.execute(root,entries,protected,quiet_check=lambda:None)
    ok('Six optimizer fixtures retired',len(r['deleted'])==6)
    ok('All eight final weights byte-identical',all(p.read_bytes()==b for p,b in retained.items()))
    ok('All other original files and main references untouched',all(p.read_bytes()==b for p,b in untouched.items()))
    rows,_=m.base.plan(root)
    ok('Repeated Bridge plan is idempotent',all(x['status']=='already_absent' for x in rows))
    ok('Bridge-only audit receipt written',len(list((root/'storage_cleanup_receipts').glob('*.json')))==1)

r=dict(checks_passed=len(checks),checks=checks,scope='Bridge profile only, tiny synthetic checkpoints; prior20 engine guards reused, no repeated real data/model experiments',
    actual_cluster_deletion_executed=False,script_sha256=m.base.sha(ROOT/'scripts/cluster_bridge_state_cleanup_v1.py'))
(ROOT/'reports/bridge_storage_cleanup_checks_20261023.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r,indent=2))

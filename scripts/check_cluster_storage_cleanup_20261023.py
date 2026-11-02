"""Exercise retirement guards only on tiny owned temporary fixture files."""
import ast
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('retirement_check',ROOT/'scripts/cluster_storage_cleanup_v1.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
checks=[]


def ok(label,value=True):assert value,label;checks.append(label)
def reject(label,fn):
    try:fn()
    except (ValueError,FileNotFoundError):checks.append(label);return
    raise AssertionError('Bad fixture accepted: '+label)
def put(path,data):path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
def save(path,value):put(path,json.dumps(value).encode())


ok('Exactly six predeclared targets',len(m.TARGETS)==6)
ok('No retained final checkpoint is a target',not(set(dict(m.TARGETS))&set(m.RETAIN)))
tree=ast.parse((ROOT/'scripts/cluster_storage_cleanup_v1.py').read_text(),feature_version=(3,9))
calls=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
ok('No recursive deletion or glob expansion','rmtree' not in calls and 'rglob' not in calls and 'glob' not in calls)

with tempfile.TemporaryDirectory(dir=ROOT/'reports',prefix='storage_retirement_fixture_') as temp:
    root=Path(temp).resolve()
    assert (ROOT/'reports').resolve() in root.parents
    for i,(rel,_) in enumerate(m.TARGETS):
        put(root/rel,('tiny fake old checkpoint%d'%i).encode())
        digest=m.sha(root/rel)
        if rel in m.PARENTS:m.PARENTS[rel]=digest
        if rel.endswith('latest.pt'):save((root/rel).with_suffix('.json'),{'sha256':digest})
        elif 'pusht-zero' in rel:save((root/rel).parent/'report.json',{'disk_resume':{'replay_checkpoint_sha256':digest}})
        else:save((root/rel).parent/'report.json',{'disk_resume':{'checkpoint_sha256':digest}})
    original_reports={
        'pusht-native84k-212158':ROOT/'reports/pusht_native_212158/report.json',
        'pusht-project-part2-215999':ROOT/'reports/pusht_project_215999/report.json',
        'pusht-zero-formal-219466':ROOT/'reports/pusht_zero_219466/report.json',
    }
    for i,rel in enumerate(m.RETAIN):
        put(root/rel,('tiny fake retained endpoint%d'%i).encode())
        m.RETAIN[rel]=m.sha(root/rel)
        save((root/rel).with_suffix('.json'),{'sha256':m.RETAIN[rel]})
    for name,source in original_reports.items():
        r=m.read(source);r['checkpoint_sha256']=m.RETAIN['runs/'+name+'/latest.pt']
        if name=='pusht-native84k-212158':r['source20k_checkpoint_sha256']=m.PARENTS['runs/pusht-native20k-211309/latest.pt']
        if name=='pusht-project-part2-215999':r['resume_source_sha256']=m.PARENTS['runs/pusht-project-part1-214273/latest.pt']
        save(root/'runs'/name/'report.json',r)
    r=m.read(ROOT/'reports/pusht_bypass_217820/report.json')
    for mode in ('global_control','spatial_candidate'):
        r['groups'][mode]['checkpoint_sha256']=m.RETAIN['runs/pusht-bypass-formal-217820/'+mode+'/latest.pt']
    save(root/'runs/pusht-bypass-formal-217820/report.json',r)
    # A small report outside the target list must survive unchanged.
    extra=root/'runs/pusht-native20k-211279/report.json';save(extra,{'errors':['old failed launch']})
    reports={p:p.read_bytes() for p in root.rglob('*.json')}
    entries,protected=m.plan(root)
    ok('Read-only plan sees six fixture candidates',sum(x['status']=='present' for x in entries)==6)
    ok('Preview did not create receipt or delete files',not(root/'storage_cleanup_receipts').exists() and all((root/x[0]).exists() for x in m.TARGETS))
    failed_side=(root/'runs/pusht-native20k-211279/latest.json');side_bytes=failed_side.read_bytes();failed_side.unlink()
    missing,_=m.plan(root)
    ok('Missing failed-launch sidecar is skipped without deleting file',
       next(x for x in missing if '211279' in x['path'])['status']=='skipped_no_sidecar'
       and (root/'runs/pusht-native20k-211279/latest.pt').exists())
    failed_side.write_bytes(side_bytes)
    reject('Reject parent path escape',lambda:m.checked_path(root,'../escape.pt',False))
    reject('Reject absolute target',lambda:m.checked_path(root,str(root/'bad.pt'),False))
    first=root/m.TARGETS[0][0];original=first.read_bytes()
    first.unlink();first.mkdir()
    reject('Reject directory where checkpoint expected',lambda:m.plan(root))
    first.rmdir();first.write_bytes(original);entries,protected=m.plan(root)
    def busy():raise ValueError('Injected active Slurm job')
    reject('Active job prevents deletion',lambda:m.execute(root,entries,protected,quiet_check=busy))
    ok('Busy rejection leaves all candidates',all((root/x[0]).exists() for x in m.TARGETS))
    target=root/m.TARGETS[2][0];target_bytes=target.read_bytes();target.write_bytes(b'corrupted')
    with contextlib.redirect_stdout(io.StringIO()):reject('Changed candidate prevents every deletion',lambda:m.execute(root,entries,protected,quiet_check=lambda:None))
    ok('Candidate rejection leaves all files',all((root/x[0]).exists() for x in m.TARGETS))
    target.write_bytes(target_bytes);entries,protected=m.plan(root)
    final=root/next(iter(m.RETAIN));final_bytes=final.read_bytes();final.write_bytes(b'bad retained checkpoint')
    with contextlib.redirect_stdout(io.StringIO()):reject('Corrupt retained endpoint prevents every deletion',lambda:m.execute(root,entries,protected,quiet_check=lambda:None))
    ok('Retained failure leaves candidates intact',all((root/x[0]).exists() for x in m.TARGETS))
    final.write_bytes(final_bytes);entries,protected=m.plan(root)
    with contextlib.redirect_stdout(io.StringIO()):result=m.execute(root,entries,protected,quiet_check=lambda:None)
    ok('Successful fixture removes only six named candidates',len(result['deleted'])==6 and all(not(root/x[0]).exists() for x in m.TARGETS))
    ok('All retained endpoint bytes unchanged',all(m.sha(root/rel)==digest for rel,digest in m.RETAIN.items()))
    ok('All original metadata remains byte-identical',all(p.read_bytes()==data for p,data in reports.items()))
    receipts=list((root/'storage_cleanup_receipts').glob('*.json'))
    ok('Durable ledger records six deletions and hashes',len(receipts)==1 and len(m.read(receipts[0])['deleted'])==6)
    later,_=m.plan(root)
    ok('Repeated plan is idempotent',all(x['status']=='already_absent' for x in later))

r=dict(checks_passed=len(checks),checks=checks,scope='Only tiny synthetic files in an owned workspace temporary directory; production expected hashes replaced in memory for fixture execution',
    real_cluster_files_deleted=False,cluster_accessed=False,script_sha256=m.sha(ROOT/'scripts/cluster_storage_cleanup_v1.py'))
(ROOT/'reports/cluster_storage_cleanup_checks_20261023.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r,indent=2))

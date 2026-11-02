"""Retire six named obsolete checkpoints; retain reports and all final endpoints.

Default is read-only preview. --apply checks Slurm, identities and paths, hashes
the six files, journals each unlink, and never recursively deletes a directory.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
from datetime import datetime, timezone
import uuid

TARGETS=(
    ('runs/pusht-bypass-preflight-217778/global_control/replay.pt','Closed bypass disposable preflight replay'),
    ('runs/pusht-bypass-preflight-217778/spatial_candidate/replay.pt','Closed bypass disposable preflight replay'),
    ('runs/pusht-native20k-211279/latest.pt','Failed original native launch; competent native84168 retained'),
    ('runs/pusht-native20k-211309/latest.pt','Native20000 parent fully resumed into retained84168 endpoint'),
    ('runs/pusht-project-part1-214273/latest.pt','Project42084 parent fully resumed into retained84168 endpoint'),
    ('runs/pusht-zero-preflight-219260/replay.pt','Disposable two-update replay; zero84168 endpoint retained'),
)
RETAIN={
    'runs/pusht-native84k-212158/latest.pt':'65df7f1e3d6d91b0a90e549a2ce991942049f99376821cfb51ef4b2bf5b11a26',
    'runs/pusht-project-part2-215999/latest.pt':'9546973d4245d1165e7a5eb96bd59dab21dd38422ea78861f0133f4610224d13',
    'runs/pusht-zero-formal-219466/latest.pt':'c8ead029cec5814508ea4e85c3dd9258d78a9075212b7290d7dbd6eda4cd4685',
    'runs/pusht-bypass-formal-217820/global_control/latest.pt':'c1e3c2359f14a6476b039337a8fae63f72c11a2216c5e80cb0f5fe37bb58e328',
    'runs/pusht-bypass-formal-217820/spatial_candidate/latest.pt':'12f16f18fb15ef25aa134cb825aec634ab2d9b027c987da9f8d30304dac7eae3',
}
PARENTS={
    'runs/pusht-native20k-211309/latest.pt':'a9b28a1f09f7d733b8cd1936403ee3e650a4fdfdfcd64cc4ab7fd8f0a3f63551',
    'runs/pusht-project-part1-214273/latest.pt':'fb196c9953c9599149496051ccf73e5dc917c210dad5330b18dc86be61f5a749',
}


def read(p):
    with p.open(encoding='utf-8') as f:return json.load(f)


def checked_path(root,relative,must_exist=True):
    rel=Path(relative)
    if rel.is_absolute() or '..' in rel.parts:raise ValueError('Unsafe relative path')
    candidate=root/rel
    if candidate.resolve(strict=must_exist)!=candidate:raise ValueError('Symlink or redirected path: '+relative)
    if root not in candidate.parents:raise ValueError('Path outside project')
    if must_exist and not stat.S_ISREG(candidate.lstat().st_mode):raise ValueError('Expected regular file')
    return candidate


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def stamp(p):
    s=p.lstat()
    if not stat.S_ISREG(s.st_mode):raise ValueError('Not a regular file: '+str(p))
    return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_nlink)


def require_quiet():
    result=subprocess.run(['squeue','--me','--noheader'],check=True,capture_output=True,text=True)
    if result.stdout.strip():raise ValueError('Slurm still lists your jobs; no files deleted.\n'+result.stdout)


def protect_endpoints(root):
    reports={}
    for directory in ('pusht-native84k-212158','pusht-project-part2-215999','pusht-zero-formal-219466'):
        r=read(checked_path(root,'runs/'+directory+'/report.json'))
        if not(r['training_completed'] and r['development_complete'] and r['completed_updates']==84168 and not r['errors']):
            raise ValueError('Final endpoint incomplete: '+directory)
        reports[directory]=r
    for key in ('pusht-native84k-212158','pusht-project-part2-215999'):
        if not reports[key]['complete_resume_state_exact']:raise ValueError('Successor lacks complete restore evidence')
    if reports['pusht-native84k-212158']['source20k_checkpoint_sha256']!=PARENTS['runs/pusht-native20k-211309/latest.pt']:
        raise ValueError('Different native parent')
    if reports['pusht-project-part2-215999']['resume_source_sha256']!=PARENTS['runs/pusht-project-part1-214273/latest.pt']:
        raise ValueError('Different project parent')
    bypass=read(checked_path(root,'runs/pusht-bypass-formal-217820/report.json'))
    if not bypass['formal_complete'] or bypass['errors'] or bypass['admission']['development_admission_passed']:
        raise ValueError('Bypass branch not completed/closed as expected')
    protected=[]
    for relative,expected in RETAIN.items():
        p=checked_path(root,relative);side=read(checked_path(root,str(Path(relative).with_suffix('.json'))))
        if p.stat().st_size<=0:raise ValueError('Empty retained checkpoint')
        if expected and side['sha256']!=expected:raise ValueError('Changed retained checkpoint sidecar')
        if 'pusht-bypass-formal-' in relative:
            mode=Path(relative).parent.name;group=bypass['groups'][mode]
            if not(group['training_completed'] and group['development_complete'] and group['completed_updates']==92360):
                raise ValueError('Incomplete retained bypass endpoint')
            expected=group['checkpoint_sha256']
        else:expected=reports[Path(relative).parent.name]['checkpoint_sha256']
        if side['sha256']!=expected:raise ValueError('Retained report/sidecar mismatch')
        protected.append(dict(path=relative,sha256_recorded_in_verified_report=expected,stamp=stamp(p)))
    return protected


def expected_target_hash(root,relative):
    p=root/relative
    if p.name=='latest.pt':
        return read(checked_path(root,str(Path(relative).with_suffix('.json'))))['sha256']
    report=read(checked_path(root,str(Path(relative).parent/'report.json')))
    if 'pusht-zero-preflight-' in relative:return report['disk_resume']['replay_checkpoint_sha256']
    return report['disk_resume']['checkpoint_sha256']


def plan(root):
    protected=protect_endpoints(root)
    protected_inodes={tuple(x['stamp'][:2]) for x in protected}
    entries=[]
    for relative,reason in TARGETS:
        p=checked_path(root,relative,must_exist=False)
        if not p.exists():entries.append(dict(path=relative,status='already_absent',reason=reason));continue
        info=stamp(p)
        if tuple(info[:2]) in protected_inodes:raise ValueError('Candidate shares a retained final checkpoint inode')
        if relative=='runs/pusht-native20k-211279/latest.pt' and not p.with_suffix('.json').exists():
            entries.append(dict(path=relative,status='skipped_no_sidecar',reason='Failed launch has no checksum sidecar; leave it untouched'))
            continue
        expected=expected_target_hash(root,relative)
        if relative in PARENTS and expected!=PARENTS[relative]:raise ValueError('Parent differs from completed successor')
        entries.append(dict(path=relative,status='present',reason=reason,bytes=info[2],stamp=info,
            link_count=info[4],expected_sha256=expected))
    return entries,protected


def execute(root,entries,protected,quiet_check=require_quiet):
    quiet_check()
    for row in protected:
        p=checked_path(root,row['path'])
        print('HASH_RETAINED_ENDPOINT',row['path'],flush=True)
        if sha(p)!=row['sha256_recorded_in_verified_report']:raise ValueError('Retained endpoint hash mismatch; nothing deleted')
    for row in entries:
        if row['status']!='present':continue
        p=checked_path(root,row['path'])
        print('HASH_BEFORE_RETIRE',row['path'],flush=True)
        if tuple(row['stamp'])!=stamp(p) or sha(p)!=row['expected_sha256']:raise ValueError('Candidate changed; nothing deleted')
    if protect_endpoints(root)!=protected:raise ValueError('Retained endpoint changed; nothing deleted')
    quiet_check()
    audit=root/'storage_cleanup_receipts'
    if audit.exists() and audit.resolve()!=audit:raise ValueError('Audit directory is redirected')
    audit.mkdir(exist_ok=True)
    if audit.resolve()!=audit:raise ValueError('Unsafe audit directory')
    ledger=audit/('retire6_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]+'.json')
    record=dict(format='checkpoint-retirement-v1',root=str(root),target_count=len(TARGETS),entries=entries,
        retained=protected,deleted=[],scope='Six obsolete checkpoints only; all reports, videos and final checkpoints retained. No checkpoint-byte backup created.')
    def commit():
        temp=ledger.with_suffix('.tmp')
        with temp.open('w',encoding='utf-8') as f:
            json.dump(record,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
        os.replace(temp,ledger)
    commit()
    for row in entries:
        if row['status']!='present':continue
        p=checked_path(root,row['path'])
        if tuple(row['stamp'])!=stamp(p):raise ValueError('Candidate changed during cleanup; see receipt')
        p.unlink()  # Exactly one predeclared regular file. No rmtree or glob.
        record['deleted'].append(dict(path=row['path'],bytes=row['bytes'],sha256=row['expected_sha256']))
        commit();print('RETIRED',row['path'],flush=True)
    if protect_endpoints(root)!=protected:raise ValueError('Unexpected retained endpoint change; see receipt')
    print('CLEANUP_RECEIPT',ledger,flush=True)
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--apply',action='store_true');a=parser.parse_args()
    root=Path('/projects/Zeroshot').resolve(strict=True)
    if str(root)!='/projects/_ssd/Zeroshot':raise ValueError('Unexpected project root; no cleanup')
    entries,protected=plan(root)
    total=sum(x.get('bytes',0) for x in entries)
    for row in entries:print(row['status'],round(row.get('bytes',0)/1024**3,3),'GiB',row['path'])
    print('TARGET_LOGICAL_GIB',round(total/1024**3,3),'Physical space release can be lower for hardlinks.')
    if a.apply:
        if any(x['status']=='present' for x in entries):execute(root,entries,protected)
        else:print('All six targets already absent; no action.')
    else:print('PREVIEW ONLY; --apply performs the six listed checkpoint deletions.')


if __name__=='__main__':main()

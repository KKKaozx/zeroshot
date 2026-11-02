"""Build a Bridge-only retirement profile from received immutable result reports."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    profiles={}
    for directory in ('bridge-late-lr-fit-200150','bridge-lr-rep-fit-200208/seed42',
                      'bridge-lr-rep-fit-200208/seed44','bridge-budget-fit-208586','bridge-fixed-fit-208964'):
        path=ROOT/'reports'/directory/'report.json';r=json.loads(path.read_text())
        assert r['execution_checks_passed'] and r['clip_unchanged']
        regular=r['checkpoints']
        opt=r.get('optimizer_checkpoints',[])
        if r.get('optimizer_checkpoint'):opt=opt+[r['optimizer_checkpoint']]
        opt=opt+[v for v in regular if v.get('includes_optimizer')]
        keep=[v for v in regular if not v.get('includes_optimizer')]
        assert keep and opt
        profiles['runs/'+directory]=dict(report_sha256=sha(path),final_update=r['final_update'],
            retire={v['path']:dict(sha256=v['sha256'],bytes=v['bytes']) for v in opt},
            retain={v['path']:dict(sha256=v['sha256'],bytes=v['bytes']) for v in keep})
    assert sum(len(v['retire']) for v in profiles.values())==6
    source='''"""Bridge optimizer-state retirement only; final named inference weights stay."""
import argparse
from pathlib import Path
import cluster_storage_cleanup_v1 as base

BASE_SHA = %r
PROFILES = %s


def protect_bridge(root):
    protected=[]
    for folder,spec in PROFILES.items():
        path=base.checked_path(root,folder+'/report.json')
        if base.sha(path)!=spec['report_sha256']:
            raise ValueError('Bridge report changed: '+folder)
        r=base.read(path)
        if not r['execution_checks_passed'] or not r['clip_unchanged'] or r['final_update']!=spec['final_update']:
            raise ValueError('Bridge final result incomplete')
        rows={v['path']:v for v in r['checkpoints']}
        for name,info in spec['retain'].items():
            if rows[name]['sha256']!=info['sha256'] or rows[name].get('includes_optimizer'):
                raise ValueError('Expected standalone final inference weights')
            p=base.checked_path(root,folder+'/'+name)
            if p.stat().st_size!=info['bytes']:raise ValueError('Retained Bridge checkpoint size differs')
            protected.append(dict(path=folder+'/'+name,sha256_recorded_in_verified_report=info['sha256'],stamp=base.stamp(p)))
    return protected


def configure():
    if base.sha(Path(base.__file__))!=BASE_SHA:raise ValueError('Upload matching base cleanup script')
    items={folder+'/'+name:info for folder,spec in PROFILES.items() for name,info in spec['retire'].items()}
    base.TARGETS=tuple((rel,'Closed Bridge training optimizer snapshot; standalone final inference weights retained') for rel in items)
    base.PARENTS={}
    base.protect_endpoints=protect_bridge
    base.expected_target_hash=lambda root,relative:items[relative]['sha256']


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--apply',action='store_true');a=p.parse_args()
    root=Path('/projects/Zeroshot').resolve(strict=True)
    if str(root)!='/projects/_ssd/Zeroshot':raise ValueError('Unexpected project root')
    configure()
    entries,protected=base.plan(root)
    for row in entries:print(row['status'],round(row.get('bytes',0)/1024**3,3),'GiB',row['path'])
    total=sum(v.get('bytes',0) for v in entries)
    print('BRIDGE_TARGET_LOGICAL_GIB',round(total/1024**3,3))
    if a.apply:
        if any(v['status']=='present' for v in entries):base.execute(root,entries,protected)
        else:print('All six Bridge states already absent; no action.')
    else:print('PREVIEW ONLY; --apply deletes only these Bridge optimizer snapshots.')


if __name__=='__main__':main()
'''%(sha(ROOT/'scripts/cluster_storage_cleanup_v1.py'),repr(profiles))
    output=ROOT/'scripts/cluster_bridge_state_cleanup_v1.py';output.write_text(source,encoding='utf-8',newline='\n')
    result=dict(status='prepared_not_executed_on_cluster',target_count=6,retained_inference_checkpoint_count=sum(len(x['retain']) for x in profiles.values()),
        target_bytes=sum(x['bytes'] for p in profiles.values() for x in p['retire'].values()),
        profiles=profiles,script_sha256=sha(output),shared_base_sha256=sha(ROOT/'scripts/cluster_storage_cleanup_v1.py'))
    (ROOT/'reports/bridge_storage_cleanup_plan_20261023.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='profiles'},indent=2))


if __name__=='__main__':main()

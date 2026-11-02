"""Retire declared rebuildable caches and superseded checkpoints. No GPU or imports.

Run on the cluster with python3 cluster_reclaim_space_20261029.py --apply.
No directory-tree deletion, no environment/final-model compaction, no job cancellation.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path,PurePosixPath
import stat
import subprocess
import uuid
import zipfile

PROTECTED_PREFIXES=(
    'data','envs','baseline_setup','dinov2_large_frozen_assets_20261023','paper_reproduction',
    'runs/oxe-sampling-pair-233531','runs/oxe-pose-pair-232587','runs/oxe-target-short-231297',
    'runs/oxe-dino-noaug-s42','runs/oxe-dino-noaug-s43','runs/oxe-bridge-supervision-s42',
    'runs/oxe-dino-pref-226133',
    'runs/pusht-native84k-212158','runs/pusht-project-part2-215999','runs/pusht-dino-part2-222758',
    'runs/pusht-zero-formal-219466','runs/pusht-bypass-formal-217820','runs/bridge-backbone-pair-a40-229116',
    'runs/bridge-coverage-rep-fit-207558','runs/bridge-rotation-weight-fit-207746',
    'oxe_sampling_pair_v1','oxe_fixed8_learnability_v1','oxe_current_pose_formal_v1',
    'oxe_current_pose_preflight_v1','oxe_target_pair_short_v1','oxe_dino_main_preflight_v1',
)
PIP_CACHE_DIRS=('.tmp/pip/http','.tmp/pip/http-v2','.tmp/pip/wheels','.tmp/pip/selfcheck')
ENDPOINTS={
 'runs/pusht-native84k-212158/latest.pt':'65df7f1e3d6d91b0a90e549a2ce991942049f99376821cfb51ef4b2bf5b11a26',
 'runs/pusht-project-part2-215999/latest.pt':'9546973d4245d1165e7a5eb96bd59dab21dd38422ea78861f0133f4610224d13',
 'runs/pusht-dino-part2-222758/latest.pt':'787e51bd7b8bd0d9c648da3abef1c17ada0605d5152afff67d26805ea5861789',
 'runs/bridge-coverage-rep-fit-207558/final_trainable.pt':'926030039ac9d32634bf193e989a44a5b891882c503cb8b40fd7b5091ac98ed6',
}
FIXED=(
 ('runs/pusht-project-cache-213439/visual.npy','Rebuildable frozen CLIP training features','clip',None),
 ('runs/pusht-dino-cache-221055/visual.npy','Rebuildable frozen DINO training features','dino',None),
 ('bridge_demonstration_expansion_v1/full_tokens.pt','Closed Bridge frozen-token cache','bridge',None),
 ('bridge_training_coverage_v1/full_tokens.pt','Closed Bridge frozen-token cache','bridge',None),
 ('runs/bridge-dino-only-cache-v1/full_tokens.pt','Closed DINO-only Bridge frozen-token cache','bridge',None),
 ('bridge_prefix_pair_v1/delta_visual_tokens.pt','Closed prefix experiment feature cache','bridge',None),
 ('bridge_state_prior_v1/full_tokens.pt','Closed Bridge frozen-token cache','bridge',None),
 ('runs/pusht-native20k-211279/latest.pt','Failed launch; not a completed model or resume source','native',None),
 ('runs/pusht-native20k-211309/latest.pt','20k parent fully resumed into native84k','native','a9b28a1f09f7d733b8cd1936403ee3e650a4fdfdfcd64cc4ab7fd8f0a3f63551'),
 ('runs/pusht-project-part1-214273/latest.pt','CLIP half-run fully resumed into completed endpoint','clip','fb196c9953c9599149496051ccf73e5dc917c210dad5330b18dc86be61f5a749'),
 ('runs/pusht-dino-part1-221186/latest.pt','DINO half-run fully resumed into completed endpoint','dino','5e9342c06dcf986a058036f12136727bc154deb3c2356c59c7a9afb63d61b59d'),
 ('runs/pusht-bypass-preflight-217778/global_control/replay.pt','Closed disposable preflight replay','clip',None),
 ('runs/pusht-bypass-preflight-217778/spatial_candidate/replay.pt','Closed disposable preflight replay','clip',None),
 ('runs/pusht-zero-preflight-219260/replay.pt','Closed disposable preflight replay','clip',None),
 ('runs/oxe-bridge-supervision-preflight-229577/latest_step.pt','Closed supervision engineering-preflight optimizer state',None,None),
)
GROUP_ENDPOINT={'native':'runs/pusht-native84k-212158/latest.pt','clip':'runs/pusht-project-part2-215999/latest.pt',
                'dino':'runs/pusht-dino-part2-222758/latest.pt','bridge':'runs/bridge-coverage-rep-fit-207558/final_trainable.pt'}
# Filled from the already-qualified Bridge retirement manifest by the local builder.
BRIDGE_PROFILES={'runs/bridge-late-lr-fit-200150': {'report_sha256': 'd0a68c6f88904bd959fa2bc58dbf89af34b00b40b8cf31bd7a38b6addd1bcebd', 'final_update': 2048, 'retire': {'common_u1024_with_adam.pt': {'sha256': '5aa7f77621b292ca82caa397eb09465780daee6784a4316eb75cfa3915fadbdc', 'bytes': 1587979022}}, 'retain': {'constant_u2048_trainable.pt': {'sha256': '4a30f2cdffe31352a3e715ababf97fe673750f113fbe9ee345d698605b4c9a72', 'bytes': 529309743}, 'reduced_u2048_trainable.pt': {'sha256': '656cd8996fe26e1842071fd0a6bd16c71b50dd845b0c7ef86cd547015b33bc6a', 'bytes': 529309428}}}, 'runs/bridge-lr-rep-fit-200208/seed42': {'report_sha256': 'f48e3f61d8cd2e57d6b2e948636012e0b6ed2bc1aecfae55c1aad886604a01ed', 'final_update': 2048, 'retire': {'common_u1024_with_adam.pt': {'sha256': '55ceee4dc0d38be069a833a6a7e628c29a8212cecf63299a811e34545acefcf8', 'bytes': 1587979150}}, 'retain': {'constant_u2048_trainable.pt': {'sha256': 'dbd4cf9cd9f408a166e46fd46650db55b7bc528772e91c0b3544ef1ce5651d9e', 'bytes': 529309871}, 'reduced_u2048_trainable.pt': {'sha256': '783ba911c1e958920f3ea1f141763ecb57170016347a5914a79169df03a0b971', 'bytes': 529309556}}}, 'runs/bridge-lr-rep-fit-200208/seed44': {'report_sha256': 'c8fa0bd21c4fce1ca4b8db2c37d691542066ca157bc2fc28b8100f28cba39e1d', 'final_update': 2048, 'retire': {'common_u1024_with_adam.pt': {'sha256': 'fc9118785279c887e3a53cb57c85a167026599d8a95bfab9a27ae2768e3bc799', 'bytes': 1587979150}}, 'retain': {'constant_u2048_trainable.pt': {'sha256': '5fe97ff14a3212084536c63ecd87b7d64152d18a8d7fa526365f780c63a08d48', 'bytes': 529309871}, 'reduced_u2048_trainable.pt': {'sha256': '6a1363a3b79536ad654209c5c5557463c146f2a675bb20c940894a7253474382', 'bytes': 529309556}}}, 'runs/bridge-budget-fit-208586': {'report_sha256': 'c4de4467d976c9d68a91e37b40ea607d3b728a8115079ab4325bb9bde3421598', 'final_update': 16170, 'retire': {'training_state_u08192.pt': {'sha256': '7ef62decf57e8e817b34b955da39e213ca0f0e2c5f2bebd4ad2a5f7f61851df1', 'bytes': 1588139196}, 'training_state_u16170.pt': {'sha256': 'ea1c4647b41b5d68e51b75b018787fe581f121dd8580110ad361003b3d3b043c', 'bytes': 1588139196}}, 'retain': {'final_trainable.pt': {'sha256': '618e591ea63d4cbd7ec80dc6dba0abe8aeb64b68be33e915553b258f801ac2f5', 'bytes': 529393826}}}, 'runs/bridge-fixed-fit-208964': {'report_sha256': '2cbefe8655915dbb3e9fc29f09bee1114c18a9b202c15cfccabe85ef805d300d', 'final_update': 16170, 'retire': {'training_state_u16170.pt': {'sha256': 'b29be8302f782b0d3a596b364b91590aa1b073257c2e8c49166e67570245becb', 'bytes': 1588159356}}, 'retain': {'final_trainable.pt': {'sha256': 'f091d0390087a6c07790d1b4a2420f41b982be7d477c229e8ae07ab593ab431f', 'bytes': 529404066}}}}


def checked(root,relative,regular=True):
    rel=PurePosixPath(relative)
    if rel.is_absolute() or '..' in rel.parts or '\\' in relative or not rel.parts:raise ValueError('Unsafe relative path')
    p=root.joinpath(*rel.parts)
    if p.resolve()!=p or root not in p.parents:raise ValueError('Symlink/escape path: '+relative)
    if p.exists() and regular and not stat.S_ISREG(p.lstat().st_mode):raise ValueError('Expected regular file: '+relative)
    return p


def protected(relative):
    return relative in ENDPOINTS or any(relative==p or relative.startswith(p+'/') for p in PROTECTED_PREFIXES)


def stamp(p):
    s=p.lstat()
    if not stat.S_ISREG(s.st_mode):raise ValueError('Expected regular file')
    return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_nlink]


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def jobs_guard():
    r=subprocess.run(['squeue','--me','--noheader','--format=%i|%T|%j'],check=True,capture_output=True,text=True)
    for line in r.stdout.splitlines():
        job,state,name=line.strip().split('|',2)
        if job!='233531' or name!='oxe-sample-pair':
            raise ValueError('Other queued/running job: '+line+'; no cleanup while its dependencies are unknown')
    return r.stdout.strip()


def cache_files(root,folder):
    p=checked(root,folder,regular=False)
    if not p.exists():return
    if not p.is_dir():raise ValueError('Expected cache directory: '+folder)
    for base,dirs,files in os.walk(p,followlinks=False):
        dirs[:]=[n for n in dirs if not (Path(base)/n).is_symlink()]
        for name in files:
            f=Path(base)/name
            if f.is_symlink():continue
            yield str(f.relative_to(root)).replace(os.sep,'/')


def unpacked_zip_exact(root,path):
    """Only discard an upload ZIP if every regular member is present, byte-identical."""
    try:
        with zipfile.ZipFile(path) as z:
            members=[m for m in z.infolist() if not m.is_dir()]
            if not members:return False
            names=[m.filename for m in members]
            if len(set(names))!=len(names):return False
            for m in members:
                rel=PurePosixPath(m.filename)
                # No loose root files, links, traversal or archive-in-archive aliases.
                if len(rel.parts)<2 or stat.S_ISLNK(m.external_attr>>16):return False
                target=checked(root,m.filename)
                if not target.is_file() or target.stat().st_size!=m.file_size:return False
                a=hashlib.sha256();b=hashlib.sha256()
                with z.open(m) as source,target.open('rb') as dest:
                    for block in iter(lambda:source.read(8*1024**2),b''):a.update(block)
                    for block in iter(lambda:dest.read(8*1024**2),b''):b.update(block)
                if a.digest()!=b.digest():return False
        return True
    except (ValueError,OSError,zipfile.BadZipFile,RuntimeError):return False


def build_plan(root):
    entries={};kept=[];guards={}
    def add(relative,reason,expected=None):
        if protected(relative):raise ValueError('Protected candidate: '+relative)
        p=checked(root,relative)
        if not p.exists():return
        entries[relative]={'path':relative,'reason':reason,'stamp':stamp(p),'bytes':p.stat().st_size,
                           'allocated_bytes':getattr(p.stat(),'st_blocks',0)*512,'expected_sha256':expected}
    for folder in PIP_CACHE_DIRS+('.tmp/conda/cache',):
        for rel in cache_files(root,folder):add(rel,'Downloaded pip wheel/HTTP cache or Conda index; installed environment stays')
    conda=checked(root,'.tmp/conda',regular=False)
    if conda.is_dir():
        for p in sorted(conda.iterdir()):
            if p.is_file() and not p.is_symlink() and (p.name.endswith('.tar.bz2') or p.name.endswith('.conda')):
                add('.tmp/conda/'+p.name,'Downloaded Conda package archive; extracted package/runtime stays')
    for rel,reason,group,expected in FIXED:
        if not checked(root,rel).exists():continue
        if group is None:
            add(rel,reason,expected);continue
        final=GROUP_ENDPOINT[group]
        if not checked(root,final).is_file():kept.append({'path':rel,'reason':'Retained final model missing: '+final});continue
        guards[final]=ENDPOINTS[final];add(rel,reason,expected)
    for folder,spec in BRIDGE_PROFILES.items():
        if not any(checked(root,folder+'/'+n).exists() for n in spec['retire']):continue
        required={folder+'/report.json':spec['report_sha256']}
        required.update({folder+'/'+n:i['sha256'] for n,i in spec['retain'].items()})
        if not all(checked(root,n).exists() for n in required):
            kept.append({'path':folder,'reason':'Missing qualified standalone Bridge inference weights/report'});continue
        guards.update(required)
        for name,info in spec['retire'].items():add(folder+'/'+name,'Obsolete Adam snapshot; standalone final weights retained',info['sha256'])
    for p in sorted(root.glob('*.zip')):
        if p.is_symlink():continue
        rel=p.name
        if p.stat().st_size>=128*1024**2:print('CHECK_DUPLICATE_ZIP',rel,flush=True)
        if unpacked_zip_exact(root,p):add(rel,'Upload ZIP with every member verified identical to extracted files')
        else:kept.append({'path':rel,'reason':'Not fully identical to retained extracted files; archive kept'})
    for rel in ENDPOINTS:
        if rel in entries:raise ValueError('Endpoint targeted')
    return list(entries.values()),guards,kept


def execute(root,entries,guards,kept,guard=jobs_guard):
    guard();retained={}
    # Complete all identity and path checks before the first unlink.
    for rel,expected in guards.items():
        p=checked(root,rel);print('VERIFY_RETAINED',rel,flush=True)
        if sha(p)!=expected:raise ValueError('Retained model/report changed: '+rel)
        retained[rel]={'sha256':expected,'stamp':stamp(p)}
    # The two immutable source files required by the active OXE job are always protected.
    for rel in ['runs/oxe-pose-pair-232587/real_pose/latest_state.pt','runs/oxe-target-short-231297/sample/latest_step.pt']:
        p=checked(root,rel)
        if not p.is_file():raise ValueError('Current OXE source missing: '+rel)
        retained[rel]={'stamp':stamp(p)}
    protected_inodes={tuple(v['stamp'][:2]) for v in retained.values()}
    for row in entries:
        rel=row['path'];p=checked(root,rel)
        if protected(rel) or stamp(p)!=row['stamp']:raise ValueError('Candidate changed/protected: '+rel)
        if tuple(row['stamp'][:2]) in protected_inodes:raise ValueError('Candidate shares retained model inode: '+rel)
        if row['expected_sha256'] and sha(p)!=row['expected_sha256']:raise ValueError('Obsolete checkpoint differs: '+rel)
    guard()
    audit=checked(root,'storage_cleanup_receipts',regular=False);audit.mkdir(exist_ok=True)
    ledger=audit/('reclaim_20261029_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]+'.jsonl')
    def write(stream,value):stream.write(json.dumps(value,ensure_ascii=False)+'\n');stream.flush();os.fsync(stream.fileno())
    deleted=0;logical=0;allocated=0
    with ledger.open('x',encoding='utf-8') as out:
        write(out,{'type':'start','root':str(root),'retained':retained,'candidate_count':len(entries),'kept':kept,
            'scope':'Declared rebuildable caches/superseded states/verified duplicate ZIPs. No final models, raw data, source code or reports deleted.',
            'consequence':'Old cached-training/replay tools may require rebuilding their retired token cache before reuse; original receipts remain historical.'})
        for i,row in enumerate(entries):
            if i%100==0 or row['bytes']>=256*1024**2:guard()
            p=checked(root,row['path'])
            now=stamp(p)
            # Earlier planned aliases can reduce nlink without changing inode/content.
            if now[:4]!=row['stamp'][:4]:raise ValueError('Candidate changed during cleanup; see receipt: '+row['path'])
            write(out,{'type':'intent','entry':row})
            p.unlink()
            deleted+=1;logical+=row['bytes']
            # Hardlinked files only release their blocks once their final link is retired.
            if now[4]==1:allocated+=row['allocated_bytes']
            write(out,{'type':'deleted','path':row['path']})
            if row['bytes']>=256*1024**2:print('REMOVED',round(row['bytes']/1024**3,3),'GiB',row['path'],flush=True)
        for rel,old in retained.items():
            if stamp(checked(root,rel))!=old['stamp']:raise ValueError('Retained source changed unexpectedly: '+rel)
        result={'type':'complete','deleted_files':deleted,'logical_gib':logical/1024**3,
            'estimated_unlinked_allocated_gib':allocated/1024**3,'receipt':str(ledger),'gpu_requested':False,'job_cancelled':False}
        write(out,result)
    print('CLEANUP_COMPLETE',json.dumps(result),flush=True);return result


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--apply',action='store_true');a=ap.parse_args()
    root=Path('/projects/Zeroshot').resolve(strict=True)
    if str(root)!='/projects/_ssd/Zeroshot':raise ValueError('Unexpected canonical project root')
    print('SLURM_ALLOWED_ONLY_233531',jobs_guard(),flush=True)
    entries,guards,kept=build_plan(root)
    totals={}
    for row in entries:totals[row['reason']]=totals.get(row['reason'],0)+row['bytes']
    print('PLANNED_LOGICAL_GIB',round(sum(r['bytes'] for r in entries)/1024**3,3),flush=True)
    print('PLAN_CATEGORIES_GIB',json.dumps({k:round(v/1024**3,3) for k,v in totals.items()}),flush=True)
    for row in kept:print('KEPT',row['path'],row['reason'],flush=True)
    if a.apply:execute(root,entries,guards,kept)
    else:print('READ_ONLY_PLAN; --apply executes the already-authorized declared cleanup')


if __name__=='__main__':main()

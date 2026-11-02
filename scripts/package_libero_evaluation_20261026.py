"""Freeze selected100/task protocol and a small simulator-only deployment."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
PACK=ROOT/'deployment/libero_original_evaluation_v1'
AUTHOR=ROOT/'training_cache/libero_author_8f1084e'
REPORT=ROOT/'reports/libero_evaluation_protocol_20261026'
sys.path.insert(0,str(PACK))
from evaluation_core import digest,file_sha


def write(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')


def main():
    assert not (PACK/'protocol.json').exists(),'Preserve frozen protocol/package; use another version for changes'
    checks=json.loads((REPORT/'ledger_checks.json').read_text())
    assert checks['passed']
    # Preserve the exact main trained implementations, plus the separately tested inference loader.
    code=PACK/'code'
    code.mkdir(exist_ok=True)
    for p in (ROOT/'code').glob('*.py'):
        shutil.copyfile(p,code/p.name)
    for old,new in [('task_roster.json','task_roster.json'),('episode_slots_candidate100.json','episode_slots.json')]:
        shutil.copyfile(REPORT/old,PACK/new)
    shutil.copyfile(ROOT/'reports/oxe_noaug_seed42_227223/cluster/preflight.json',PACK/'training_frozen_profile.json')
    write(PACK/'seed42_policy_admission.json',{
        'condition':'no_augmentation','training_seed':42,'completed_updates':65536,
        'file_sha256':'33df7eb93ff0d0f7d4ed470643f45ccabb3c8d2c001dc81396f4b66f2c6516d3',
        'learned_state_sha256':'2295d8add3e735a7340eb8e244fbc1d4b6abc09d95692120fde1573303abd66d',
        'training_contract_sha256':'b3b59ab65e0c422d5de774630749bc27afb943edf4fdfd854a2a3cd96daf76fc',
        'verification_evidence':'Independent complete downloaded endpoint audit; reports/oxe_noaug_seed42_227223/download_verification.json'})
    # Full official environment source+assets, no .git, demos, HDF5, or model weights.
    files={}
    for top in ('libero','setup.py','README.md','LICENSE','requirements.txt'):
        p=AUTHOR/top
        candidates=sorted(p.rglob('*')) if p.is_dir() else [p]
        for f in candidates:
            if not f.is_file() or '__pycache__' in f.parts or f.suffix in ('.pyc','.hdf5') or 'datasets' in f.parts:
                continue
            files[f.relative_to(AUTHOR).as_posix()]=file_sha(f)
    write(PACK/'simulator_files.json',files)
    simzip=ROOT/'cluster_packages/libero_simulator_assets_8f1084e_v1.zip'
    assert not simzip.exists()
    with zipfile.ZipFile(simzip,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name in files:z.write(AUTHOR/name,name)
    write(PACK/'simulator_archive.json',{'file':simzip.name,'sha256':file_sha(simzip),
        'bytes':simzip.stat().st_size,'uncompressed_bytes':sum((AUTHOR/n).stat().st_size for n in files),
        'official_commit':'8f1084e3132a39270c3a13ebe37270a43ece2a01','demonstrations_included':False})
    candidate=json.loads((REPORT/'protocol_candidate.json').read_text())
    p=copy.deepcopy(candidate)
    p.pop('candidate_identity')
    p.update(format='libero-original-instruction-evaluation-v1',status='frozen_user_selected100_per_task',
        formal_task_scoring_admitted=True,score_generated=False,
        candidate_parent_identity=candidate['candidate_identity'])
    p['rollouts']={'selection':'per_task100','episodes_per_task':100,'episodes_per_suite':1000,
        'original_instruction_episodes_per_model':4000,'models_planned':13,
        'original_instruction_model_episodes_planned':52000,'separate_baseline_episodes':4000,
        'counts_apply_to_original_instruction_only':True,
        'authorization':'User explicit answer on2026-10-26:100 per individual task,4000 per model'}
    p['conditions']={k:{'visual_backbone':b,'training_seeds':s} for k,b,s in [
        ('no_augmentation','dinov2',[42,43,44]),('paraphrase_only','dinov2',[42,43,44]),
        ('counterfactual_only','dinov2',[42,43,44]),('combined','dinov2',[42,43,44]),('clip_combined','clip',[42])]}
    p['runtime_strategy']={'primary':'Cluster Linux EGL; isolated simulator venv inherits existing training torch',
        'first_segment':'One complete fixed slot counts toward4000; checks native Linux backend and measures full rollout time',
        'segmentation':'Operational max-new-episodes/wallclock do not alter per-episode600tick budget',
        'exact_runtime_binding':'Refuse resume across changed packages/GPU class/driver/software/renderer; no silent mixed-runtime results',
        'local_engineering':'Existing32tick excluded engineering result retained; not repeated or counted',
        'batching':'Unchanged verified B1 sampler; no unverified fast/batched/DDIM path',
        'cross_platform_physics_qualified':False}
    p['evaluator_files_sha256']={f.name:file_sha(f) for f in PACK.glob('*.py')}
    p['policy_source_sha256']={f.name:file_sha(f) for f in code.glob('*.py')}
    p['simulator_files_sha256']=digest(files)
    p['episode_slots_sha256']=file_sha(PACK/'episode_slots.json')
    p['remaining_before_scoring']=['Cluster isolated simulator install and first actual Linux episode; no automatic job submission']
    p['separate_future_work']=['Reviewed evaluation paraphrases and separate rollout budget; no paraphrase score claim yet',
        'Three independently trained checkpoints for each main condition; no invented seed variation']
    p['protocol_identity']=digest(p)
    write(PACK/'protocol.json',p)
    write(REPORT/'protocol_frozen.json',p)
    # Shell scripts must have Unix LF, even when edited in Windows.
    for f in PACK.glob('*.sh'):
        f.write_bytes(f.read_bytes().replace(b'\r\n',b'\n'))
    manifest={f.relative_to(PACK).as_posix():file_sha(f) for f in sorted(PACK.rglob('*'))
        if f.is_file() and '__pycache__' not in f.parts and f.name!='SHA256SUMS'}
    (PACK/'SHA256SUMS').write_bytes(''.join(v+'  '+k+'\n' for k,v in manifest.items()).encode())
    target=ROOT/'cluster_packages/libero_original_evaluation_v1.zip'
    assert not target.exists()
    with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for name in [*manifest,'SHA256SUMS']:z.write(PACK/name,PACK.name+'/'+name)
    write(REPORT/'package_report.json',{'protocol_identity':p['protocol_identity'],'model_episodes':52000,
        'per_model_episodes':4000,'separate_baseline':4000,'ledger_checks_passed':True,
        'packages':[{'path':str(f),'bytes':f.stat().st_size,'sha256':file_sha(f)} for f in (simzip,target)],
        'new_cluster_jobs_submitted':0,'real_formal_episodes':0})
    print((REPORT/'package_report.json').read_text())


if __name__=='__main__':main()

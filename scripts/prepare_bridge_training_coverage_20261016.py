"""Prepare one fixed-budget training-coverage experiment from existing inventories."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys
import zipfile
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
PACK=ROOT/'cluster_packages/bridge_training_coverage_v1'

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def write(p,v): p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8',newline='\n')
def key(v): return v['shard'],v['record_index'],v['start_index']


def prepare():
    import torch
    assert not (PACK/'protocol.json').exists(), 'Preserve frozen preparation'
    PACK.mkdir(exist_ok=True)
    old=read(ROOT/'cluster_packages/bridge_growth128_v1/manifest.json')
    pool_path=ROOT/'training_cache/exports/bridge_expansion_training_v1/manifest.json'
    pool=read(pool_path)
    audit_path=ROOT/'reports/bridge_expansion_source_audit_verified_v1.json'
    audit=read(audit_path)
    assert audit['passed'] and sha(audit_path)==pool['source_audit_sha256']
    assert audit['dataset_source_sha256']==old['source_sha256']['dataset.py']
    episode_audit={(v['shard'],v['record_index']):v for v in audit['episodes']}
    all_train=[]
    for e in pool['partitions']['train']:
        check=episode_audit[(e['shard'],e['record_index'])]
        starts=list(range(0,e['steps']-16,4)); assert starts==check['starts']
        for start in starts:
            all_train.append(dict(shard=e['shard'],record_index=e['record_index'],start_index=start,
                                  task=e['instruction'],partition='train'))
    retained={key(r) for r in old['train']}
    assert retained <= set(map(key,all_train))
    rows=[{k:r[k] for k in ['shard','record_index','start_index','task','partition']} for r in old['train']]
    rows += [r for r in all_train if key(r) not in retained]
    assert len(rows)==1836 and len(set(map(key,rows)))==1836
    rows += [{k:r[k] for k in ['shard','record_index','start_index','task','partition']} for r in old['development']]
    ep=lambda r:(r['shard'],r['record_index'])
    assert len(set(map(ep,rows[:1836])))==405 and len(set(map(ep,rows[1836:])))==10
    assert not set(map(ep,rows[:1836])) & set(map(ep,rows[1836:]))
    assert set(map(ep,rows[1836:]))==set(map(ep,pool['partitions']['validation']))
    old_batches=torch.cat([torch.randperm(128,generator=torch.Generator().manual_seed(440001+c)).reshape(16,8) for c in range(512)]).numpy()
    assert hashlib.sha256(old_batches.tobytes()).hexdigest()==old['schedule_sha256']['mixed']
    tasks=np.array([r['task'] for r in rows]); requested=tasks[old_batches]
    streams={}; cursors=Counter()
    for ti,task in enumerate(sorted(set(tasks[:1836]))):
        indices=np.flatnonzero(tasks[:1836]==task).tolist(); stream=[]; cycle=0
        needed=int((requested==task).sum())+8
        while len(stream)<needed:
            order=indices.copy(); random.Random(460001+ti*100000+cycle).shuffle(order)
            stream.extend(order); cycle+=1
        streams[task]=stream
    schedule=np.empty((8192,8),np.int64)
    for update,labels in enumerate(requested):
        used=set()
        for j,task in enumerate(labels):
            stream=streams[task]; cursor=cursors[task]
            if stream[cursor] in used:
                at=next(i for i in range(cursor+1,len(stream)) if stream[i] not in used)
                stream[cursor],stream[at]=stream[at],stream[cursor]
            schedule[update,j]=stream[cursor];used.add(stream[cursor]);cursors[task]+=1
    assert np.array_equal(tasks[schedule],requested)
    assert all(len(set(v))==8 for v in schedule)
    counts=np.bincount(schedule.ravel(),minlength=1836);assert counts.min()>0
    np.savez_compressed(PACK/'schedule.npz',batches=schedule)
    write(PACK/'rows.json',dict(train=rows[:1836],development=rows[1836:]))
    write(PACK/'episode_reference.json',dict(episodes=audit['episodes'],
        source_audit_sha256=sha(audit_path),dataset_source_sha256=audit['dataset_source_sha256']))
    inherited=read(ROOT/'cluster_packages/bridge_image_augmentation_v1/protocol.json')['inherited_sha256']
    for path,h in inherited.items(): assert sha(PACK.parent/path)==h
    protocol=dict(date='2026-10-16',objective='Same-recipe fixed-budget generalization versus training coverage.',
        inherited_sha256=inherited,control_report_sha256=sha(ROOT/'reports/bridge-state-fit-202278/real/report.json'),
        pool_manifest_sha256=sha(pool_path),source_audit_sha256=sha(audit_path),
        rows_sha256=sha(PACK/'rows.json'),episode_reference_sha256=sha(PACK/'episode_reference.json'),
        schedule_file_sha256=sha(PACK/'schedule.npz'),schedule_tensor_sha256=hashlib.sha256(schedule.tobytes()).hexdigest(),
        config=old['config'],counts=dict(train_windows=1836,train_episodes=405,development_windows=40,development_episodes=10),
        retained128_indices=list(range(128)),state_statistics='Original128 training observations only, unchanged protocol and features.',
        recipe=dict(initialization_seed=43,batch_size=8,updates=8192,learning_rates=[1e-4,1e-5],switch_after=4096,
                    sampling_seeds=[1101,1102,1103],sampling_steps=100,optimizer='Adam',weight_decay=0,gradient_clip=1),
        schedule=dict(method='Original8192x8 task labels preserved exactly; fixed per-task shuffled streams; avoid repeated row within a batch.',
            task_draw_counts=dict(Counter(requested.ravel())),minimum_window_draws=int(counts.min()),maximum_window_draws=int(counts.max()),
            mean_window_draws=float(counts.mean())),
        screen=read(ROOT/'cluster_packages/bridge_image_augmentation_v1/protocol.json')['screen'],
        assessment='Original40 development pose metrics primary; all10episodes/5tasks. Final full training pool diagnostic only, no millimetre fit requirement or extension.',
        old_control_retrained=False,no_augmentation=True,gripper_frozen=True,
        existing_data='Reuse bridge_expansion_training_v1/data on cluster; no download or local full token cache.',
        input_proof='Reuse415-episode complete payload/geometry audit with identical frozen dataset.py and exact starts. Verify payload hashes at runtime; original168 observations/labels additionally use frozen numeric references.',
        local_preflight='Only needed first8training batches plus original168 inputs. Allocate representative full GPU token array, encode newly needed rows only, no full local cache. Cannot authorize formal run.',
        cluster_preflight='All415episodes/1876windows validated and complete token cache built once;16operational updates plus4training-window probes; no development scoring.',
        resource_gate=dict(peak_reserved_gib_below=7.5,formal_estimate_seconds_below=9000,cache_budget_bytes=3*1024**3,
                           extra_storage_plan_gib=5,minimum_project_headroom_gib=15,project_planning_limit_gib=200),
        stopping='One initialization/coverage pool/8192update budget/final only. No window ladder or extra steps based on training fit. Reused development set,not final test.',
        limits=['Coverage includes more demonstrations AND more within-demo windows; not pure episode-count causal effect.',
                'Fixed compute changes exposure per window; no claim about all budgets.','Frozen gripper; offline measured pose,not grasp.'])
    write(PACK/'protocol.json',protocol)
    print(json.dumps(dict(rows=len(rows),train=1836,retained=128,all_task_slots_match=True,
        draws_min=int(counts.min()),draws_max=int(counts.max()),protocol_sha256=sha(PACK/'protocol.json')),indent=2))


def package():
    report_path=ROOT/'reports/bridge_training_coverage_local_preflight_v2_20261016/report.json'
    report=read(report_path)
    audit=read(report_path.with_name('independent_audit.json'))
    assert report['execution_checks_passed'] and report['branch_restore_passed'] and report['local_bounded'] and audit['verified']
    assert report['peak_reserved_gib']<7.5
    write(PACK/'local_preflight_summary.json',dict(report_sha256=sha(report_path),audit_sha256=sha(report_path.with_name('independent_audit.json')),
        local_bounded=True,operational_updates=16,development_scored=False,full_pool_validated_locally=False,
        peak_reserved_gib=report['peak_reserved_gib'],elapsed_seconds=report['elapsed_seconds']))
    files=sorted(p for p in PACK.iterdir() if p.is_file() and p.name!='SHA256SUMS')
    assert all(p.suffix in {'.py','.json','.npz','.sh','.md'} for p in files)
    (PACK/'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.name}\n' for p in files),encoding='ascii',newline='\n')
    archive=PACK.with_suffix('.zip');assert not archive.exists()
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in files+[PACK/'SHA256SUMS']:z.write(p,p.relative_to(PACK.parent).as_posix())
    with zipfile.ZipFile(archive) as z: assert z.testzip() is None and all(z.read(n)==(PACK.parent/n).read_bytes() for n in z.namelist())
    assert archive.stat().st_size<1024**2
    PACK.with_suffix('.sha256').write_text(f'{sha(archive)}  {archive.name}\n',encoding='ascii',newline='\n')
    print('PACKAGE',archive.stat().st_size,sha(archive))


def launchers():
    assert (PACK/'protocol.json').is_file()
    for stage in ['preflight','fit']:
        path=PACK/f'{stage}.sh';assert not path.exists()
        script=f'''#!/bin/bash
#SBATCH --job-name=bridge-coverage-{stage}
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=22528M
#SBATCH --time={'00:30:00' if stage=='preflight' else '03:00:00'}
#SBATCH --chdir=/projects/Zeroshot
#SBATCH --output=/projects/Zeroshot/logs/bridge-coverage-{stage}-%j.out
set -eo pipefail
source /projects/Zeroshot/scripts/ssd_cache.sh
export TMPDIR="/projects/Zeroshot/.tmp/${{SLURM_JOB_ID}}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 USE_TF=0 OMP_NUM_THREADS=4
mkdir -p "$TMPDIR" /projects/Zeroshot/runs
'''
        if stage=='fit':script+=': "${COVERAGE_PREFLIGHT_JOB:?Supply successful COMPLETE cluster preflight ID}"\n'
        script+='''PROJECT_USED_BYTES=$(du -s -B1 /projects/_ssd/Zeroshot | cut -f1)
if (( PROJECT_USED_BYTES + 5*1024*1024*1024 + 15*1024*1024*1024 > 200*1024*1024*1024 )); then echo "Storage planning headroom insufficient"; exit 1; fi
cd /projects/Zeroshot/bridge_training_coverage_v1
sha256sum -c SHA256SUMS
nvidia-smi
'''
        if stage=='fit':script+='/projects/Zeroshot/envs/multitask-preflight-v1/bin/python validate_preflight.py "/projects/Zeroshot/runs/bridge-coverage-preflight-${COVERAGE_PREFLIGHT_JOB}"\n'
        command=['/projects/Zeroshot/envs/multitask-preflight-v1/bin/python -u run_training_coverage.py',
            '  --data-root /projects/Zeroshot/bridge_expansion_training_v1/data',
            '  --data-reference /projects/Zeroshot/bridge_batch_input_audit_v1/reference',
            '  --clip /projects/Zeroshot/baseline_setup/clip-vit-large-patch14-v1',
            '  --old-cache /projects/Zeroshot/bridge_state_prior_v1/full_tokens.pt',
            '  --cache /projects/Zeroshot/bridge_training_coverage_v1/full_tokens.pt',
            '  --control-report /projects/Zeroshot/runs/bridge-state-fit-202278/real/report.json',
            f'  --stage {stage} --output "/projects/Zeroshot/runs/bridge-coverage-{stage}-${{SLURM_JOB_ID}}"']
        if stage=='fit':command.append('  --preflight-reference "/projects/Zeroshot/runs/bridge-coverage-preflight-${COVERAGE_PREFLIGHT_JOB}"')
        script+=(' '+chr(92)+'\n').join(command)+'\n'
        assert not any(v.startswith('+') for v in script.splitlines())
        path.write_text(script,encoding='utf-8',newline='\n')
    print('LAUNCHERS WRITTEN; NO JOB SUBMITTED')


if __name__=='__main__':
    assert sys.argv[1:] in [['prepare'],['package'],['launchers']]
    globals()[sys.argv[1]]()

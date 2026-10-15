"""Freeze one bounded author-Bridge reference after verified resource preflight203498."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
import zipfile
import yaml

ROOT=Path(__file__).resolve().parents[1]
PACK=ROOT/'cluster_packages/bridge_author_reference_v1'
PREPACK=ROOT/'cluster_packages/bridge_author_preflight_v1'
PRE=ROOT/'reports/bridge-author-preflight-203498'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def prepare():
    assert not (PACK/'training_protocol.json').exists()
    v=read(PRE/'verification.json');r=read(PRE/'author/report.json')
    assert v['passed'] and r['passed'] and v['preflight_report_sha256']==sha(PRE/'author/report.json')
    old=read(PREPACK/'protocol.json')
    cfg=yaml.safe_load((PREPACK/'preflight_config.yaml').read_text(encoding='utf-8'));cfg=copy.deepcopy(cfg)
    cfg['name']='bridge_author_reference';cfg['logging']['name']='fixed_final_same_bridge_reference'
    cfg['training'].update(num_epochs=512,max_train_steps=None,max_val_steps=1,
        sample_every=5,val_every=16,rollout_every=512,checkpoint_every=512,tqdm_interval_sec=30.)
    assert cfg['policy']==yaml.safe_load((PREPACK/'preflight_config.yaml').read_text(encoding='utf-8'))['policy']
    (PACK/'training_config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8',newline='\n')
    p=dict(date='2026-10-14',stage='same_bridge_author_reference',preflight_job=203498,
        preflight_report_sha256=sha(PRE/'author/report.json'),input_export_report_sha256=sha(PRE/'inputs/export_report.json'),
        preflight_verification_sha256=sha(PRE/'verification.json'),input_protocol_sha256=sha(PREPACK/'protocol.json'),
        reused_adapter_source_sha256={n:sha(PREPACK/n) for n in ('bridge_author_io.py','bridge_author_dataset.py')},
        rows=old['rows'],sampling_seeds=[1101,1102,1103],initialization_seed=43,
        training=dict(epochs=512,batch=8,updates=8192,draw_slots_per_window=512,resume=False,
            optimizer=cfg['optimizer'],warmup_steps=500,ema_primary=True,policy=cfg['policy']),
        evaluation=dict(final_only=True,primary='ema',secondary='raw',train_windows=128,development_windows=40,
            train_fit='Unchanged project gate: >=90% windows pass all3 draws at <=0.5cm/5deg; per-draw overall<=0.5cm/5deg, endpoint<=1cm/10deg',
            useful_reference='PrimaryEMA fits train; development path position AND rotation below static; >=6/10 demos position below static',
            no_checkpoint_or_sampling_seed_selection=True,no_gripper_or_rollout_success=True),
        budget=dict(time_limit_seconds=10800,gate_max_estimate_seconds=10200,
            first8_updates_timed_within8192=True,planning_inflation=1.25,checkpoint_limit_bytes=3*2**30,
            peak_reserved_gib_limit=40,planning_project_limit_bytes=200000000000,min_planning_headroom_gib=15),
        changes_from_project=old['changes_from_project'],limits=old['limitations'],
        stopping_rules=['If measured budget estimate fails, stop before further updates; no checkpoint and no automatic larger time request.',
            'Fixed512 epochs only, no extra initialization or dataset expansion.',
            'If primaryEMA does not fit, this budget has not established an effective reference; do not attribute failure to a unique project module.',
            'If it fits but development fails, report the same-data generalization limit; do not select raw or a different epoch as primary afterward.',
            'If effective, design a matched module-boundary experiment next; full-pipeline difference alone cannot locate the cause.'])
    (PACK/'training_protocol.json').write_text(json.dumps(p,indent=2)+'\n',encoding='utf-8',newline='\n')
    shell='''#!/bin/bash
#SBATCH --job-name=bridge-author-reference
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=22528M
#SBATCH --time=03:00:00
#SBATCH --chdir=/projects/Zeroshot
#SBATCH --output=/projects/Zeroshot/logs/bridge-author-reference-%j.out
set -eo pipefail
source /projects/Zeroshot/scripts/ssd_cache.sh
export TMPDIR="/projects/Zeroshot/.tmp/author-reference-${SLURM_JOB_ID}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export OMP_NUM_THREADS=4 SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy
export MPLCONFIGDIR="/projects/Zeroshot/.tmp/author-reference-mpl"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" /projects/Zeroshot/runs
PROJECT_BYTES=$(du -sb /projects/_ssd/Zeroshot | cut -f1)
if (( PROJECT_BYTES + 4*1024*1024*1024 + 15*1024*1024*1024 > 200000000000 )); then exit 1; fi
SOURCE=/projects/Zeroshot/paper_reproduction/official_pusht_v1/source
AUTHOR_ENV=/projects/Zeroshot/envs/official-pusht-v1
test "$(git -C "$SOURCE" rev-parse HEAD)" = 5ba07ac6661db573af695b419a7947ecb704690f
test -z "$(git -C "$SOURCE" status --porcelain)"
cd /projects/Zeroshot/bridge_author_reference_v1
sha256sum -c SHA256SUMS
nvidia-smi
LD_LIBRARY_PATH="$AUTHOR_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" "$AUTHOR_ENV/bin/python" -u run_bridge_author_reference.py \\
  --project /projects/Zeroshot --source "$SOURCE" --output "/projects/Zeroshot/runs/bridge-author-reference-${SLURM_JOB_ID}"
'''
    (PACK/'train.sh').write_text(shell,encoding='utf-8',newline='\n')
    print('Prepared one fixed-final author reference; input cache reused, no optimizer checkpoints.')

def package():
    proof=read(ROOT/'reports/bridge_author_reference_preparation_20261014.json')
    assert proof['passed'] and proof['training_protocol_sha256']==sha(PACK/'training_protocol.json')
    for f in PACK.glob('*.py'):ast.parse(f.read_text(encoding='utf-8'),feature_version=(3,9))
    (PACK/'local_preparation_summary.json').write_text(json.dumps(proof,indent=2)+'\n',encoding='utf-8')
    files=sorted(p for p in PACK.iterdir() if p.is_file() and p.name!='SHA256SUMS')
    (PACK/'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.name}\n' for p in files),encoding='ascii',newline='\n');files.append(PACK/'SHA256SUMS')
    archive=PACK.with_suffix('.zip');assert not archive.exists()
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in files:z.write(p,p.relative_to(PACK.parent).as_posix())
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None and all(z.read(p.relative_to(PACK.parent).as_posix())==p.read_bytes() for p in files)
    assert archive.stat().st_size<100*2**20
    PACK.with_suffix('.sha256').write_text(f'{sha(archive)}  {archive.name}\n',encoding='ascii',newline='\n')
    print(json.dumps(dict(bytes=archive.stat().st_size,sha256=sha(archive),files=len(files))))
if __name__=='__main__':
    assert len(sys.argv)==2 and sys.argv[1] in ('prepare','package')
    globals()[sys.argv[1]]()

"""Prepare/freeze the predeclared9-window frozen-author input intervention."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[1]
PACK=ROOT/'cluster_packages/bridge_author_observation_probe_v1'
AUDIT=ROOT/'reports/bridge_broccoli_observation_audit_203699'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))

def prepare():
    assert not (PACK/'protocol.json').exists()
    plan=read(AUDIT/'frozen_observation_intervention_plan.json');a=read(AUDIT/'analysis.json')
    assert a['passed'] and len(plan['donors'])==9 and plan['updates']==0
    donor_rows=[x for x in a['windows'] if x['partition']=='development']
    assert plan['donors']==[dict(window=x['window'],donor_window=x['neighbors']['visual']['window']) for x in donor_rows]
    report=read(ROOT/'reports/bridge-author-reference-203699/report.json')
    assert report['execution_passed'] and report['primary']=='ema'
    assert plan['checkpoint_sha256']==next(x['sha256'] for x in report['checkpoints'] if x['name']=='ema_final.pt')
    required={}
    for n in ('bridge_author_preflight_v1/protocol.json','bridge_author_preflight_v1/bridge_author_dataset.py',
              'bridge_author_preflight_v1/bridge_author_io.py','bridge_author_reference_v1/training_protocol.json',
              'bridge_author_reference_v1/reference_metrics.py'):
        required[n]=sha(ROOT/'cluster_packages'/n)
    for n in ('report.json','runtime_config.yaml','ema_development.npz'):
        required['runs/bridge-author-reference-203699/'+n]=sha(ROOT/'reports/bridge-author-reference-203699'/n)
    required['runs/bridge-author-preflight-203498/inputs/export_report.json']=sha(ROOT/'reports/bridge-author-preflight-203498/inputs/export_report.json')
    p=copy.deepcopy(plan)
    p.update(stage='prepared_frozen_observation_intervention',torch_version=report['torch_version'],
        required_file_sha256=required,replay_tolerance=1e-4,
        tolerance_interpretation='Maximum raw action-component difference; XYZ units are10cm, so threshold corresponds to0.001cm per XYZ component',
        job_wall_seconds=600,sampler_calls=39,expected_output_limit_mib=5,
        local_model_executed=False,local_cuda_replay_verified=False,
        next_action='One short cluster inference job; no optimizer, training, package installation or checkpoint output')
    (PACK/'protocol.json').write_text(json.dumps(p,indent=2)+'\n',encoding='utf-8',newline='\n')
    shell='''#!/bin/bash
#SBATCH --job-name=bridge-author-obs
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=00:10:00
#SBATCH --chdir=/projects/Zeroshot
#SBATCH --output=/projects/Zeroshot/logs/bridge-author-obs-%j.out
set -eo pipefail
source /projects/Zeroshot/scripts/ssd_cache.sh
export TMPDIR="/projects/Zeroshot/.tmp/author-obs-${SLURM_JOB_ID}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export OMP_NUM_THREADS=4 SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy
export MPLCONFIGDIR="/projects/Zeroshot/.tmp/author-reference-mpl"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" /projects/Zeroshot/runs
SOURCE=/projects/Zeroshot/paper_reproduction/official_pusht_v1/source
AUTHOR_ENV=/projects/Zeroshot/envs/official-pusht-v1
test "$(git -C "$SOURCE" rev-parse HEAD)" = 5ba07ac6661db573af695b419a7947ecb704690f
test -z "$(git -C "$SOURCE" status --porcelain)"
cd /projects/Zeroshot/bridge_author_observation_probe_v1
sha256sum -c SHA256SUMS
nvidia-smi
LD_LIBRARY_PATH="$AUTHOR_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" "$AUTHOR_ENV/bin/python" -u probe_author_observations.py \\
  --project /projects/Zeroshot --source "$SOURCE" --output "/projects/Zeroshot/runs/bridge-author-obs-${SLURM_JOB_ID}"
'''
    (PACK/'probe.sh').write_text(shell,encoding='utf-8',newline='\n')
    print('Prepared fixed donor plan, original replay gate and10-minute inference launcher.')

def package():
    proof=read(ROOT/'reports/bridge_author_observation_probe_preparation.json')
    assert proof['passed'] and proof['protocol_sha256']==sha(PACK/'protocol.json')
    assert proof['bash_syntax_passed']
    for name,h in proof['source_sha256'].items():assert sha(PACK/name)==h
    for f in PACK.glob('*.py'):ast.parse(f.read_text(encoding='utf-8'),feature_version=(3,9))
    (PACK/'local_preparation.json').write_text(json.dumps(proof,indent=2)+'\n',encoding='utf-8')
    files=sorted(f for f in PACK.iterdir() if f.is_file() and f.name!='SHA256SUMS')
    (PACK/'SHA256SUMS').write_text(''.join(f'{sha(f)}  {f.name}\n' for f in files),encoding='ascii',newline='\n')
    files.append(PACK/'SHA256SUMS');archive=PACK.with_suffix('.zip');assert not archive.exists()
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for f in files:z.write(f,f.relative_to(PACK.parent).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert all(z.read(f.relative_to(PACK.parent).as_posix())==f.read_bytes() for f in files)
    assert archive.stat().st_size<2**20
    PACK.with_suffix('.sha256').write_text(f'{sha(archive)}  {archive.name}\n',encoding='ascii',newline='\n')
    print(json.dumps(dict(bytes=archive.stat().st_size,sha256=sha(archive),files=len(files))))

if __name__=='__main__':
    assert len(sys.argv)==2 and sys.argv[1] in ('prepare','package')
    globals()[sys.argv[1]]()

"""Reuse the executed LR runner, adding only the preregistered seeds42/44."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT/"cluster_packages/bridge_late_lr_replication_v1"
ENTRY = "check_bridge_late_lr_replication.py"


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old,new)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--package',action='store_true');args=ap.parse_args()
    if not args.package:
        assert not PACK.with_suffix('.zip').exists()
        original=ROOT/'cluster_packages/bridge_late_lr_pair_v1'
        manifest=copy.deepcopy(json.loads((original/'manifest.json').read_text()))
        PACK.mkdir(exist_ok=True)
        for name,h in manifest['source_sha256'].items():
            assert sha(original/name)==h
            shutil.copyfile(original/name,PACK/name)
        source=(original/'check_bridge_late_lr_pair.py').read_text()
        source=replace_once(source,'    ap.add_argument("--preflight-reference", type=Path)',
            '    ap.add_argument("--initialization-seed", type=int, choices=(42,44), required=True)\n'
            '    ap.add_argument("--preflight-reference", type=Path)')
        source=replace_once(source,'        init = 43; seed(init)','        init = args.initialization_seed; seed(init)')
        source=replace_once(source,'            assert pre["execution_checks_passed"]',
            '            assert pre["initialization_seed"] == args.initialization_seed\n'
            '            assert pre["execution_checks_passed"]')
        source=replace_once(source,'for k,v in row["origin_final_metric_differences"].items()) and not final["fit_gate"]["pose_fit_passed"]',
            'for k,v in row["origin_final_metric_differences"].items()) and final["fit_gate"]["pose_fit_passed"] == reference["final"]["train"]["fit_gate"]["pose_fit_passed"]')
        (PACK/ENTRY).write_text(source,newline='\n')
        manifest['source_sha256'][ENTRY]=sha(PACK/ENTRY)
        manifest.update(stage='preregistered_late_lr_replication',initialization_seeds=[42,44],
            changing_variable='Same fitted-state late LR comparison, replicated at remaining seeds42/44; recipe/rates unchanged')
        manifest['late_lr'].pop('initialization_seed')
        manifest['late_lr'].update(initialization_seeds=[42,44],
            control_reproduction='Match original final metrics AND original fit-gate verdict; seed44 control may pass',
            prior_seed43_report_sha256=sha(ROOT/'reports/bridge-late-lr-fit-200150/report.json'),
            replication_total_pt_budget_bytes=8*2**30)
        (PACK/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        for stage,limit in [('preflight','00:25:00'),('fit','02:00:00')]:
            reference='' if stage=='preflight' else ' \\\n    --preflight-reference "/projects/Zeroshot/runs/bridge-lr-rep-preflight-${LR_REP_PREFLIGHT_JOB}/seed$INIT"'
            guard='' if stage=='preflight' else ''': "${LR_REP_PREFLIGHT_JOB:?Supply successful replication preflight job ID}"
PROJECT_USED_BYTES=$(du -s -B1 /projects/_ssd/Zeroshot | cut -f1)
echo "Project apparent bytes (planning estimate, not live quota): $PROJECT_USED_BYTES"
if (( PROJECT_USED_BYTES + 8*1024*1024*1024 + 15*1024*1024*1024 > 200*1024*1024*1024 )); then
  echo "Insufficient planning headroom"; exit 1
fi
'''
            source=f'''#!/bin/bash
#SBATCH --job-name=bridge-lr-rep-{stage}
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=22528M
#SBATCH --time={limit}
#SBATCH --chdir=/projects/Zeroshot
#SBATCH --output=/projects/Zeroshot/logs/bridge-lr-rep-{stage}-%j.out
set -eo pipefail
source /projects/Zeroshot/scripts/ssd_cache.sh
export TMPDIR="/projects/Zeroshot/.tmp/${{SLURM_JOB_ID}}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 USE_TF=0 OMP_NUM_THREADS=4
mkdir -p "$TMPDIR" /projects/Zeroshot/runs
{guard}cd /projects/Zeroshot/bridge_late_lr_replication_v1
sha256sum -c SHA256SUMS
nvidia-smi
for INIT in 42 44; do
  /projects/Zeroshot/envs/multitask-preflight-v1/bin/python -u {ENTRY} \\
    --manifest manifest.json --data-root /projects/Zeroshot/bridge_expansion_training_v1/data \\
    --clip /projects/Zeroshot/baseline_setup/clip-vit-large-patch14-v1 \\
    --data-reference /projects/Zeroshot/bridge_batch_input_audit_v1/reference \\
    --origin-report /projects/Zeroshot/runs/bridge-trace-fit-200039/report.json \\
    --initialization-seed "$INIT" --stage {stage} \\
    --output "/projects/Zeroshot/runs/bridge-lr-rep-{stage}-${{SLURM_JOB_ID}}/seed$INIT"{reference}
done
echo "LR REPLICATION {stage.upper()} EXECUTION COMPLETE; inspect both seed reports"
'''
            (PACK/f'{stage}.sh').write_text(source,newline='\n')
        (PACK/'README.md').write_text('# Late LR replication on seeds42/44\n\n'
            'Repeat the seed43 comparison with unchanged data, rates and Adam state branching. '
            'Only add seed CLI and match the original fit verdict (seed44 constant may pass). '
            'Original ten source files remain unchanged. PT cap4GiB per seed,8GiB total, cluster only. '
            'No installs, model/data downloads or local weights. Preflight replays restoration for both seeds.\n')
        print('LR REPLICATION PREPARED',PACK);return
    summaries=[]
    for init in (42,44):
        f=ROOT/f'reports/bridge_lr_replication_local_preflight_20261012/seed{init}/report.json'
        r=json.loads(f.read_text())
        assert r['initialization_seed']==init and r['execution_checks_passed'] and r['branch_restore_passed']
        assert r['lr_intervention_changes_parameters'] and not r['checkpoints']
        assert r['manifest_sha256']==sha(PACK/'manifest.json')
        verification=f.with_name('verification.json')
        v=json.loads(verification.read_text())
        assert v['passed'] and v['initialization_seed']==init and v['report_sha256']==sha(f)
        summaries.append(dict(seed=init,report_sha256=sha(f),restore=r['branch_restore'],
            independent_verification_sha256=sha(verification),
            peak_reserved_gib=r['peak_reserved_gib'],elapsed_seconds=r['elapsed_seconds']))
    (PACK/'local_preflight_summary.json').write_text(json.dumps(dict(passed=True,seeds=summaries),indent=2)+'\n')
    files=sorted(p for p in PACK.iterdir() if p.is_file() and p.name!='SHA256SUMS')
    (PACK/'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.name}\n' for p in files),newline='\n')
    archive=PACK.with_suffix('.zip');assert not archive.exists()
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(PACK.iterdir()):
            if f.is_file():z.write(f,PACK.name+'/'+f.name)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert all(z.read(PACK.name+'/'+f.name)==f.read_bytes() for f in PACK.iterdir() if f.is_file())
    archive.with_suffix('.sha256').write_text(f'{sha(archive)}  {archive.name}\n',newline='\n')
    print('LR REPLICATION VERIFIED',archive,archive.stat().st_size,'bytes')


if __name__=='__main__':main()

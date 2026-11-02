"""Small additive formal launcher; never modifies the uploaded preflight package."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT=Path(__file__).resolve().parents[1]
PACK=ROOT/'cluster_packages/bridge_prefix_formal_v1'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    assert not PACK.exists(),'Do not overwrite a released package'
    pre=ROOT/'cluster_packages/bridge_prefix_pair_v1'
    for line in (pre/'SHA256SUMS').read_text().splitlines():
        expected,name=line.split('  ',1);assert sha(pre/name)==expected
    p=json.loads((pre/'protocol.json').read_text())
    evidence=json.loads((ROOT/'reports/bridge_prefix_formal_preparation_checks_20261017.json').read_text())
    assert evidence['passed']
    inputs=json.loads((ROOT/'reports/bridge_prefix_formal_input_checks_20261017.json').read_text())
    assert inputs['passed'] and inputs['formal_source_sha256']==sha(ROOT/'scripts/run_bridge_prefix_formal_v1.py')
    PACK.mkdir()
    for source,name in [('run_bridge_prefix_formal_v1.py','run_formal.py'),
        ('bridge_prefix_pair_metrics_v1.py','bridge_prefix_pair_metrics_v1.py'),
        ('verify_bridge_prefix_formal_v1.py','verify_formal.py')]:
        shutil.copy2(ROOT/'scripts'/source,PACK/name)
        ast.parse((PACK/name).read_text(encoding='utf-8'))
    contract=dict(preflight_job=209420,preflight_source_sums_sha256=sha(pre/'SHA256SUMS'),
        protocol_sha256=sha(pre/'protocol.json'),initial_checkpoint_sha256=p['initialization']['sha256'],
        user_reported_preflight_verification=dict(peak_reserved_gib=3.75,warm_update_seconds=.1722938776559507,
            estimated_paired_formal_seconds=2112.4922619849212),
        original_protocol_snapshot_not_updated=True,
        scope='Two4096-update continuations, final weights only, new10 dev, final25 closed',
        preflight_actual_report_validated_on_cluster_before_training=True,
        no_formal_job_submitted_locally=True)
    (PACK/'formal_contract.json').write_text(json.dumps(contract,indent=2)+'\n',encoding='utf-8')
    bash='''#!/bin/bash
#SBATCH --job-name=bridge-prefix-pair
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --chdir=/projects/Zeroshot
#SBATCH --output=/projects/Zeroshot/logs/bridge-prefix-formal-%j.out
set -euo pipefail
source /projects/Zeroshot/scripts/ssd_cache.sh
export TMPDIR="/projects/Zeroshot/.tmp/${SLURM_JOB_ID}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 USE_TF=0 OMP_NUM_THREADS=4
mkdir -p "$TMPDIR" /projects/Zeroshot/runs
PROJECT_USED_BYTES=$(du --apparent-size -s -B1 /projects/_ssd/Zeroshot | cut -f1)
if (( PROJECT_USED_BYTES + 4*1024*1024*1024 + 15*1024*1024*1024 > 200*1024*1024*1024 )); then
  echo "Planning headroom insufficient; no live quota claim"
  exit 1
fi
cd /projects/Zeroshot/bridge_prefix_formal_v1
sha256sum -c SHA256SUMS
nvidia-smi
/projects/Zeroshot/envs/multitask-preflight-v1/bin/python -u run_formal.py \\
  --project /projects/Zeroshot \\
  --output "/projects/Zeroshot/runs/bridge-prefix-formal-${SLURM_JOB_ID}"
'''
    submit='''#!/bin/bash
set -euo pipefail
CODE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$CODE"
exec 9>submit_formal.lock
flock -x 9
if [[ -s submitted_formal.job ]]; then
  printf 'Existing formal pair: %s\\n' "$(cat submitted_formal.job)"
  exit 0
fi
sha256sum -c SHA256SUMS
/projects/Zeroshot/envs/multitask-preflight-v1/bin/python \\
  /projects/Zeroshot/bridge_prefix_pair_v1/verify_prefix_preflight.py \\
  /projects/Zeroshot/runs/bridge-prefix-preflight-209420
JOB_REPLY=$(sbatch --parsable "$CODE/formal.sh")
JOB_NUMBER="${JOB_REPLY%%;*}"
printf '%s\\n' "$JOB_NUMBER" > submitted_formal.job
printf 'PREFIX_FORMAL: %s\\n' "$JOB_NUMBER"
'''
    (PACK/'formal.sh').write_text(bash,encoding='utf-8',newline='\n')
    (PACK/'submit_formal_once.sh').write_text(submit,encoding='utf-8',newline='\n')
    (PACK/'README.md').write_text('''# Fixed prefix coverage pair after verified209420

An additive launcher. Requires the unchanged bridge_prefix_pair_v1 directory,
its1558-row delta_visual_tokens.pt, and the ACTUAL complete209420 report.
No preflight rerun, original8192-prefix rerun, data expansion, downloads or installs.
Both arms reload the SAME207558 trained weights, zero request port and freshAdam.
Each4096 new updates, batch8, LR1e-5, oddK16/evenK1, original frozen matched draws.
ControlK1=old1836; candidateK1=all3394. CLIP/gripper frozen. Both final weights
saved and roundtrip-checked before any development scoring. No model selection.

Only the new10/50 development set is scored. Full16 (DDPM100,3 seeds) is primary;
K1 (also100 steps) is auxiliary. Original207558 and static use exactly this set.
Both aggregate weighting schemes, every episode/task, endpoint and first-target
errors are saved. Same predeclared admission gates, no revision after scoring.
No final25 payloads opened; even admission success does NOT evaluate them here.
Failure retains207558 and closes the branch without sweeps/repeated final tests.

Submit exactly once with bash submit_formal_once.sh. It verifies209420 first;
there is no afterok dependency on an expired Slurm job. A lock/id file avoids
duplicate submissions. No automatic retry/resume after a failed formal job.
Logs: /projects/Zeroshot/logs/bridge-prefix-formal-JOB.out
Output: /projects/Zeroshot/runs/bridge-prefix-formal-JOB/
Validate completed output with verify_formal.py RUN_DIRECTORY (CPU metrics only).
Do not download the two530MB weights or1.5GB feature cache just to review results.
''',encoding='utf-8',newline='\n')
    paths=sorted(p for p in PACK.iterdir() if p.is_file())
    (PACK/'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.name}\n' for p in paths),encoding='utf-8',newline='\n')
    archive=PACK.with_suffix('.zip');assert not archive.exists()
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for path in sorted(PACK.iterdir()):z.write(path,PACK.name+'/'+path.name)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for line in z.read(PACK.name+'/SHA256SUMS').decode().splitlines():
            expected,name=line.split('  ',1)
            assert hashlib.sha256(z.read(PACK.name+'/'+name)).hexdigest()==expected
    archive.with_suffix('.sha256').write_text(f'{sha(archive)}  {archive.name}\n')
    release=dict(archive=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),
        user_reported_preflight_verified=True,preflight_job=209420,
        original_preflight_package_unchanged=True,formal_GPU_executed_locally=False,
        final25_payloads_opened=False,CPU_preparation_checks_passed=True)
    (ROOT/'reports/bridge_prefix_formal_release_20261017.json').write_text(json.dumps(release,indent=2)+'\n')
    print(json.dumps(release,indent=2))


if __name__=='__main__':main()

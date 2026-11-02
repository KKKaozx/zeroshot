"""Release append-only formal code after user verification of210291."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT=Path(__file__).resolve().parents[1]
PRE=ROOT/'cluster_packages/bridge_spatial_readout_v1'
PREFIX=ROOT/'cluster_packages/bridge_prefix_pair_v1'
OUT=ROOT/'cluster_packages/bridge_spatial_formal_v1'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name,text):(OUT/name).write_bytes(text.replace('\r\n','\n').encode('utf-8'))


def main():
    archive=OUT.with_suffix('.zip')
    if OUT.exists() or archive.exists():raise FileExistsError('Preserve completed release')
    OUT.mkdir()
    files={
        'run_bridge_spatial_formal_v1.py':'run_formal.py',
        'bridge_spatial_pair_metrics_v1.py':'bridge_spatial_pair_metrics_v1.py',
        'bridge_spatial_validation_v1.py':'bridge_spatial_validation_v1.py',
        'verify_bridge_spatial_formal_v1.py':'verify_formal.py'}
    for source,name in files.items():shutil.copyfile(ROOT/'scripts'/source,OUT/name)
    metric_source=ROOT/'cluster_packages/bridge_prefix_formal_v1/bridge_prefix_pair_metrics_v1.py'
    text=metric_source.read_text();tree=ast.parse(text)
    selected=[n for n in tree.body if (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='METRICS' for t in n.targets))
              or (isinstance(n,ast.FunctionDef) and n.name=='summarize')]
    assert len(selected)==2
    write('base_pose_metrics.py','"""Unchanged float64 pose-summary code extracted from209481; no old admission logic."""\nimport numpy as np\n\n'+
          '\n\n'.join(ast.get_source_segment(text,n) for n in selected)+'\n')
    # Training arithmetic is EXACTLY the successful preflight calculate function.
    def calculate_ast(path):
        nodes=[n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n,ast.FunctionDef) and n.name=='calculate']
        assert len(nodes)==1;return ast.dump(nodes[0],include_attributes=False)
    assert calculate_ast(PRE/'run_preflight.py')==calculate_ast(OUT/'run_formal.py')
    contract=dict(date='2026-10-18',preflight_job=210291,
        preflight_source_sums_sha256=sha(PRE/'SHA256SUMS'),protocol_sha256=sha(PRE/'protocol.json'),
        initial_checkpoint_sha256='926030039ac9d32634bf193e989a44a5b891882c503cb8b40fd7b5091ac98ed6',
        user_reported_verified_resources=dict(peak_reserved_gib=3.77734375,
            warm_update_seconds=.16032358317170292,estimated_paired_formal_seconds=1956.0530471117818),
        scope='Two4096-update continuations; same1836 full16 windows, frozenCLIP, only readout learns1026 extra parameters',
        first8_loss_preflight_comparison_atol=1e-6,
        preflight_reexecuted=False,old1836_or40_or13_rescored=False,
        formal_training_calculate_AST_matches_preflight=True,
        base_metrics_source=dict(path=str(metric_source.relative_to(ROOT)),sha256=sha(metric_source)),
        prefix_validation_dependencies={name:sha(PREFIX/name) for name in
            ('protocol.json','input_reference/report.json','input_reference/validation_rows.json',
             'input_reference/validation_reference.npz','state_protocol.json')},
        baseline_result_files={name:sha(ROOT/'reports/bridge-prefix-formal-209481'/name) for name in
            ('report.json','validation_rows.json','validation_inputs.npz','unchanged207558_full16.npz')})
    write('formal_contract.json',json.dumps(contract,indent=2)+'\n')
    write('formal.sh','''#!/bin/bash
#SBATCH --job-name=bridge-spatial-formal
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --chdir=/projects/Zeroshot
#SBATCH --output=/projects/Zeroshot/logs/bridge-spatial-formal-%j.out
set -euo pipefail
source /projects/Zeroshot/scripts/ssd_cache.sh
export TMPDIR="/projects/Zeroshot/.tmp/${SLURM_JOB_ID}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 USE_TF=0 OMP_NUM_THREADS=4
mkdir -p "$TMPDIR" /projects/Zeroshot/runs
SPATIAL_USED_BYTES=$(du --apparent-size -s -B1 /projects/_ssd/Zeroshot | cut -f1)
if (( SPATIAL_USED_BYTES + 8*1024*1024*1024 + 15*1024*1024*1024 > 200*1024*1024*1024 )); then
  echo "Planning headroom insufficient; no live quota claim"
  exit 1
fi
cd /projects/Zeroshot/bridge_spatial_formal_v1
sha256sum -c SHA256SUMS
nvidia-smi
/projects/Zeroshot/envs/multitask-preflight-v1/bin/python -u run_formal.py \\
  --project /projects/Zeroshot \\
  --output "/projects/Zeroshot/runs/bridge-spatial-formal-${SLURM_JOB_ID}"
''')
    write('submit_formal_once.sh','''#!/bin/bash
set -euo pipefail
CODE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$CODE"
exec 9>submit_formal.lock
flock -x 9
if [[ -s submitted_formal.job ]]; then
  printf 'Existing spatial formal job: %s\\n' "$(cat submitted_formal.job)"
  exit 0
fi
sha256sum -c SHA256SUMS
/projects/Zeroshot/envs/multitask-preflight-v1/bin/python /projects/Zeroshot/bridge_spatial_readout_v1/verify_preflight.py /projects/Zeroshot/runs/bridge-spatial-preflight-210291
SPATIAL_REPLY=$(sbatch --parsable "$CODE/formal.sh")
SPATIAL_NUMBER="${SPATIAL_REPLY%%;*}"
[[ "$SPATIAL_NUMBER" =~ ^[0-9]+$ ]]
printf '%s\\n' "$SPATIAL_NUMBER" > submitted_formal.job
printf 'SPATIAL_FORMAL: %s\\n' "$SPATIAL_NUMBER"
''')
    write('README.md','''# Fixed spatial-readout formal pair after VERIFIED210291

This is an append-only code package, not a modification of preflight sources or207558. CLIP visual/text weights and gripper stay frozen. Both arms continue the same207558 for4096 updates, freshAdam1e-5, batch8, exact same1836 windows and schedule, original full16/DDPM100. Candidate learns1026 region scores; control fixes those scores at zero. Original base trainables match. No K1/request port or added/deleted windows.

The launcher first verifies the existing210291 report; it does NOT rerun preflight. Runtime binds immutable preflight SHA sums/protocol, old cache and checkpoint, training reference, qualified10-development exports, and original209481 baseline arrays. First8 losses must match the successful corresponding preflight arm within1e-6; actual calculate() AST is unchanged.

Both final weights are saved and roundtrip-checked BEFORE development inputs are encoded/scored. Only final weights, no model selection or intermediate/Adam checkpoints. Same already-used10-development episodes/50 full16 windows; old207558 predictions reused, no old1836/40/13 rescoring. Per-arm3 sampling seeds; first target comes from the ORIGINAL full16 trajectory, not a new K1 policy. ALL prior protocol admission gates apply. Failure closes this branch, no sweeps or automatic final25.

Dependencies already on cluster: bridge_spatial_readout_v1, VERIFIED runs/bridge-spatial-preflight-210291/report.json, bridge_prefix_pair_v1 qualified validation exports, original training assets/cache, runs/bridge-prefix-formal-209481 small baseline outputs. Missing/changed assets STOP, no downloads/fallbacks. New validation50 frozen encodings are checked against209481 SHA before comparing.

Submit ONCE after unpacking to /projects/Zeroshot/bridge_spatial_formal_v1:

    bash /projects/Zeroshot/bridge_spatial_formal_v1/submit_formal_once.sh

Inspect completed output with:

    /projects/Zeroshot/envs/multitask-preflight-v1/bin/python /projects/Zeroshot/bridge_spatial_formal_v1/verify_formal.py /projects/Zeroshot/runs/bridge-spatial-formal-JOBID

Verifier checks final-weight SHA and independently recomputes full16/first-target metrics and admission from NPZ. A successful job or verifier is NOT itself an improvement verdict. No closed-loop, gripper-learning, unseen-task or statistical-significance claim. Final25 payloads never opened.
''')
    for path in OUT.rglob('*.py'):compile(path.read_text(),str(path),'exec')
    files=sorted(p for p in OUT.rglob('*') if p.is_file())
    write('SHA256SUMS',''.join(f'{sha(p)}  {p.relative_to(OUT).as_posix()}\n' for p in files))
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(OUT.parent).as_posix())
    summary=dict(package=str(archive.relative_to(ROOT)),bytes=archive.stat().st_size,sha256=sha(archive),
        files=len(files)+1,preflight_verified_by_user=True,preflight_report_received_locally=False,
        formal_submitted=False,formal_model_executed_locally=False,CLIP_frozen=True)
    (ROOT/'reports/bridge_spatial_formal_release_20261018.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()

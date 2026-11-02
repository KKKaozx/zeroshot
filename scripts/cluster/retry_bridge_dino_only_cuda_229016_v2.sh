#!/bin/bash
# One manual retry on the sole10-GPU A6000 node. Slurm chooses the GPU.
set -euo pipefail
ROOT=/projects/Zeroshot
BUNDLE="$ROOT/bridge_dino_only_v1"
PY="$ROOT/envs/multitask-preflight-v1/bin/python"
cd "$BUNDLE"
exec 9>jobs/.submit.lock
flock -n 9 || { echo "Another submission is active"; exit 1; }
[[ "$(cat jobs/fit.id)" == 229016 ]] || { echo "Formal marker changed; no duplicate submitted"; exit 1; }
[[ "$(cat jobs/preflight.id)" == 228978 ]] || exit 2
ROW=$(sacct -X -n -P -j 229016 --format=JobIDRaw,State%32,NodeList)
STATE=$(awk -F'|' '$1=="229016" {print $2; exit}' <<<"$ROW")
NODE=$(awk -F'|' '$1=="229016" {print $3; exit}' <<<"$ROW")
case "$STATE" in FAILED|NODE_FAIL) ;; *) echo "229016 is '$STATE'; no retry submitted"; exit 2 ;; esac
[[ "$NODE" == gpu-a6000-1 ]] || { echo "Unexpected node '$NODE'; nothing submitted"; exit 2; }
[[ ! -e "$ROOT/runs/bridge-dino-only-fit-229016" ]] || {
  echo "Old output exists; inspect it first, nothing deleted"; exit 2;
}
"$PY" -B - <<'PY'
from pathlib import Path
text=Path('/projects/Zeroshot/logs/bridge-dino-only-229016.out').read_text()
assert 'ValueError: Allocated working CUDA GPU required' in text
assert not any(x in text for x in ['COVERAGE_UPDATE ', 'DINO_ONLY_INPUT ', 'DINO_ONLY_ENCODE '])
print('Old failure was before training; original trial budget remains unspent')
PY
sha256sum -c SHA256SUMS
"$PY" -B check_assets.py --root "$ROOT"
"$PY" -B verify.py "$ROOT/runs/bridge-dino-only-preflight-228978" --stage preflight

WRAP=$(mktemp "$BUNDLE/jobs/cuda-health-229016-XXXXXX.sh")
"$PY" -B - "$WRAP" <<'PY'
from pathlib import Path
import sys
source=Path('job.sh').read_text()
assert source.count('set -euo pipefail\n')==1
header=source.split('set -euo pipefail\n',1)[0]
# Preserve ALL original SBATCH resource/time/log settings. This process exits
# before the original entry when CUDA fails; no GPU override or retry loop.
health=r'''set -euo pipefail
ROOT=/projects/Zeroshot
source "$ROOT/scripts/ssd_cache.sh"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
PY="$ROOT/envs/multitask-preflight-v1/bin/python"
echo "CUDA_HEALTH_BEFORE_TRAIN job=${SLURM_JOB_ID} node=${SLURMD_NODENAME:-unknown} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
nvidia-smi --query-gpu=uuid,pci.bus_id,name --format=csv,noheader
timeout --kill-after=5s 60s "$PY" -B - <<'HEALTH'
import json,os,torch
torch.cuda.init()
x=torch.ones(1,device='cuda')
torch.cuda.synchronize()
assert float(x.cpu()[0])==1.
print('CUDA_HEALTH_PASSED '+json.dumps(dict(gpu=torch.cuda.get_device_name(),
    torch_version=str(torch.__version__),cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
    optimizer_updates=0,model_loaded=False)))
HEALTH
exec bash "$ROOT/bridge_dino_only_v1/job.sh"
'''
Path(sys.argv[1]).write_text(header+health)
PY
if [[ -e jobs/fit.failed-229016.id ]]; then
  [[ "$(cat jobs/fit.failed-229016.id)" == 229016 ]] || exit 2
else
  cp -p jobs/fit.id jobs/fit.failed-229016.id
fi
JOB=$(sbatch --parsable --nodelist=gpu-a6000-1 \
  --export=ALL,BRIDGE_DINO_STAGE=fit,BRIDGE_DINO_PREFLIGHT=228978 "$WRAP")
JOB="${JOB%%;*}"
[[ "$JOB" =~ ^[0-9]+$ ]] || exit 2
printf '%s\n' "$JOB" >jobs/fit.id
printf 'failed_job=229016\npreflight_job=228978\nnode=gpu-a6000-1\nretry_job=%s\nwrapper=%s\n' \
  "$JOB" "$WRAP" >"jobs/fit.retry-${JOB}.txt"
echo "BRIDGE DINO-ONLY same-model CUDA-gated retry: $JOB"

#!/bin/bash
set -euo pipefail
ROOT=/projects/Zeroshot
cd "$ROOT/bridge_dino_epsilon_control_v1"
mkdir -p jobs "$ROOT/logs"
exec 9>jobs/.submit.lock
flock -n 9 || exit 2
[[ ! -e jobs/candidate.id ]] || { echo "Already submitted: $(cat jobs/candidate.id)"; exit 2; }
sha256sum -c SHA256SUMS
PY="$ROOT/envs/multitask-preflight-v1/bin/python"
"$PY" -B checks.py --root "$ROOT"
"$PY" -B "$ROOT/bridge_backbone_pair_a40_v1/check_assets.py" --root "$ROOT"
JOB=$(sbatch --parsable job.sh)
JOB="${JOB%%;*}"
[[ "$JOB" =~ ^[0-9]+$ ]] || exit 2
printf '%s\n' "$JOB" >jobs/candidate.id
echo "BRIDGE DINO EPSILON CONDITIONAL CONTROL: $JOB"

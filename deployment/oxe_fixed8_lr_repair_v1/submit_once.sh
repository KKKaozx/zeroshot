#!/usr/bin/env bash
set -euo pipefail
ROOT=/projects/Zeroshot
PROJECT="$ROOT/envs/multitask-preflight-v1"
export LD_LIBRARY_PATH="$PROJECT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "$ROOT/oxe_fixed8_lr_repair_v1"
exec 9>.submit.lock
flock -n 9 || { echo 'Another repair submission is active'; exit 1; }
[[ ! -e job_id.txt ]] || { echo "Already submitted: $(cat job_id.txt)"; exit 1; }
[[ "$(cat "$ROOT/oxe_fixed8_learnability_v1/job_id.txt")" == 233339 ]] || { echo 'Original job identity differs'; exit 1; }
STATE=$(sacct -X -nP -j 233339 --format=State | head -n1 | cut -d'|' -f1)
case "$STATE" in
  FAILED|TIMEOUT|CANCELLED*|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED) ;;
  *) echo "Original job233339 state=$STATE; do not duplicate"; exit 1 ;;
esac
sha256sum -c SHA256SUMS
PYTHONNOUSERSITE=1 "$PROJECT/bin/python" -B run_repair.py --root "$ROOT" --check-only
mkdir -p "$ROOT/logs"
JOB=$(sbatch --parsable job.sh)
JOB="${JOB%%;*}"
[[ "$JOB" =~ ^[0-9]+$ ]] || exit 1
printf '%s\n' "$JOB" > job_id.txt
echo "OXE FIXED8 LR REPAIRED: $JOB"

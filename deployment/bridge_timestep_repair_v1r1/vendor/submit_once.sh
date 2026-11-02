#!/usr/bin/env bash
set -euo pipefail
ROOT=/projects/Zeroshot
PROJECT="$ROOT/envs/multitask-preflight-v1"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="$PROJECT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "$ROOT/bridge_full_capability_v1"
exec 9>.submit.lock
flock -n 9 || { echo 'Submission already in progress'; exit 1; }
[[ ! -e job_id.txt ]] || { echo "Submission already recorded: $(cat job_id.txt)"; exit 1; }
sha256sum -c SHA256SUMS
"$PROJECT/bin/python" -B run_bridge.py --root "$ROOT" --check-only
mkdir -p "$ROOT/logs"
printf '%s\n' 'SUBMITTING: do not delete or retry without checking squeue/sacct' > job_id.txt
JOB=$(sbatch --parsable job.sh)
JOB="${JOB%%;*}"
[[ "$JOB" =~ ^[0-9]+$ ]] || { echo 'Ambiguous sbatch response; preserve job_id.txt and check scheduler'; exit 1; }
printf '%s\n' "$JOB" > job_id.txt
echo "BRIDGE FULL CAPABILITY: $JOB"

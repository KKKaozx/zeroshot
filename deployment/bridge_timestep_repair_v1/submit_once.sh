#!/usr/bin/env bash
# Run by the user only after reviewing/authorizing this concrete budget.
set -euo pipefail
ROOT=/projects/Zeroshot
PROJECT="$ROOT/envs/multitask-preflight-v1"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="$PROJECT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "$ROOT/bridge_timestep_repair_v1"
exec 9>.submit.lock
flock -n 9 || { echo 'Submission already in progress'; exit 1; }
[[ ! -e job_id.txt ]] || { echo "Submission already recorded: $(cat job_id.txt)"; exit 1; }
sha256sum -c SHA256SUMS
"$PROJECT/bin/python" -B run_pair.py --root "$ROOT" --check-only
mkdir -p "$ROOT/logs"
printf '%s\n' 'SUBMITTING: preserve this marker and check scheduler before any retry' > job_id.txt
JOB=$(sbatch --parsable job.sh)
JOB="${JOB%%;*}"
[[ "$JOB" =~ ^[0-9]+$ ]] || { echo 'Ambiguous submission response; preserve marker'; exit 1; }
printf '%s\n' "$JOB" > job_id.txt
echo "BRIDGE TIMESTEP REPAIR: $JOB"

#!/usr/bin/env bash
set -euo pipefail
ROOT=/projects/Zeroshot
PROJECT="$ROOT/envs/multitask-preflight-v1"
export LD_LIBRARY_PATH="$PROJECT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
cd "$ROOT/oxe_sampling_pair_v1"
exec 9>.submit.lock
flock -n 9 || { echo 'Another pair submission is active'; exit 1; }
[[ ! -e job_id.txt ]] || { echo "Already submitted: $(cat job_id.txt)"; exit 1; }
sha256sum -c SHA256SUMS
"$PROJECT/bin/python" -B run_pair.py --root "$ROOT" --check-only
mkdir -p "$ROOT/logs"
JOB=$(sbatch --parsable job.sh)
JOB="${JOB%%;*}"
[[ "$JOB" =~ ^[0-9]+$ ]] || exit 1
printf '%s\n' "$JOB" > job_id.txt
echo "OXE SAMPLING SHORT PAIR: $JOB"

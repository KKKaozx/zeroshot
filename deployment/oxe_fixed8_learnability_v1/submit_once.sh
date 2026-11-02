#!/usr/bin/env bash
set -euo pipefail
ROOT=/projects/Zeroshot
cd "$ROOT/oxe_fixed8_learnability_v1"
exec 9>.submit.lock
flock -n 9 || { echo 'Another submission is active'; exit 1; }
[[ ! -e job_id.txt ]] || { echo "Already submitted: $(cat job_id.txt)"; exit 1; }
sha256sum -c SHA256SUMS
PYTHONNOUSERSITE=1 "$ROOT/envs/multitask-preflight-v1/bin/python" -B run_check.py --root "$ROOT" --check-only
mkdir -p "$ROOT/logs"
JOB=$(sbatch --parsable job.sh)
JOB="${JOB%%;*}"
[[ "$JOB" =~ ^[0-9]+$ ]] || exit 1
printf '%s\n' "$JOB" > job_id.txt
echo "OXE FIXED8 LEARNABILITY: $JOB"

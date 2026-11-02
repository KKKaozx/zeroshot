#!/usr/bin/env bash
set -euo pipefail
ROOT=/projects/Zeroshot
cd "$ROOT/oxe_target_pair_short_v1"
exec 9>.submit.lock
flock -n 9 || { echo 'Another submission is active'; exit 1; }
if [[ -e job_id.txt ]]; then
    echo "Submission already recorded: $(cat job_id.txt). Inspect it; no automatic retry."
    exit 1
fi
sha256sum -c SHA256SUMS
PYTHONNOUSERSITE=1 "$ROOT/envs/multitask-preflight-v1/bin/python" -B checks.py --root "$ROOT"
mkdir -p "$ROOT/logs"
JOB=$(sbatch --parsable job.sh)
JOB="${JOB%%;*}"
[[ "$JOB" =~ ^[0-9]+$ ]] || { echo "Unexpected submission output: $JOB"; exit 1; }
printf '%s\n' "$JOB" > job_id.txt
echo "OXE MAIN TARGET SHORT PAIR: $JOB"

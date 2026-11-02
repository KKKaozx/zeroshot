#!/bin/bash
set -euo pipefail
ROOT=/projects/Zeroshot
BUNDLE="$ROOT/oxe_dino_main_preflight_v1"
PROJECT="$ROOT/envs/multitask-preflight-v1"
cd "$BUNDLE"
exec 9>"$BUNDLE/.submit.lock"
flock -n 9 || { echo "Another submission is active"; exit 1; }
if [[ -e preflight_job_id.txt ]]; then echo "Already submitted: $(cat preflight_job_id.txt)"; exit 1; fi
sha256sum -c SHA256SUMS
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 LD_LIBRARY_PATH="$PROJECT/lib" "$PROJECT/bin/python" -B run_preflight.py --root "$ROOT" --check-only
mkdir -p "$ROOT/logs"
JOB=$(sbatch --parsable "$BUNDLE/preflight.sh")
[[ "$JOB" =~ ^[0-9]+$ ]] || { echo "Unexpected sbatch output: $JOB"; exit 1; }
printf '%s\n' "$JOB" > preflight_job_id.txt
echo "OXE DINO MAIN PREFLIGHT: $JOB"
squeue --start -j "$JOB"

#!/bin/bash
set -euo pipefail
cd /projects/Zeroshot/oxe_frozen_identity_diag_v1
exec 9>.submit.lock
flock -n 9 || exit 1
if [[ -e job_id.txt ]]; then echo "Already submitted: $(cat job_id.txt)"; exit 1; fi
sha256sum -c SHA256SUMS
mkdir -p /projects/Zeroshot/logs
JOB=$(sbatch --parsable diagnosis.sh)
[[ "$JOB" =~ ^[0-9]+$ ]] || exit 1
printf '%s\n' "$JOB" > job_id.txt
echo "CPU FROZEN IDENTITY DIAGNOSIS: $JOB"
squeue --start -j "$JOB"

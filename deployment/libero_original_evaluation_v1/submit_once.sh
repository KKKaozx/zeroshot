#!/usr/bin/env bash
set -euo pipefail
cd /projects/Zeroshot/libero_original_evaluation_v1
N=${1:-1}
if ! [[ "$N" =~ ^[1-9][0-9]*$ ]]; then echo 'Positive episode count required' >&2; exit 1; fi
sha256sum -c SHA256SUMS
if [ -f job_id.txt ]; then
  OLD=$(cat job_id.txt)
  STATE=$(squeue -h -j "$OLD" -o '%T' 2>/dev/null || true)
  if [ -n "$STATE" ]; then echo "Existing evaluation job $OLD is $STATE; no duplicate submission"; exit 0; fi
fi
test -x /projects/Zeroshot/envs/libero-eval-v1/bin/python
test -d /projects/Zeroshot/libero_runtime_v1/libero/libero/assets
test -f /projects/Zeroshot/runs/oxe-dino-noaug-s42/final_policy.pt
JOB=$(sbatch --parsable evaluate.sh "$N")
JOB=${JOB%%;*}
printf '%s\n' "$JOB" >job_id.txt
echo "LIBERO original-instruction segment: $JOB (up to $N NEW completed episodes; total remains4000)"

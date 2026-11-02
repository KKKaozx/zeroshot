#!/bin/bash
set -euo pipefail
cd /projects/Zeroshot/pusht_zero_block_preflight_v1
sha256sum -c SHA256SUMS
if ! mkdir .submission; then echo 'Already submitted; inspect existing job before retrying.' >&2; exit 1; fi
if JOB=$(sbatch --parsable preflight.sh); then
  printf '%s
' "$JOB" > .submission/job_id
  echo "Push-T无注意力块控制预检：$JOB"
else
  echo 'Submission failed; guard retained for inspection.' >&2
  exit 1
fi

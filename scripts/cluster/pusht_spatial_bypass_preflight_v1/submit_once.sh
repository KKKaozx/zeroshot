#!/bin/bash
set -euo pipefail
cd /projects/Zeroshot/pusht_spatial_bypass_preflight_v1
sha256sum -c SHA256SUMS
if ! mkdir .submission; then echo 'Already submitted; inspect existing job before another attempt.' >&2; exit 1; fi
if JOB=$(sbatch --parsable preflight.sh); then
  printf '%s
' "$JOB" > .submission/job_id
  echo "Push-T空间旁路预检：$JOB"
else
  echo 'Submission failed; guard retained for inspection.' >&2
  exit 1
fi

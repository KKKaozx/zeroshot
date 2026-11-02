#!/bin/bash
set -euo pipefail
cd /projects/Zeroshot/pusht_spatial_diagnostic_v1
sha256sum -c SHA256SUMS
if ! mkdir .submission; then echo 'Already submitted; inspect existing job before retrying.' >&2; exit 1; fi
if JOB=$(sbatch --parsable diagnostic.sh); then
  printf '%s
' "$JOB" > .submission/job_id
  echo "Push-T冻结空间诊断：$JOB"
else
  echo 'Submission failed; retained submission guard for inspection.' >&2
  exit 1
fi

#!/bin/bash
set -euo pipefail
cd /projects/Zeroshot/pusht_endpoint_action_audit_v1
sha256sum -c SHA256SUMS
if ! mkdir .submission; then echo 'Already submitted; inspect existing job before retrying.' >&2; exit 1; fi
if JOB=$(sbatch --parsable audit.sh); then
  printf '%s
' "$JOB" > .submission/job_id
  echo "Push-T冻结端点动作诊断：$JOB"
else
  echo 'Submission failed; guard retained for inspection.' >&2
  exit 1
fi

#!/bin/bash
set -euo pipefail
cd /projects/Zeroshot/pusht_project_training_v1
sha256sum -c SHA256SUMS
if [[ -n "${PUSHT_PROJECT_RESUME:-}" ]]; then echo 'Use a separately reviewed resume, not the fresh submit wrapper.' >&2; exit 1; fi
LOCK=.part1_submission
if ! mkdir "$LOCK"; then echo 'Submission already attempted. Inspect existing job; no automatic repeat.' >&2; exit 1; fi
if JOB=$(sbatch --parsable part1.sh); then
  printf '%s
' "$JOB" > "$LOCK/job_id"
  echo "Push-T项目组第一段：$JOB"
else
  echo 'Submission failed; retained guard. Inspect the scheduler error before manual retry.' >&2
  exit 1
fi

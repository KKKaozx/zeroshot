#!/bin/bash
set -euo pipefail
ROOT=/projects/Zeroshot
BUNDLE="$ROOT/pusht_dino_training_v1"
OFFICIAL="$ROOT/envs/official-pusht-v1"
cd "$BUNDLE"
sha256sum -c SHA256SUMS
if [[ -n "${PUSHT_DINO_RESUME:-}" ]]; then echo 'Fresh submit only; inspect interrupted runs before an explicit resume.' >&2; exit 1; fi
LD_LIBRARY_PATH="$OFFICIAL/lib" "$OFFICIAL/bin/python" -B formal_binding.py --root "$ROOT"
if ! mkdir .part1_submission; then echo 'Already submitted; inspect saved job before any retry.' >&2; exit 1; fi
if JOB=$(sbatch --parsable part1.sh); then
  printf '%s\n' "$JOB" > .part1_submission/job_id
  echo "Push-T DINO项目组第一段：$JOB"
else
  echo 'Submission failed; guard retained for inspection.' >&2
  exit 1
fi

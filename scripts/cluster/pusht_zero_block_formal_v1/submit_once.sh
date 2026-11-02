#!/bin/bash
set -euo pipefail
ROOT=/projects/Zeroshot
BUNDLE="$ROOT/pusht_zero_block_formal_v1"
OFFICIAL="$ROOT/envs/official-pusht-v1"
cd "$BUNDLE"
sha256sum -c SHA256SUMS
if ! mkdir .submission; then echo 'Already submitted; inspect saved job/checkpoint before retrying.' >&2; exit 1; fi
LD_LIBRARY_PATH="$OFFICIAL/lib" "$OFFICIAL/bin/python" -B bind_preflight.py --root "$ROOT" --binding "$BUNDLE/.submission/preflight_binding.json"
if JOB=$(sbatch --parsable formal.sh); then
  printf '%s
' "$JOB" > .submission/job_id
  echo "Push-T无注意力块正式对照：$JOB"
else
  echo 'Submission failed; binding and guard retained for inspection.' >&2
  exit 1
fi

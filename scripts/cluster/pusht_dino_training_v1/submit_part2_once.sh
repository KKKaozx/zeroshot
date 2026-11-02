#!/bin/bash
set -euo pipefail
ROOT=/projects/Zeroshot
BUNDLE="$ROOT/pusht_dino_training_v1"
OFFICIAL="$ROOT/envs/official-pusht-v1"
: "${1:?Pass the verified DINO segment1 latest.pt}"
PUSHT_DINO_RESUME=$(realpath -e -- "$1")
export PUSHT_DINO_RESUME
case "$PUSHT_DINO_RESUME" in
  /projects/_ssd/Zeroshot/runs/pusht-dino-part1-*/latest.pt) ;;
  *) echo 'Expected exact DINO part1 checkpoint under canonical project/runs.' >&2; exit 1 ;;
esac
cd "$BUNDLE"
sha256sum -c SHA256SUMS
LD_LIBRARY_PATH="$OFFICIAL/lib" "$OFFICIAL/bin/python" -B verify_dino.py "$(dirname "$PUSHT_DINO_RESUME")"
LD_LIBRARY_PATH="$OFFICIAL/lib" "$OFFICIAL/bin/python" -B formal_binding.py --root "$ROOT"
if ! mkdir .part2_submission; then echo 'Already submitted; inspect saved job before retrying.' >&2; exit 1; fi
if JOB=$(sbatch --parsable --export=ALL part2.sh); then
  printf '%s\n' "$JOB" > .part2_submission/job_id
  echo "Push-T DINO项目组第二段：$JOB"
else
  echo 'Submission failed; guard retained for inspection.' >&2
  exit 1
fi

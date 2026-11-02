#!/bin/bash
set -euo pipefail
ROOT=/projects/Zeroshot
BUNDLE="$ROOT/bridge_dino_only_v1"
PY="$ROOT/envs/multitask-preflight-v1/bin/python"
cd "$BUNDLE"
STAGE="${1:-preflight}"
[[ "$STAGE" == preflight || "$STAGE" == fit ]] || { echo "Use: bash submit_once.sh preflight | fit PREFLIGHT_ID"; exit 2; }
mkdir -p jobs "$ROOT/logs"
exec 9>jobs/.submit.lock
flock -n 9 || { echo "Another submission is active"; exit 1; }
[[ ! -e "jobs/${STAGE}.id" ]] || { echo "Already submitted: $(cat "jobs/${STAGE}.id"); no duplicate"; exit 1; }
sha256sum -c SHA256SUMS
"$PY" -B check_assets.py --root "$ROOT"
if [[ "$STAGE" == fit ]]; then
  PREFLIGHT="${2:?New successful preflight ID required}"
  [[ "$PREFLIGHT" =~ ^[0-9]+$ ]] || exit 2
  [[ -f jobs/preflight.id && "$(cat jobs/preflight.id)" == "$PREFLIGHT" ]] || { echo "Preflight marker differs"; exit 2; }
  "$PY" -B verify.py "$ROOT/runs/bridge-dino-only-preflight-${PREFLIGHT}" --stage preflight
  JOB=$(sbatch --parsable --export="ALL,BRIDGE_DINO_STAGE=fit,BRIDGE_DINO_PREFLIGHT=${PREFLIGHT}" job.sh)
else
  JOB=$(sbatch --parsable --time=00:30:00 --export=ALL,BRIDGE_DINO_STAGE=preflight job.sh)
fi
JOB="${JOB%%;*}"
[[ "$JOB" =~ ^[0-9]+$ ]] || exit 2
printf '%s\n' "$JOB" >"jobs/${STAGE}.id"
echo "BRIDGE DINO-ONLY ${STAGE}: $JOB"

#!/bin/bash
set -euo pipefail
ROOT=/projects/Zeroshot
PROJECT="$ROOT/envs/multitask-preflight-v1"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 LD_LIBRARY_PATH="$PROJECT/lib"
PY="$PROJECT/bin/python"
cd "$ROOT/oxe_bridge_supervision_frozen_repair_v1"
mkdir -p jobs "$ROOT/logs"
exec 9>jobs/.submit.lock
flock -n 9 || { echo "Another submission is active"; exit 2; }
sha256sum -c SHA256SUMS
STAGE="${1:?Use 1, 2 or evaluate}"
state_of() { sacct -X -n -P -j "$1" --format=JobIDRaw,State%32 | awk -F'|' -v id="$1" '$1==id {print $2;exit}'; }
[[ "$(state_of 229597)" == FAILED ]] || { echo "Original229597 must be confirmed FAILED; no retry submitted"; exit 2; }
[[ "$(state_of 229577)" == COMPLETED ]] || { echo "Passed preflight229577 must be COMPLETED"; exit 2; }
[[ "$(cat "$ROOT/oxe_bridge_supervision_v1/jobs/stage1.id")" == 229597 ]] || exit 2
"$PY" -B compare_existing.py --root "$ROOT"
"$PY" -B "$ROOT/oxe_bridge_supervision_v1/verify.py" "$ROOT/runs/oxe-bridge-supervision-preflight-229577" --mode preflight
if [[ "$STAGE" == 1 || "$STAGE" == 2 ]]; then
  MARKER="jobs/stage${STAGE}.id"
  [[ ! -e "$MARKER" ]] || { echo "Already submitted: $(cat "$MARKER")"; exit 2; }
  if [[ "$STAGE" == 2 ]]; then
    PREVIOUS=$(cat jobs/stage1.id)
    [[ "$(state_of "$PREVIOUS")" == COMPLETED ]] || exit 2
    "$PY" -B "$ROOT/oxe_bridge_supervision_v1/verify.py" "$ROOT/runs/oxe-bridge-supervision-s42" --mode fit --stage 1
  fi
  "$PY" -B run_repaired.py --root "$ROOT" --stage "$STAGE" --check-only
  JOB=$(sbatch --parsable --export="ALL,REPAIR_MODE=fit,REPAIR_STAGE=${STAGE}" job.sh)
elif [[ "$STAGE" == evaluate ]]; then
  MARKER=jobs/evaluate.id
  [[ ! -e "$MARKER" ]] || exit 2
  PREVIOUS=$(cat jobs/stage2.id)
  [[ "$(state_of "$PREVIOUS")" == COMPLETED ]] || exit 2
  "$PY" -B "$ROOT/oxe_bridge_supervision_v1/verify.py" "$ROOT/runs/oxe-bridge-supervision-s42" --mode fit --stage 2
  JOB=$(sbatch --parsable --time=00:30:00 --export=ALL,REPAIR_MODE=evaluate job.sh)
else
  exit 2
fi
JOB="${JOB%%;*}"
[[ "$JOB" =~ ^[0-9]+$ ]] || exit 2
printf '%s\n' "$JOB" >"$MARKER"
echo "BRIDGE SUPERVISION REPAIRED stage=${STAGE}: $JOB"

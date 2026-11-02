#!/bin/bash
set -euo pipefail
ROOT=/projects/Zeroshot
BUNDLE="$ROOT/oxe_bridge_supervision_v1"
PROJECT="$ROOT/envs/multitask-preflight-v1"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 LD_LIBRARY_PATH="$PROJECT/lib"
PY="$PROJECT/bin/python"
cd "$BUNDLE"
mkdir -p jobs "$ROOT/logs"
exec 9>jobs/.submit.lock
flock -n 9 || { echo "Another submission is active"; exit 1; }
MODE="${1:-preflight}"
sha256sum -c SHA256SUMS
if [[ "$MODE" == preflight ]]; then
  MARKER=jobs/preflight.id
  [[ ! -e "$MARKER" ]] || { echo "Preflight already submitted: $(cat "$MARKER")"; exit 2; }
  "$PY" -B run_experiment.py --root "$ROOT" --mode preflight --check-only
  JOB=$(sbatch --parsable --time=00:20:00 --export=ALL,SUPERVISION_MODE=preflight job.sh)
elif [[ "$MODE" == fit ]]; then
  STAGE="${2:?Use fit STAGE PREFLIGHT_JOB_ID}"
  PREFLIGHT="${3:?Need successful new preflight ID}"
  [[ "$STAGE" =~ ^[12]$ && "$PREFLIGHT" =~ ^[0-9]+$ ]] || exit 2
  [[ "$(cat jobs/preflight.id)" == "$PREFLIGHT" ]] || exit 2
  STATE=$(sacct -X -n -P -j "$PREFLIGHT" --format=JobIDRaw,State%32 | awk -F'|' -v id="$PREFLIGHT" '$1==id {print $2;exit}')
  [[ "$STATE" == COMPLETED ]] || { echo "Preflight is $STATE; no formal training submitted"; exit 2; }
  MARKER="jobs/stage${STAGE}.id"
  [[ ! -e "$MARKER" ]] || { echo "Stage already submitted: $(cat "$MARKER")"; exit 2; }
  "$PY" -B verify.py "$ROOT/runs/oxe-bridge-supervision-preflight-${PREFLIGHT}" --mode preflight
  if [[ "$STAGE" == 2 ]]; then
    PREVIOUS=$(cat jobs/stage1.id)
    STATE=$(sacct -X -n -P -j "$PREVIOUS" --format=JobIDRaw,State%32 | awk -F'|' -v id="$PREVIOUS" '$1==id {print $2;exit}')
    [[ "$STATE" == COMPLETED ]] || { echo "Stage1 is $STATE; no stage2 submitted"; exit 2; }
    "$PY" -B verify.py "$ROOT/runs/oxe-bridge-supervision-s42" --mode fit --stage 1
  fi
  "$PY" -B run_experiment.py --root "$ROOT" --mode fit --stage "$STAGE"     --preflight "$ROOT/runs/oxe-bridge-supervision-preflight-${PREFLIGHT}" --check-only
  JOB=$(sbatch --parsable --export="ALL,SUPERVISION_MODE=fit,SUPERVISION_STAGE=${STAGE},SUPERVISION_PREFLIGHT=${PREFLIGHT}" job.sh)
elif [[ "$MODE" == evaluate ]]; then
  MARKER=jobs/evaluate.id
  [[ ! -e "$MARKER" ]] || { echo "Evaluation already submitted: $(cat "$MARKER")"; exit 2; }
  "$PY" -B verify.py "$ROOT/runs/oxe-bridge-supervision-s42" --mode fit --stage 2
  JOB=$(sbatch --parsable --time=00:30:00 --export=ALL,SUPERVISION_MODE=evaluate job.sh)
else
  echo "Use preflight | fit STAGE PREFLIGHT_ID | evaluate"; exit 2
fi
JOB="${JOB%%;*}"
[[ "$JOB" =~ ^[0-9]+$ ]] || exit 2
printf '%s\n' "$JOB" >"$MARKER"
echo "OXE BRIDGE SUPERVISION ${MODE}: $JOB"

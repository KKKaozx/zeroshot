#!/bin/bash
set -euo pipefail
ROOT=/projects/Zeroshot
BUNDLE="$ROOT/oxe_dino_noaug_training_v1"
PROJECT="$ROOT/envs/multitask-preflight-v1"
SEED="${1:?Usage: bash submit_stage_once.sh SEED STAGE [retry]}"
STAGE="${2:?Usage: bash submit_stage_once.sh SEED STAGE [retry]}"
RETRY="${3:-}"
[[ "$SEED" =~ ^(42|43|44)$ && "$STAGE" =~ ^(1|2)$ ]] || { echo "Invalid seed/stage"; exit 1; }
[[ -z "$RETRY" || "$RETRY" == retry ]] || { echo "Only literal retry is allowed"; exit 1; }
cd "$BUNDLE"
mkdir -p jobs "$ROOT/logs"
exec 9>"$BUNDLE/.submit.lock"
flock -n 9 || { echo "Another submission is active"; exit 1; }
MARKER="$BUNDLE/jobs/seed${SEED}-stage${STAGE}.id"
if [[ -e "$MARKER" ]]; then
  OLD=$(cat "$MARKER")
  [[ "$OLD" =~ ^[0-9]+$ ]] || { echo "Invalid old marker"; exit 1; }
  [[ "$RETRY" == retry ]] || { echo "Already submitted: $OLD; do not duplicate"; exit 1; }
  STATE=$(sacct -X -nP -j "$OLD" --format=State | head -n1 | cut -d'|' -f1)
  case "$STATE" in FAILED|TIMEOUT|CANCELLED*|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED) ;; *) echo "Old job state is $STATE; no retry submitted"; exit 1 ;; esac
fi
sha256sum -c SHA256SUMS
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 LD_LIBRARY_PATH="$PROJECT/lib" "$PROJECT/bin/python" -B bind_preflight.py --root "$ROOT"
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 LD_LIBRARY_PATH="$PROJECT/lib" "$PROJECT/bin/python" -B run_stage.py --root "$ROOT" --seed "$SEED" --stage "$STAGE" --check-only
JOB=$(sbatch --parsable --export="ALL,OXE_SEED=$SEED,OXE_STAGE=$STAGE" --output="$ROOT/logs/oxe-dino-noaug-s${SEED}-p${STAGE}-%j.out" "$BUNDLE/train_stage.sh")
[[ "$JOB" =~ ^[0-9]+$ ]] || { echo "Unexpected sbatch response: $JOB"; exit 1; }
if [[ -e "$MARKER" ]]; then cp "$MARKER" "${MARKER}.previous"; fi
printf '%s\n' "$JOB" > "$MARKER"
echo "OXE DINO NO-AUG seed=$SEED stage=$STAGE JOB=$JOB"
squeue --start -j "$JOB"

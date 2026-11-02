#!/usr/bin/env bash
set -euo pipefail
ROOT=/projects/Zeroshot
REPAIR="$ROOT/oxe_resume_and_libero_dependency_repair_v1"
BUNDLE="$ROOT/oxe_dino_noaug_training_v1"
PROJECT="$ROOT/envs/multitask-preflight-v1"
cd "$REPAIR"
sha256sum -c SHA256SUMS
exec 9>"$BUNDLE/.submit.lock"
flock -n 9 || { echo 'Another training submission is active'; exit 1; }
MARKER="$BUNDLE/jobs/seed43-stage2.id"
OLD=$(cat "$MARKER")
[[ "$OLD" =~ ^[0-9]+$ ]] || { echo 'Invalid stage2 job marker'; exit 1; }
STATE=$(sacct -X -nP -j "$OLD" --format=State | head -n1 | cut -d'|' -f1)
case "$STATE" in FAILED|TIMEOUT|CANCELLED*|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED) ;; *) echo "Old seed43 stage2 job $OLD is $STATE; no duplicate submitted"; exit 1 ;; esac
cd "$BUNDLE"
sha256sum -c SHA256SUMS
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 LD_LIBRARY_PATH="$PROJECT/lib" "$PROJECT/bin/python" -B run_stage.py --root "$ROOT" --seed 43 --stage 2 --check-only
JOB=$(sbatch --parsable --output="$ROOT/logs/oxe-dino-noaug-s43-p2-%j.out" "$REPAIR/train_seed43_resume.sh")
[[ "$JOB" =~ ^[0-9]+$ ]] || { echo 'Unexpected sbatch response'; exit 1; }
cp "$MARKER" "$REPAIR/stage2_previous_job_${OLD}.txt"
printf '%s\n' "$JOB" >"$MARKER"
printf '%s\n' "$JOB" >"$REPAIR/resume_job_id.txt"
echo "OXE seed43 exact-constant stage2 resume: $JOB; original contract/optimizer/RNG/65536 budget retained"

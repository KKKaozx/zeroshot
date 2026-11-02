#!/usr/bin/env bash
#SBATCH --job-name=bridge-command8-fresh-scaled
#SBATCH --partition=cluster02
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=01:30:00
#SBATCH --no-requeue
#SBATCH --signal=B:USR1@180
#SBATCH --chdir=/projects/Zeroshot
#SBATCH --output=/projects/Zeroshot/logs/bridge-command8-fresh-scaled-%j.out
set -euo pipefail
ROOT=/projects/Zeroshot
PROJECT="$ROOT/envs/multitask-preflight-v1"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 USE_TF=0 WANDB_MODE=disabled
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export LD_LIBRARY_PATH="$PROJECT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "$ROOT/bridge_command8_fresh_scaled_v1"
sha256sum -c SHA256SUMS
CHILD=
stop_child() { if [[ -n "$CHILD" ]]; then kill -USR1 "$CHILD" || true; fi; }
trap stop_child USR1 TERM
"$PROJECT/bin/python" -B train_command8.py --root "$ROOT" --output "$ROOT/runs/bridge-command8-fresh-scaled-${SLURM_JOB_ID}" &
CHILD=$!
set +e
wait "$CHILD"
STATUS=$?
if kill -0 "$CHILD" 2>/dev/null; then wait "$CHILD"; STATUS=$?; fi
exit "$STATUS"

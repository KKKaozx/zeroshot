#!/bin/bash
#SBATCH --job-name=oxe-dino-noaug
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=20:00:00
#SBATCH --no-requeue
#SBATCH --chdir=/projects/Zeroshot
set -euo pipefail
ROOT=/projects/Zeroshot
BUNDLE="$ROOT/oxe_dino_noaug_training_v1"
PROJECT="$ROOT/envs/multitask-preflight-v1"
: "${OXE_SEED:?Need explicit seed}" "${OXE_STAGE:?Need explicit stage}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OXE_FROZEN_REFERENCE_JSON="$ROOT/runs/oxe-id-cpu-226567/preflight.json"
cd "$BUNDLE"
sha256sum -c SHA256SUMS
nvidia-smi
LD_LIBRARY_PATH="$PROJECT/lib" "$PROJECT/bin/python" -B run_stage.py --root "$ROOT" --seed "$OXE_SEED" --stage "$OXE_STAGE"

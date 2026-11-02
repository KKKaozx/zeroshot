#!/bin/bash
#SBATCH --job-name=oxe-dino-pref
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --chdir=/projects/Zeroshot
#SBATCH --output=/projects/Zeroshot/logs/oxe-dino-pref-%j.out
set -euo pipefail
ROOT=/projects/Zeroshot
BUNDLE="$ROOT/oxe_dino_main_preflight_v1"
PROJECT="$ROOT/envs/multitask-preflight-v1"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "$BUNDLE"
sha256sum -c SHA256SUMS
nvidia-smi
LD_LIBRARY_PATH="$PROJECT/lib" "$PROJECT/bin/python" -B run_preflight.py --root "$ROOT" --output "$ROOT/runs/oxe-dino-pref-${SLURM_JOB_ID}"

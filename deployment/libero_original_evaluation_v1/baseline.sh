#!/usr/bin/env bash
#SBATCH --job-name=libero-static
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --output=/projects/Zeroshot/logs/libero-static-%j.out
set -euo pipefail
cd /projects/Zeroshot/libero_original_evaluation_v1
sha256sum -c SHA256SUMS
export OMP_NUM_THREADS=2 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
/projects/Zeroshot/envs/libero-eval-v1/bin/python run_evaluation.py \
  --baseline --libero-root /projects/Zeroshot/libero_runtime_v1 \
  --output /projects/Zeroshot/runs/libero-original-zero-action \
  --max-new-episodes "${1:-1}" --wall-seconds 39600

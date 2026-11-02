#!/usr/bin/env bash
#SBATCH --job-name=libero-noaug-s42
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --output=/projects/Zeroshot/logs/libero-s42-%j.out
set -euo pipefail
cd /projects/Zeroshot/libero_original_evaluation_v1
sha256sum -c SHA256SUMS
export OMP_NUM_THREADS=2
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
PY=/projects/Zeroshot/envs/libero-eval-v1/bin/python
# First complete rollout counts toward4000. Increase operational segment length
# only after checking Linux qualification and timings; no change to task budget.
N=${1:-1}
"$PY" run_evaluation.py \
  --libero-root /projects/Zeroshot/libero_runtime_v1 \
  --checkpoint /projects/Zeroshot/runs/oxe-dino-noaug-s42/final_policy.pt \
  --training-seed 42 --condition no_augmentation \
  --clip-assets /projects/Zeroshot/baseline_setup/clip-vit-large-patch14-v1 \
  --dino-assets /projects/Zeroshot/dinov2_large_frozen_assets_20261023 \
  --frozen-profile /projects/Zeroshot/libero_original_evaluation_v1/training_frozen_profile.json \
  --output /projects/Zeroshot/runs/libero-original-dino-noaug-s42 \
  --max-new-episodes "$N" --wall-seconds 39600

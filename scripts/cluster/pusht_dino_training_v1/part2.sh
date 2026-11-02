#!/bin/bash
#SBATCH --job-name=pusht-dino2
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=20:00:00
#SBATCH --no-requeue
#SBATCH --chdir=/projects/Zeroshot
#SBATCH --output=/projects/Zeroshot/logs/pusht-dino-part2-%j.out
set -euo pipefail
ROOT=/projects/Zeroshot
BUNDLE="$ROOT/pusht_dino_training_v1"
OFFICIAL="$ROOT/envs/official-pusht-v1"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "$BUNDLE"
sha256sum -c SHA256SUMS
ARGS=(--root "$ROOT" --output "$ROOT/runs/pusht-dino-part2-${SLURM_JOB_ID}" --segment 2)
: "${PUSHT_DINO_RESUME:?Set exact verified segment1 latest.pt}"
ARGS+=(--resume "$PUSHT_DINO_RESUME")
LD_LIBRARY_PATH="$OFFICIAL/lib" "$OFFICIAL/bin/python" -B train_dino.py "${ARGS[@]}"

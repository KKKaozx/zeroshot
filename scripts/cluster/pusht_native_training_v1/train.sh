#!/bin/bash
#SBATCH --job-name=pusht-native20k
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=22G
#SBATCH --time=08:00:00
#SBATCH --chdir=/projects/Zeroshot
#SBATCH --output=/projects/Zeroshot/logs/pusht-native20k-%j.out
set -euo pipefail
ROOT=/projects/Zeroshot
BUNDLE="$ROOT/pusht_native_training_v1"
OFFICIAL="$ROOT/envs/official-pusht-v1"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "$BUNDLE"
sha256sum -c SHA256SUMS
ARGS=(--root "$ROOT" --output "$ROOT/runs/pusht-native20k-${SLURM_JOB_ID}")
if [[ -n "${PUSHT_NATIVE_RESUME:-}" ]]; then
  ARGS+=(--resume "$PUSHT_NATIVE_RESUME")
fi
LD_LIBRARY_PATH="$OFFICIAL/lib" "$OFFICIAL/bin/python" -B train_native.py "${ARGS[@]}"

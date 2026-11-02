#!/bin/bash
#SBATCH --job-name=pusht-float-data
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --partition=cluster02
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:15:00
#SBATCH --chdir=/projects/Zeroshot
#SBATCH --output=/projects/Zeroshot/logs/pusht-float-data-%j.out
set -euo pipefail
ROOT=/projects/Zeroshot
BUNDLE="$ROOT/pusht_float_image_repair_v1"
TASK_ENV="$ROOT/envs/official-pusht-v1"
export CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export LD_LIBRARY_PATH="$TASK_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONUNBUFFERED=1
cd "$BUNDLE"
sha256sum -c SHA256SUMS
"$TASK_ENV/bin/python" -B "$BUNDLE/check_data.py" \
  --root "$ROOT" --output "$ROOT/runs/pusht-float-data-${SLURM_JOB_ID}"

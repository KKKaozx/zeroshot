#!/bin/bash
#SBATCH --job-name=pusht-front-check
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=22G
#SBATCH --time=00:45:00
#SBATCH --chdir=/projects/Zeroshot
#SBATCH --output=/projects/Zeroshot/logs/pusht-front-resource-%j.out
set -euo pipefail
ROOT=/projects/Zeroshot
BUNDLE="$ROOT/pusht_frontend_resource_v1"
OFFICIAL="$ROOT/envs/official-pusht-v1"
PROJECT="$ROOT/envs/multitask-preflight-v1"
OUT="$ROOT/runs/pusht-front-resource-${SLURM_JOB_ID}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy
cd "$BUNDLE"
sha256sum -c SHA256SUMS
LD_LIBRARY_PATH="$OFFICIAL/lib" "$OFFICIAL/bin/python" -B resource_probe.py prepare --root "$ROOT" --output "$OUT"
LD_LIBRARY_PATH="$PROJECT/lib" "$PROJECT/bin/python" -B resource_probe.py clip --root "$ROOT" --output "$OUT"
LD_LIBRARY_PATH="$OFFICIAL/lib" "$OFFICIAL/bin/python" -B resource_probe.py native --root "$ROOT" --output "$OUT"
LD_LIBRARY_PATH="$OFFICIAL/lib" "$OFFICIAL/bin/python" -B resource_probe.py project --root "$ROOT" --output "$OUT"
LD_LIBRARY_PATH="$OFFICIAL/lib" "$OFFICIAL/bin/python" -B resource_probe.py summary --root "$ROOT" --output "$OUT"

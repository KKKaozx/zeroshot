#!/bin/bash
#SBATCH --job-name=oxe-bridge-repaired
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --chdir=/projects/Zeroshot
#SBATCH --output=/projects/Zeroshot/logs/oxe-bridge-supervision-repaired-%j.out
set -euo pipefail
ROOT=/projects/Zeroshot
PROJECT="$ROOT/envs/multitask-preflight-v1"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export LD_LIBRARY_PATH="$PROJECT/lib"
PY="$PROJECT/bin/python"
cd "$ROOT/oxe_bridge_supervision_frozen_repair_v1"
sha256sum -c SHA256SUMS
timeout --kill-after=5s 60s "$PY" -B - <<'HEALTH'
import torch
assert torch.cuda.is_available() and torch.cuda.get_device_name()=='NVIDIA A40'
torch.ones(1,device='cuda');torch.cuda.synchronize()
print('CUDA_HEALTH_PASSED A40')
HEALTH
MODE="${REPAIR_MODE:?Need fit or evaluate}"
if [[ "$MODE" == fit ]]; then
  "$PY" -B -u run_repaired.py --root "$ROOT" --stage "${REPAIR_STAGE:?Need stage}"
  "$PY" -B "$ROOT/oxe_bridge_supervision_v1/verify.py" "$ROOT/runs/oxe-bridge-supervision-s42" --mode fit --stage "$REPAIR_STAGE"
elif [[ "$MODE" == evaluate ]]; then
  "$PY" -B -u run_repaired.py --root "$ROOT" --mode evaluate
else
  exit 2
fi

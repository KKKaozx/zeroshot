#!/bin/bash
#SBATCH --job-name=oxe-bridge-supervision
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --chdir=/projects/Zeroshot
#SBATCH --output=/projects/Zeroshot/logs/oxe-bridge-supervision-%j.out
set -euo pipefail
ROOT=/projects/Zeroshot
BUNDLE="$ROOT/oxe_bridge_supervision_v1"
PROJECT="$ROOT/envs/multitask-preflight-v1"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export LD_LIBRARY_PATH="$PROJECT/lib"
PY="$PROJECT/bin/python"
cd "$BUNDLE"
sha256sum -c SHA256SUMS
nvidia-smi --query-gpu=uuid,pci.bus_id,name --format=csv,noheader
timeout --kill-after=5s 60s "$PY" -B - <<'HEALTH'
import torch
assert torch.cuda.is_available(),'CUDA unavailable; no training started'
x=torch.ones(1,device='cuda');torch.cuda.synchronize()
assert float(x.cpu()[0])==1.
assert torch.cuda.get_device_name()=='NVIDIA A40'
print('CUDA_HEALTH_PASSED A40')
HEALTH
MODE="${SUPERVISION_MODE:?Need preflight,fit,evaluate}"
if [[ "$MODE" == preflight ]]; then
  "$PY" -B -u run_experiment.py --root "$ROOT" --mode preflight
  "$PY" -B verify.py "$ROOT/runs/oxe-bridge-supervision-preflight-${SLURM_JOB_ID}" --mode preflight
elif [[ "$MODE" == fit ]]; then
  : "${SUPERVISION_STAGE:?Need stage1 or2}" "${SUPERVISION_PREFLIGHT:?Need new preflight ID}"
  "$PY" -B -u run_experiment.py --root "$ROOT" --mode fit --stage "$SUPERVISION_STAGE"     --preflight "$ROOT/runs/oxe-bridge-supervision-preflight-${SUPERVISION_PREFLIGHT}"
  "$PY" -B verify.py "$ROOT/runs/oxe-bridge-supervision-s42" --mode fit --stage "$SUPERVISION_STAGE"
elif [[ "$MODE" == evaluate ]]; then
  "$PY" -B -u evaluate.py --root "$ROOT" --output "$ROOT/runs/oxe-bridge-supervision-eval-${SLURM_JOB_ID}"
else
  echo "Unknown mode"; exit 2
fi

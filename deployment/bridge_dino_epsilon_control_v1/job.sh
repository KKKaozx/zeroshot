#!/bin/bash
#SBATCH --job-name=bridge-dino-epsilon
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40960M
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --chdir=/projects/Zeroshot
#SBATCH --output=/projects/Zeroshot/logs/bridge-dino-epsilon-%j.out
set -euo pipefail
ROOT=/projects/Zeroshot
source "$ROOT/scripts/ssd_cache.sh"
export TMPDIR="$ROOT/.tmp/${SLURM_JOB_ID}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 USE_TF=0 OMP_NUM_THREADS=4
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
mkdir -p "$TMPDIR" "$ROOT/runs"
PY="$ROOT/envs/multitask-preflight-v1/bin/python"
cd "$ROOT/bridge_dino_epsilon_control_v1"
sha256sum -c SHA256SUMS
timeout --kill-after=5s 60s "$PY" -B - <<'HEALTH'
import torch
assert torch.cuda.is_available() and torch.cuda.get_device_name()=='NVIDIA A40'
assert str(torch.__version__)=='2.5.1+cu118'
torch.ones(1,device='cuda');torch.cuda.synchronize()
print('CUDA_HEALTH_PASSED A40')
HEALTH
"$PY" -B -u run.py --root "$ROOT" --output "$ROOT/runs/bridge-dino-epsilon-${SLURM_JOB_ID}"

#!/bin/bash
#SBATCH --job-name=bridge-backbone-pair-a40
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40960M
#SBATCH --time=04:00:00
#SBATCH --chdir=/projects/Zeroshot
#SBATCH --output=/projects/Zeroshot/logs/bridge-backbone-pair-a40-%j.out
set -euo pipefail
ROOT=/projects/Zeroshot
BUNDLE="$ROOT/bridge_backbone_pair_a40_v1"
source "$ROOT/scripts/ssd_cache.sh"
export TMPDIR="$ROOT/.tmp/${SLURM_JOB_ID}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 USE_TF=0 OMP_NUM_THREADS=4
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
mkdir -p "$TMPDIR" "$ROOT/runs"
PY="$ROOT/envs/multitask-preflight-v1/bin/python"
echo "PAIR_ALLOCATED job=${SLURM_JOB_ID} node=${SLURMD_NODENAME:-unknown} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
nvidia-smi --query-gpu=uuid,pci.bus_id,name --format=csv,noheader
timeout --kill-after=5s 60s "$PY" -B - <<'HEALTH'
import torch
assert torch.cuda.is_available(), 'CUDA unavailable; no training started'
x=torch.ones(1,device='cuda');torch.cuda.synchronize()
assert float(x.cpu()[0])==1.
assert torch.cuda.get_device_name()=='NVIDIA A40'
assert str(torch.__version__)=='2.5.1+cu118'
print('CUDA_HEALTH_PASSED A40; no model or optimizer loaded')
HEALTH
cd "$BUNDLE"
sha256sum -c SHA256SUMS
"$PY" -B -u run_pair.py --root "$ROOT"   --output "$ROOT/runs/bridge-backbone-pair-a40-${SLURM_JOB_ID}"

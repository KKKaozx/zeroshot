#!/usr/bin/env bash
#SBATCH --job-name=oxe-target-short
#SBATCH --partition=cluster02
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --no-requeue
#SBATCH --chdir=/projects/Zeroshot
#SBATCH --output=/projects/Zeroshot/logs/oxe-target-short-%j.out
set -euo pipefail
ROOT=/projects/Zeroshot
PROJECT="$ROOT/envs/multitask-preflight-v1"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export LD_LIBRARY_PATH="$PROJECT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "$ROOT/oxe_target_pair_short_v1"
sha256sum -c SHA256SUMS
"$PROJECT/bin/python" -B - <<'PY'
import torch
assert torch.cuda.is_available() and torch.cuda.get_device_name()=='NVIDIA A40'
x=torch.ones(2,device='cuda');assert float(x.sum())==2
print('CUDA_HEALTH_PASSED',torch.cuda.get_device_name(),str(torch.__version__),torch.version.cuda,flush=True)
PY
"$PROJECT/bin/python" -B run_pair.py --root "$ROOT" --output "$ROOT/runs/oxe-target-short-${SLURM_JOB_ID}"

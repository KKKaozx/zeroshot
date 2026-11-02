#!/bin/bash
#SBATCH --job-name=bridge-dino-only
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=22528M
#SBATCH --time=03:00:00
#SBATCH --chdir=/projects/Zeroshot
#SBATCH --output=/projects/Zeroshot/logs/bridge-dino-only-%j.out
set -euo pipefail
ROOT=/projects/Zeroshot
source "$ROOT/scripts/ssd_cache.sh"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
PY="$ROOT/envs/multitask-preflight-v1/bin/python"
echo "CUDA_HEALTH_BEFORE_TRAIN job=${SLURM_JOB_ID} node=${SLURMD_NODENAME:-unknown} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
nvidia-smi --query-gpu=uuid,pci.bus_id,name --format=csv,noheader
timeout --kill-after=5s 60s "$PY" -B - <<'HEALTH'
import json,os,torch
torch.cuda.init()
x=torch.ones(1,device='cuda')
torch.cuda.synchronize()
assert float(x.cpu()[0])==1.
print('CUDA_HEALTH_PASSED '+json.dumps(dict(gpu=torch.cuda.get_device_name(),
    torch_version=str(torch.__version__),cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
    optimizer_updates=0,model_loaded=False)))
HEALTH
exec bash "$ROOT/bridge_dino_only_v1/job.sh"

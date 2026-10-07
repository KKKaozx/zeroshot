#!/bin/bash
#SBATCH --job-name=multitask-check
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=22528M
#SBATCH --time=00:15:00
#SBATCH --chdir=/projects/Zeroshot
#SBATCH --output=/projects/Zeroshot/logs/multitask-preflight-%j.out
set -eo pipefail
source /projects/Zeroshot/scripts/ssd_cache.sh
export TMPDIR="/projects/Zeroshot/.tmp/${SLURM_JOB_ID}"
export OMP_NUM_THREADS=4
mkdir -p "$TMPDIR" /projects/Zeroshot/runs
nvidia-smi
# Reuse PyTorch; transformers must be present. No automatic package/weight downloads.
/projects/Zeroshot/envs/bridge-diffusion/bin/python -u \
  /projects/Zeroshot/multitask_preflight_v1/multitask_resource_preflight.py \
  --output "/projects/Zeroshot/runs/multitask-preflight-${SLURM_JOB_ID}.json"

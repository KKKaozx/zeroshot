#!/bin/bash
#SBATCH --job-name=bridge-diffusion-fit
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=00:30:00
#SBATCH --chdir=/projects/Zeroshot
#SBATCH --output=/projects/Zeroshot/logs/bridge-diffusion-fit-%j.out
set -eo pipefail
source /projects/Zeroshot/scripts/ssd_cache.sh
export TMPDIR="/projects/Zeroshot/.tmp/$SLURM_JOB_ID"
mkdir -p "$TMPDIR" /projects/Zeroshot/runs
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4
export WANDB_MODE=disabled
nvidia-smi
/projects/Zeroshot/envs/bridge-diffusion/bin/python -u \
  /projects/Zeroshot/baseline_setup/bridge_diffusion_fit_v1/run_fit.py \
  --output-dir "/projects/Zeroshot/runs/bridge-diffusion-fit-$SLURM_JOB_ID"

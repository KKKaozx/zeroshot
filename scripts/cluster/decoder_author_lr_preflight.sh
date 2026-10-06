#!/bin/bash
#SBATCH --job-name=author-lr-check
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=00:15:00
#SBATCH --chdir=/projects/Zeroshot
#SBATCH --output=/projects/Zeroshot/logs/author-lr-preflight-%j.out
set -eo pipefail
source /projects/Zeroshot/scripts/ssd_cache.sh
export TMPDIR="/projects/Zeroshot/.tmp/${SLURM_JOB_ID}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4
mkdir -p "$TMPDIR" /projects/Zeroshot/runs
nvidia-smi
/projects/Zeroshot/envs/bridge-diffusion/bin/python -u \
  /projects/Zeroshot/baseline_setup/decoder_author_lr_v1/run_decoder_pair.py \
  --mode preflight --arm author --learning-rate 3e-5 --monitor-train \
  --output-dir "/projects/Zeroshot/runs/author-lr-preflight-${SLURM_JOB_ID}"

#!/bin/bash
#SBATCH --job-name=author-lr-train
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=02:00:00
#SBATCH --chdir=/projects/Zeroshot
#SBATCH --output=/projects/Zeroshot/logs/author-lr-train-%j.out
set -eo pipefail
: "${AUTHOR_LR_PREFLIGHT:?Set AUTHOR_LR_PREFLIGHT to the new passed preflight.json path}"
test -f "$AUTHOR_LR_PREFLIGHT"
source /projects/Zeroshot/scripts/ssd_cache.sh
export TMPDIR="/projects/Zeroshot/.tmp/${SLURM_JOB_ID}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4
mkdir -p "$TMPDIR" /projects/Zeroshot/runs
nvidia-smi
/projects/Zeroshot/envs/bridge-diffusion/bin/python -u \
  /projects/Zeroshot/baseline_setup/decoder_author_lr_v1/run_decoder_pair.py \
  --mode train --arm author --learning-rate 3e-5 --monitor-train \
  --preflight-report "$AUTHOR_LR_PREFLIGHT" \
  --output-dir "/projects/Zeroshot/runs/author-lr-${SLURM_JOB_ID}"

#!/bin/bash
#SBATCH --job-name=decoder-pair-train
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=06:00:00
#SBATCH --chdir=/projects/Zeroshot
#SBATCH --output=/projects/Zeroshot/logs/decoder-pair-train-%j.out
set -eo pipefail
: "${PAIR_PREFLIGHT:?Set PAIR_PREFLIGHT to the passed preflight.json absolute path}"
test -f "$PAIR_PREFLIGHT"
source /projects/Zeroshot/scripts/ssd_cache.sh
export TMPDIR="/projects/Zeroshot/.tmp/${SLURM_JOB_ID}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4
mkdir -p "$TMPDIR" /projects/Zeroshot/runs
nvidia-smi
/projects/Zeroshot/envs/bridge-diffusion/bin/python -u \
  /projects/Zeroshot/baseline_setup/decoder_pair_v1/run_decoder_pair.py \
  --mode train --preflight-report "$PAIR_PREFLIGHT" \
  --output-dir "/projects/Zeroshot/runs/decoder-pair-${SLURM_JOB_ID}"

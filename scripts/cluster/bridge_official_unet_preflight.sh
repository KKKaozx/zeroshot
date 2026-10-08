#!/bin/bash
#SBATCH --job-name=bridge-unet-check
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40960M
#SBATCH --time=00:10:00
#SBATCH --chdir=/projects/Zeroshot
#SBATCH --output=/projects/Zeroshot/logs/bridge-unet-check-%j.out

set -eo pipefail
source /projects/Zeroshot/scripts/ssd_cache.sh
export TMPDIR="/projects/Zeroshot/.tmp/${SLURM_JOB_ID}"
export OMP_NUM_THREADS=4
mkdir -p "$TMPDIR" /projects/Zeroshot/logs /projects/Zeroshot/runs

cd /projects/Zeroshot/bridge_official_unet_preflight_v1
sha256sum -c SHA256SUMS
/projects/Zeroshot/envs/multitask-preflight-v1/bin/python -u preflight_official_unet.py \
  --output "/projects/Zeroshot/runs/bridge-unet-check-${SLURM_JOB_ID}.json"

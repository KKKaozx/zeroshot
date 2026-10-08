#!/bin/bash
#SBATCH --job-name=bridge-unet-train
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40960M
#SBATCH --time=01:30:00
#SBATCH --chdir=/projects/Zeroshot
#SBATCH --output=/projects/Zeroshot/logs/bridge-unet-train-%j.out

set -eo pipefail
source /projects/Zeroshot/scripts/ssd_cache.sh
export TMPDIR="/projects/Zeroshot/.tmp/${SLURM_JOB_ID}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export USE_TF=0
export OMP_NUM_THREADS=4
mkdir -p "$TMPDIR" /projects/Zeroshot/logs /projects/Zeroshot/runs

cd /projects/Zeroshot/bridge_official_unet_training_v1
sha256sum -c SHA256SUMS
(cd /projects/Zeroshot/bridge_expansion_training_v1 && sha256sum -c SHA256SUMS)

/projects/Zeroshot/envs/multitask-preflight-v1/bin/python -u run_bridge_official_unet_training.py \
  --pack /projects/Zeroshot/bridge_expansion_training_v1 \
  --head-report /projects/Zeroshot/runs/bridge-head-train-192651/report.json \
  --window-reference /projects/Zeroshot/runs/bridge-full-chain-191004.json \
  --preflight /projects/Zeroshot/runs/bridge-unet-real-193711.json \
  --output "/projects/Zeroshot/runs/bridge-unet-train-${SLURM_JOB_ID}"

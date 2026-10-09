#!/bin/bash
#SBATCH --job-name=unet-tail-probe
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40960M
#SBATCH --time=00:15:00
#SBATCH --chdir=/projects/Zeroshot
#SBATCH --output=/projects/Zeroshot/logs/unet-tail-%j.out
set -eo pipefail
source /projects/Zeroshot/scripts/ssd_cache.sh
export TMPDIR="/projects/Zeroshot/.tmp/${SLURM_JOB_ID}"
export OMP_NUM_THREADS=4
mkdir -p "$TMPDIR"
cd /projects/Zeroshot/multiscale_tail_probe_v1
sha256sum -c SHA256SUMS
/projects/Zeroshot/envs/multitask-preflight-v1/bin/python -u diagnose_multiscale_noise.py \
  --pack /projects/Zeroshot/bridge_expansion_training_v1 \
  --training-run /projects/Zeroshot/runs/bridge-unet-train-193761 \
  --training-report /projects/Zeroshot/runs/bridge-unet-train-193761/report.json \
  --window-reference /projects/Zeroshot/runs/bridge-full-chain-191004.json \
  --reference-probe /projects/Zeroshot/runs/unet-noise-194978.json \
  --chain-tail \
  --output /projects/Zeroshot/runs/unet-tail-${SLURM_JOB_ID}.json

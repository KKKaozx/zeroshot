#!/bin/bash
#SBATCH --job-name=bridge-ddim
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40960M
#SBATCH --time=00:25:00
#SBATCH --chdir=/projects/Zeroshot
#SBATCH --output=/projects/Zeroshot/logs/bridge-ddim-%j.out

set -eo pipefail
source /projects/Zeroshot/scripts/ssd_cache.sh
export TMPDIR="/projects/Zeroshot/.tmp/${SLURM_JOB_ID}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export USE_TF=0
export OMP_NUM_THREADS=4
mkdir -p "$TMPDIR"

cd /projects/Zeroshot/bridge_ddim_steps_v1
sha256sum -c SHA256SUMS
(cd /projects/Zeroshot/bridge_expansion_training_v1 && sha256sum -c SHA256SUMS)

/projects/Zeroshot/envs/multitask-preflight-v1/bin/python -u evaluate_bridge_ddim_steps.py \
  --pack /projects/Zeroshot/bridge_expansion_training_v1 \
  --head-run /projects/Zeroshot/runs/bridge-head-train-192651 \
  --window-reference /projects/Zeroshot/runs/bridge-full-chain-191004.json \
  --head-report /projects/Zeroshot/runs/bridge-head-train-192651/report.json \
  --one-step-report /projects/Zeroshot/runs/bridge-one-step-193030/report.json \
  --output "/projects/Zeroshot/runs/bridge-ddim-${SLURM_JOB_ID}"

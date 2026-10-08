#!/bin/bash
#SBATCH --job-name=bridge-one-step
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40960M
#SBATCH --time=00:20:00
#SBATCH --chdir=/projects/Zeroshot
#SBATCH --output=/projects/Zeroshot/logs/bridge-one-step-%j.out

set -eo pipefail
source /projects/Zeroshot/scripts/ssd_cache.sh
export TMPDIR="/projects/Zeroshot/.tmp/${SLURM_JOB_ID}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export USE_TF=0
export OMP_NUM_THREADS=4
mkdir -p "$TMPDIR"

cd /projects/Zeroshot/bridge_one_step_sampler_v1
sha256sum -c SHA256SUMS
(cd /projects/Zeroshot/bridge_expansion_training_v1 && sha256sum -c SHA256SUMS)

/projects/Zeroshot/envs/multitask-preflight-v1/bin/python -u evaluate_bridge_one_step_sampler.py \
  --pack /projects/Zeroshot/bridge_expansion_training_v1 \
  --head-run /projects/Zeroshot/runs/bridge-head-train-192651 \
  --window-reference /projects/Zeroshot/runs/bridge-full-chain-191004.json \
  --head-report /projects/Zeroshot/runs/bridge-head-train-192651/report.json \
  --output "/projects/Zeroshot/runs/bridge-one-step-${SLURM_JOB_ID}"

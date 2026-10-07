#!/bin/bash
#SBATCH --job-name=bridge-full-chain
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=49152M
#SBATCH --time=00:40:00
#SBATCH --chdir=/projects/Zeroshot
#SBATCH --output=/projects/Zeroshot/logs/bridge-full-chain-%j.out
set -eo pipefail
source /projects/Zeroshot/scripts/ssd_cache.sh
export TMPDIR="/projects/Zeroshot/.tmp/${SLURM_JOB_ID}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export USE_TF=0
export OMP_NUM_THREADS=4
mkdir -p "$TMPDIR"
cd /projects/Zeroshot/bridge_full_chain_probe_v1
sha256sum -c SHA256SUMS
/projects/Zeroshot/envs/multitask-preflight-v1/bin/python -u probe_bridge_conditioning.py \
  --pack /projects/Zeroshot/bridge_expansion_training_v1 \
  --run /projects/Zeroshot/runs/bridge_expansion_training_v1-190189 \
  --chain-probe --all-train-windows \
  --reference-report /projects/Zeroshot/runs/bridge-conditioning-190853.json \
  --output "/projects/Zeroshot/runs/bridge-full-chain-${SLURM_JOB_ID}.json"

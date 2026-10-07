#!/bin/bash
#SBATCH --job-name=bridge-condition
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=22528M
#SBATCH --time=00:45:00
#SBATCH --chdir=/projects/Zeroshot
#SBATCH --output=/projects/Zeroshot/logs/bridge-condition-%j.out
set -eo pipefail
source /projects/Zeroshot/scripts/ssd_cache.sh
export TMPDIR="/projects/Zeroshot/.tmp/${SLURM_JOB_ID}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export USE_TF=0
export OMP_NUM_THREADS=4
mkdir -p "$TMPDIR"
cd /projects/Zeroshot/bridge_conditioning_probe_v1
sha256sum -c SHA256SUMS
/projects/Zeroshot/envs/multitask-preflight-v1/bin/python -u probe_bridge_conditioning.py \
  --pack /projects/Zeroshot/bridge_expansion_training_v1 \
  --run /projects/Zeroshot/runs/bridge_expansion_training_v1-190189 \
  --output "/projects/Zeroshot/runs/bridge-conditioning-${SLURM_JOB_ID}.json"

#!/bin/bash
#SBATCH --job-name=octo-bridge-step
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --chdir=/projects/Zeroshot
#SBATCH --output=/projects/Zeroshot/logs/octo-bridge-step-%j.out
set -eo pipefail
source /projects/Zeroshot/scripts/ssd_cache.sh
export TMPDIR="/projects/Zeroshot/.tmp/${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export XLA_PYTHON_CLIENT_PREALLOCATE=false
unset JAX_PLATFORMS
nvidia-smi
/projects/Zeroshot/envs/octo-small-v1/bin/python \
    /projects/Zeroshot/octo_bridge_single_step_v1/octo_bridge_single_step.py \
    --data /projects/Zeroshot/data/bridge_single_step_subset \
    --source /projects/Zeroshot/baselines/octo-official-v1 \
    --plan /projects/Zeroshot/octo_official_smoke_v1/octo_minimal_reproduction.json \
    --assets /projects/Zeroshot/baseline_setup/octo-assets-v1 \
    --output "/projects/Zeroshot/runs/octo-bridge-step-${SLURM_JOB_ID}"

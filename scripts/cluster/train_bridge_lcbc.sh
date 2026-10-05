#!/bin/bash
#SBATCH --job-name=bridge-lcbc-v1
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:a40:1
#SBATCH --time=02:00:00
#SBATCH --chdir=/projects/Zeroshot
#SBATCH --output=/projects/Zeroshot/logs/bridge-lcbc-%j.out

set -eo pipefail
source /projects/Zeroshot/scripts/ssd_cache.sh
export TMPDIR="/projects/Zeroshot/.tmp/${SLURM_JOB_ID}"
mkdir -p "$TMPDIR" /projects/Zeroshot/runs
unset JAX_PLATFORMS
export JAX_PLATFORM_NAME=gpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export WANDB_MODE=disabled
BRIDGE_REPO=/projects/Zeroshot/baselines/bridge_data_v2
[ "$(git -C "$BRIDGE_REPO" rev-parse HEAD)" = bc60a35b701a12021c8c95e9d8601274d3acd928 ]
git -C "$BRIDGE_REPO" diff --quiet
git -C "$BRIDGE_REPO" diff --cached --quiet
export PYTHONPATH="$BRIDGE_REPO${PYTHONPATH:+:$PYTHONPATH}"
nvidia-smi
/projects/Zeroshot/envs/zeroshot/bin/python -u /projects/Zeroshot/scripts/run_bridge_lcbc_subset.py \
  --data-dir /projects/Zeroshot/data/bridge_single_step_subset \
  --language-cache /projects/Zeroshot/baseline_setup/muse_fixed_subset.npz \
  --output-dir "/projects/Zeroshot/runs/bridge-lcbc-${SLURM_JOB_ID}" \
  --steps 1000 --batch-size 16 --seed 42

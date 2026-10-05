#!/bin/bash
#SBATCH --job-name=bridge-lcbc-replay
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=00:20:00
#SBATCH --chdir=/projects/Zeroshot
#SBATCH --output=/projects/Zeroshot/logs/bridge-lcbc-replay-%j.out

set -eo pipefail
source /projects/Zeroshot/scripts/ssd_cache.sh
export TMPDIR="/projects/Zeroshot/.tmp/$SLURM_JOB_ID"
mkdir -p "$TMPDIR" /projects/Zeroshot/runs
unset JAX_PLATFORMS
export JAX_PLATFORM_NAME=gpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export WANDB_MODE=disabled
BRIDGE_REPO=/projects/Zeroshot/baselines/bridge_data_v2
[ "$(git -C "$BRIDGE_REPO" rev-parse HEAD)" = bc60a35b701a12021c8c95e9d8601274d3acd928 ]
git -C "$BRIDGE_REPO" diff --quiet
git -C "$BRIDGE_REPO" diff --cached --quiet
test -f /projects/Zeroshot/scripts/run_bridge_lcbc_wrap.py
export PYTHONPATH="$BRIDGE_REPO"
nvidia-smi
/projects/Zeroshot/envs/zeroshot/bin/python -u /projects/Zeroshot/scripts/run_bridge_lcbc_single_episode.py \
  --data-dir /projects/Zeroshot/data/bridge_single_step_subset \
  --language-cache /projects/Zeroshot/baseline_setup/muse_fixed_subset.npz \
  --output-dir "/projects/Zeroshot/runs/bridge-lcbc-replay-$SLURM_JOB_ID"

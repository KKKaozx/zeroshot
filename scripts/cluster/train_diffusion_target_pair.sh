#!/bin/bash
#SBATCH --job-name=diffusion-target-pair
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=00:30:00
#SBATCH --chdir=/projects/Zeroshot
#SBATCH --output=/projects/Zeroshot/logs/diffusion-target-pair-%j.out
set -eo pipefail
source /projects/Zeroshot/scripts/ssd_cache.sh
export TMPDIR="/projects/Zeroshot/.tmp/$SLURM_JOB_ID"
mkdir -p "$TMPDIR" /projects/Zeroshot/runs
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4
export WANDB_MODE=disabled
PYTHON=/projects/Zeroshot/envs/bridge-diffusion/bin/python
test -x "$PYTHON"
cd /projects/Zeroshot
sha256sum -c bridge_diffusion_target_pair_v1.sha256
"$PYTHON" -m zipfile -e bridge_diffusion_target_pair_v1.zip "$TMPDIR"
nvidia-smi
"$PYTHON" -u "$TMPDIR/bridge_diffusion_target_pair_v1/run_pair.py" \
  --output-dir "/projects/Zeroshot/runs/diffusion-target-pair-$SLURM_JOB_ID"

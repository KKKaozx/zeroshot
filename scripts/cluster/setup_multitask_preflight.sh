#!/bin/bash
#SBATCH --job-name=multitask-setup
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --partition=cluster02
#SBATCH --cpus-per-task=4
#SBATCH --mem=8192M
#SBATCH --time=01:00:00
#SBATCH --chdir=/projects/Zeroshot
#SBATCH --output=/projects/Zeroshot/logs/multitask-setup-%j.out
set -eo pipefail
source /projects/Zeroshot/scripts/ssd_cache.sh
export TMPDIR="/projects/Zeroshot/.tmp/${SLURM_JOB_ID}"
export HF_HUB_DISABLE_XET=1
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
mkdir -p "$TMPDIR" /projects/Zeroshot/baseline_setup
ENV_PATH=/projects/Zeroshot/envs/multitask-preflight-v1
BASE_PY=/projects/Zeroshot/envs/bridge-diffusion/bin/python
# Inherit the existing PyTorch installation; new dependencies stay in this venv.
if [ ! -e "$ENV_PATH" ]; then
  "$BASE_PY" -m venv --system-site-packages "$ENV_PATH"
fi
printf 'torch==2.5.1+cu118\n' > "$TMPDIR/constraints.txt"
"$ENV_PATH/bin/python" -m pip install --constraint "$TMPDIR/constraints.txt" 'transformers==5.17.0'
"$ENV_PATH/bin/python" -m pip check
"$ENV_PATH/bin/python" -u /projects/Zeroshot/multitask_preflight_v2/setup_clip_assets.py

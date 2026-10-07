#!/bin/bash
#SBATCH --job-name=multitask-reader
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --partition=cluster02
#SBATCH --cpus-per-task=4
#SBATCH --mem=8192M
#SBATCH --time=00:30:00
#SBATCH --chdir=/projects/Zeroshot
#SBATCH --output=/projects/Zeroshot/logs/multitask-reader-%j.out
set -eo pipefail
source /projects/Zeroshot/scripts/ssd_cache.sh
export TMPDIR="/projects/Zeroshot/.tmp/${SLURM_JOB_ID}"
export USE_TF=0
mkdir -p "$TMPDIR"
PY=/projects/Zeroshot/envs/multitask-preflight-v1/bin/python
printf 'torch==2.5.1+cu118\ntransformers==5.17.0\n' > "$TMPDIR/constraints.txt"
"$PY" -m pip install --constraint "$TMPDIR/constraints.txt" 'tensorflow==2.15.0'
"$PY" -m pip check
PACK=/projects/Zeroshot/multitask_training_v1
cd "$PACK"
sha256sum -c SHA256SUMS
"$PY" -u train.py --prepare-only --dataset-dir "$PACK/data" --bridge-task-plan "$PACK/manifest.json" \
  --output-dir "/projects/Zeroshot/baseline_setup/multitask-split-${SLURM_JOB_ID}" \
  --sources tfrecord --exclude-path-parts '' --exclude-schemas '' --min-trajectory-steps 17 \
  --bridge-current-gripper continuous --bridge-gripper-policy reverse_scan_valid_steps_v2 \
  --gripper-target-mode state --no-balanced-gripper-loss --gripper-change-weight 1 --no-balanced-sampling --adapter-pooling cls_patch_mean \
  --decoder-type diffusion --diffusion-prediction-type sample

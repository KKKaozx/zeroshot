#!/bin/bash
#SBATCH --job-name=bridge-ema-eval
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40960M
#SBATCH --time=03:00:00
#SBATCH --chdir=/projects/Zeroshot
#SBATCH --output=/projects/Zeroshot/logs/bridge-ema-eval-%j.out

set -eo pipefail
: "${EMA_TRAIN_ID:?Submit with --export=ALL,EMA_TRAIN_ID=the_completed_training_job_id}"
[[ "$EMA_TRAIN_ID" =~ ^[0-9]+$ ]] || { echo 'Invalid training job ID'; exit 1; }
source /projects/Zeroshot/scripts/ssd_cache.sh
export TMPDIR="/projects/Zeroshot/.tmp/${SLURM_JOB_ID}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 USE_TF=0 OMP_NUM_THREADS=4
mkdir -p "$TMPDIR" /projects/Zeroshot/logs /projects/Zeroshot/runs
cd /projects/Zeroshot/bridge_ema_pair_v1
sha256sum -c SHA256SUMS
(cd /projects/Zeroshot/bridge_expansion_training_v1 && sha256sum -c SHA256SUMS)
/projects/Zeroshot/envs/multitask-preflight-v1/bin/python -u evaluate_bridge_ema_pair.py \
  --pack /projects/Zeroshot/bridge_expansion_training_v1 \
  --training-run "/projects/Zeroshot/runs/bridge-ema-train-${EMA_TRAIN_ID}" \
  --window-reference /projects/Zeroshot/runs/bridge-full-chain-191004.json \
  --output "/projects/Zeroshot/runs/bridge-ema-eval-${SLURM_JOB_ID}"

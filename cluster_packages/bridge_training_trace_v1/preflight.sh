#!/bin/bash
#SBATCH --job-name=bridge-trace-preflight
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40960M
#SBATCH --time=00:15:00
#SBATCH --chdir=/projects/Zeroshot
#SBATCH --output=/projects/Zeroshot/logs/bridge-trace-preflight-%j.out
set -eo pipefail
source /projects/Zeroshot/scripts/ssd_cache.sh
export TMPDIR="/projects/Zeroshot/.tmp/${SLURM_JOB_ID}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 USE_TF=0 OMP_NUM_THREADS=4
mkdir -p "$TMPDIR" /projects/Zeroshot/runs
cd /projects/Zeroshot/bridge_training_trace_v1
sha256sum -c SHA256SUMS
nvidia-smi
/projects/Zeroshot/envs/multitask-preflight-v1/bin/python -u trace_bridge_batch_training.py \
  --manifest manifest.json --data-root /projects/Zeroshot/bridge_expansion_training_v1/data \
  --clip /projects/Zeroshot/baseline_setup/clip-vit-large-patch14-v1 \
  --data-reference /projects/Zeroshot/bridge_batch_input_audit_v1/reference --stage preflight \
  --output "/projects/Zeroshot/runs/bridge-trace-preflight-${SLURM_JOB_ID}"

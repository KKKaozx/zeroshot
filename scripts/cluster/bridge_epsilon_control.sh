#!/bin/bash
#SBATCH --job-name=bridge-epsilon
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40960M
#SBATCH --time=01:30:00
#SBATCH --chdir=/projects/Zeroshot
#SBATCH --output=/projects/Zeroshot/logs/bridge-epsilon-%j.out

set -eo pipefail
source /projects/Zeroshot/scripts/ssd_cache.sh
export TMPDIR="/projects/Zeroshot/.tmp/${SLURM_JOB_ID}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export USE_TF=0
export OMP_NUM_THREADS=4
mkdir -p "$TMPDIR"

cd /projects/Zeroshot/bridge_epsilon_control_v1
sha256sum -c SHA256SUMS
(cd /projects/Zeroshot/bridge_expansion_training_v1 && sha256sum -c SHA256SUMS)

PHASE="${1:-preflight}"
EXTRA=()
if [ "$PHASE" = "train" ]; then
  : "${2:?Provide the passed preflight report path}"
  EXTRA=(--preflight "$2")
fi

/projects/Zeroshot/envs/multitask-preflight-v1/bin/python -u run_bridge_epsilon_control.py \
  --pack /projects/Zeroshot/bridge_expansion_training_v1 \
  --head-run /projects/Zeroshot/runs/bridge-head-train-192651 \
  --head-report /projects/Zeroshot/runs/bridge-head-train-192651/report.json \
  --window-reference /projects/Zeroshot/runs/bridge-full-chain-191004.json \
  --one-step-report /projects/Zeroshot/runs/bridge-one-step-193030/report.json \
  --ddim-report /projects/Zeroshot/runs/bridge-ddim-193087/report.json \
  --phase "$PHASE" "${EXTRA[@]}" \
  --output "/projects/Zeroshot/runs/bridge-epsilon-${PHASE}-${SLURM_JOB_ID}"

#!/bin/bash
#SBATCH --job-name=bridge-head-check
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=49152M
#SBATCH --time=00:15:00
#SBATCH --chdir=/projects/Zeroshot
#SBATCH --output=/projects/Zeroshot/logs/bridge-head-%j.out
set -eo pipefail
source /projects/Zeroshot/scripts/ssd_cache.sh
export TMPDIR="/projects/Zeroshot/.tmp/${SLURM_JOB_ID}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export USE_TF=0
export OMP_NUM_THREADS=4
mkdir -p "$TMPDIR"
cd /projects/Zeroshot/bridge_head_control_v1
sha256sum -c SHA256SUMS
(cd /projects/Zeroshot/bridge_expansion_training_v1 && sha256sum -c SHA256SUMS)
PHASE="${1:-preflight}"
EXTRA=()
if [ "$PHASE" = train ]; then
  : "${2:?Provide the passed preflight report path}"
  EXTRA=(--preflight "$2")
fi
/projects/Zeroshot/envs/multitask-preflight-v1/bin/python -u run_bridge_head_control.py \
  --pack /projects/Zeroshot/bridge_expansion_training_v1 \
  --source-run /projects/Zeroshot/runs/bridge_expansion_training_v1-190189 \
  --reference /projects/Zeroshot/runs/bridge-full-chain-191004.json \
  --phase "$PHASE" "${EXTRA[@]}" \
  --output "/projects/Zeroshot/runs/bridge-head-${PHASE}-${SLURM_JOB_ID}"

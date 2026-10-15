#!/bin/bash
#SBATCH --job-name=bridge-history-fit
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=02:30:00
#SBATCH --chdir=/projects/Zeroshot
#SBATCH --output=/projects/Zeroshot/logs/bridge-history-fit-%j.out
set -eo pipefail
source /projects/Zeroshot/scripts/ssd_cache.sh
export TMPDIR="/projects/Zeroshot/.tmp/history-fit-${SLURM_JOB_ID}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 USE_TF=0 OMP_NUM_THREADS=4
mkdir -p "$TMPDIR" /projects/Zeroshot/runs
: "${HISTORY_PREFLIGHT_JOB:?Supply successful preflight job ID}"
USED_BYTES=$(du -s -B1 /projects/_ssd/Zeroshot | cut -f1)
echo "Project bytes (du planning estimate, not live quota): $USED_BYTES"
if (( USED_BYTES + 2*1024*1024*1024 + 15*1024*1024*1024 > 200*1024*1024*1024 )); then exit 1; fi
cd /projects/Zeroshot/bridge_history_generalization_v1
sha256sum -c SHA256SUMS
nvidia-smi
/projects/Zeroshot/envs/multitask-preflight-v1/bin/python -u run_history_pair.py --project /projects/Zeroshot --stage fit --output "/projects/Zeroshot/runs/bridge-history-fit-${SLURM_JOB_ID}" --preflight-reference "/projects/Zeroshot/runs/bridge-history-preflight-${HISTORY_PREFLIGHT_JOB}"

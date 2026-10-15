#!/bin/bash
#SBATCH --job-name=bridge-lr-rep-fit
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=22528M
#SBATCH --time=02:00:00
#SBATCH --chdir=/projects/Zeroshot
#SBATCH --output=/projects/Zeroshot/logs/bridge-lr-rep-fit-%j.out
set -eo pipefail
source /projects/Zeroshot/scripts/ssd_cache.sh
export TMPDIR="/projects/Zeroshot/.tmp/${SLURM_JOB_ID}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 USE_TF=0 OMP_NUM_THREADS=4
mkdir -p "$TMPDIR" /projects/Zeroshot/runs
: "${LR_REP_PREFLIGHT_JOB:?Supply successful replication preflight job ID}"
PROJECT_USED_BYTES=$(du -s -B1 /projects/_ssd/Zeroshot | cut -f1)
echo "Project apparent bytes (planning estimate, not live quota): $PROJECT_USED_BYTES"
if (( PROJECT_USED_BYTES + 8*1024*1024*1024 + 15*1024*1024*1024 > 200*1024*1024*1024 )); then
  echo "Insufficient planning headroom"; exit 1
fi
cd /projects/Zeroshot/bridge_late_lr_replication_v1
sha256sum -c SHA256SUMS
nvidia-smi
for INIT in 42 44; do
  /projects/Zeroshot/envs/multitask-preflight-v1/bin/python -u check_bridge_late_lr_replication.py \
    --manifest manifest.json --data-root /projects/Zeroshot/bridge_expansion_training_v1/data \
    --clip /projects/Zeroshot/baseline_setup/clip-vit-large-patch14-v1 \
    --data-reference /projects/Zeroshot/bridge_batch_input_audit_v1/reference \
    --origin-report /projects/Zeroshot/runs/bridge-trace-fit-200039/report.json \
    --initialization-seed "$INIT" --stage fit \
    --output "/projects/Zeroshot/runs/bridge-lr-rep-fit-${SLURM_JOB_ID}/seed$INIT" \
    --preflight-reference "/projects/Zeroshot/runs/bridge-lr-rep-preflight-${LR_REP_PREFLIGHT_JOB}/seed$INIT"
done
echo "LR REPLICATION FIT EXECUTION COMPLETE; inspect both seed reports"

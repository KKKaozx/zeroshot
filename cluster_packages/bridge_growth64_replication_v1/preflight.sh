#!/bin/bash
#SBATCH --job-name=bridge-growth64-rep-preflight
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=22528M
#SBATCH --time=00:25:00
#SBATCH --chdir=/projects/Zeroshot
#SBATCH --output=/projects/Zeroshot/logs/bridge-growth64-rep-preflight-%j.out
set -eo pipefail
source /projects/Zeroshot/scripts/ssd_cache.sh
export TMPDIR="/projects/Zeroshot/.tmp/${SLURM_JOB_ID}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 USE_TF=0 OMP_NUM_THREADS=4
mkdir -p "$TMPDIR" /projects/Zeroshot/runs
PROJECT_USED_BYTES=$(du -s -B1 /projects/_ssd/Zeroshot | cut -f1)
echo "Project bytes (du planning estimate, not live quota): $PROJECT_USED_BYTES"
if (( PROJECT_USED_BYTES + 1024*1024*1024 + 15*1024*1024*1024 > 200*1024*1024*1024 )); then exit 1; fi
cd /projects/Zeroshot/bridge_growth64_replication_v1
sha256sum -c SHA256SUMS
nvidia-smi
/projects/Zeroshot/envs/multitask-preflight-v1/bin/python -u fit_bridge_growth64.py \
  --manifest manifest.json --data-root /projects/Zeroshot/bridge_expansion_training_v1/data \
  --clip /projects/Zeroshot/baseline_setup/clip-vit-large-patch14-v1 \
  --data-reference /projects/Zeroshot/bridge_batch_input_audit_v1/reference \
  --origin-report /projects/Zeroshot/runs/bridge-trace-fit-200039/report.json \
  --stage preflight --output "/projects/Zeroshot/runs/bridge-growth64-rep-preflight-${SLURM_JOB_ID}"

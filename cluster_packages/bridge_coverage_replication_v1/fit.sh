#!/bin/bash
#SBATCH --job-name=bridge-coverage-rep-fit
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=22528M
#SBATCH --time=03:00:00
#SBATCH --chdir=/projects/Zeroshot
#SBATCH --output=/projects/Zeroshot/logs/bridge-coverage-rep-fit-%j.out
set -eo pipefail
source /projects/Zeroshot/scripts/ssd_cache.sh
export TMPDIR="/projects/Zeroshot/.tmp/${SLURM_JOB_ID}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 USE_TF=0 OMP_NUM_THREADS=4
mkdir -p "$TMPDIR" /projects/Zeroshot/runs
: "${COVERAGE_REP_PREFLIGHT_JOB:?Supply successful COMPLETE cluster preflight ID}"
PROJECT_USED_BYTES=$(du -s -B1 /projects/_ssd/Zeroshot | cut -f1)
if (( PROJECT_USED_BYTES + 2*1024*1024*1024 + 15*1024*1024*1024 > 200*1024*1024*1024 )); then echo "Storage planning headroom insufficient"; exit 1; fi
cd /projects/Zeroshot/bridge_coverage_replication_v1
sha256sum -c SHA256SUMS
nvidia-smi
/projects/Zeroshot/envs/multitask-preflight-v1/bin/python validate_preflight.py "/projects/Zeroshot/runs/bridge-coverage-rep-preflight-${COVERAGE_REP_PREFLIGHT_JOB}"
/projects/Zeroshot/envs/multitask-preflight-v1/bin/python -u run_training_coverage.py \
  --data-root /projects/Zeroshot/bridge_expansion_training_v1/data \
  --data-reference /projects/Zeroshot/bridge_batch_input_audit_v1/reference \
  --clip /projects/Zeroshot/baseline_setup/clip-vit-large-patch14-v1 \
  --old-cache /projects/Zeroshot/bridge_state_prior_v1/full_tokens.pt \
  --cache /projects/Zeroshot/bridge_training_coverage_v1/full_tokens.pt \
  --control-report /projects/Zeroshot/runs/bridge-state-rep-fit-202463/real/report.json \
  --stage fit --output "/projects/Zeroshot/runs/bridge-coverage-rep-fit-${SLURM_JOB_ID}" \
  --preflight-reference "/projects/Zeroshot/runs/bridge-coverage-rep-preflight-${COVERAGE_REP_PREFLIGHT_JOB}"

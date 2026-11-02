#!/bin/bash
#SBATCH --job-name=bridge-demo806-fit
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=22528M
#SBATCH --time=04:00:00
#SBATCH --chdir=/projects/Zeroshot
#SBATCH --output=/projects/Zeroshot/logs/bridge-demo806-fit-%j.out
set -eo pipefail
source /projects/Zeroshot/scripts/ssd_cache.sh
export TMPDIR="/projects/Zeroshot/.tmp/${SLURM_JOB_ID}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 USE_TF=0 OMP_NUM_THREADS=4
mkdir -p "$TMPDIR" /projects/Zeroshot/runs
cd /projects/Zeroshot/bridge_demonstration_expansion_v1
sha256sum --status -c SHA256SUMS
/projects/Zeroshot/envs/multitask-preflight-v1/bin/python unpack_data.py --check-only
nvidia-smi
PROJECT_USED_BYTES=$(du -s -B1 /projects/_ssd/Zeroshot | cut -f1)
if (( PROJECT_USED_BYTES + 2*1024*1024*1024 + 15*1024*1024*1024 > 200*1024*1024*1024 )); then echo "Storage planning headroom insufficient; this is not a live quota query"; exit 1; fi
: "${DEMO806_PREFLIGHT_JOB:?Supply the successful complete806 preflight ID}"
/projects/Zeroshot/envs/multitask-preflight-v1/bin/python validate_preflight.py "/projects/Zeroshot/runs/bridge-demo806-preflight-${DEMO806_PREFLIGHT_JOB}"
/projects/Zeroshot/envs/multitask-preflight-v1/bin/python -u run_expansion.py \
  --data-root /projects/Zeroshot/bridge_expansion_training_v1/data \
  --data-reference /projects/Zeroshot/bridge_batch_input_audit_v1/reference \
  --clip /projects/Zeroshot/baseline_setup/clip-vit-large-patch14-v1 \
  --old-cache /projects/Zeroshot/bridge_training_coverage_v1/full_tokens.pt \
  --cache /projects/Zeroshot/bridge_demonstration_expansion_v1/full_tokens.pt \
  --control-report /projects/Zeroshot/runs/bridge-state-rep-fit-202463/real/report.json \
  --stage fit --output "/projects/Zeroshot/runs/bridge-demo806-fit-${SLURM_JOB_ID}" \
  --preflight-reference "/projects/Zeroshot/runs/bridge-demo806-preflight-${DEMO806_PREFLIGHT_JOB}"

#!/bin/bash
#SBATCH --job-name=bridge-scheduler-check
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40960M
#SBATCH --time=00:15:00
#SBATCH --chdir=/projects/Zeroshot
#SBATCH --output=/projects/Zeroshot/logs/bridge-scheduler-%j.out
set -eo pipefail
source /projects/Zeroshot/scripts/ssd_cache.sh
export TMPDIR="/projects/Zeroshot/.tmp/${SLURM_JOB_ID}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 USE_TF=0 OMP_NUM_THREADS=4
mkdir -p "$TMPDIR" /projects/Zeroshot/logs /projects/Zeroshot/runs
cd /projects/Zeroshot/bridge_scheduler_compare_v1
sha256sum -c SHA256SUMS
(cd /projects/Zeroshot/bridge_expansion_training_v1 && sha256sum -c SHA256SUMS)
# Check the reference dependency before spending GPU time on tensor collection.
LD_LIBRARY_PATH="/projects/Zeroshot/envs/official-pusht-v1/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  /projects/Zeroshot/envs/official-pusht-v1/bin/python - <<'PY'
import diffusers,torch
from diffusers import DDPMScheduler
assert diffusers.__version__=='0.11.1'
s=DDPMScheduler(num_train_timesteps=100,prediction_type='sample',clip_sample=False)
assert torch.isfinite(s.step(torch.zeros(1,16,7),0,torch.ones(1,16,7)).prev_sample).all()
print('Existing official scheduler import and CPU step: PASSED',flush=True)
PY
OUT="/projects/Zeroshot/runs/bridge-scheduler-${SLURM_JOB_ID}"
mkdir "$OUT"
/projects/Zeroshot/envs/multitask-preflight-v1/bin/python -u diagnose_multiscale_noise.py \
  --pack /projects/Zeroshot/bridge_expansion_training_v1 \
  --training-run /projects/Zeroshot/runs/bridge-unet-train-193761 \
  --training-report /projects/Zeroshot/runs/bridge-unet-train-193761/report.json \
  --window-reference /projects/Zeroshot/runs/bridge-full-chain-191004.json \
  --scheduler-export --output "$OUT/export.json"
# Reuse the already working official environment; no packages installed.
LD_LIBRARY_PATH="/projects/Zeroshot/envs/official-pusht-v1/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  /projects/Zeroshot/envs/official-pusht-v1/bin/python -u compare_official_ddpm.py \
  --export "$OUT/export.json" --output "$OUT/report.json"

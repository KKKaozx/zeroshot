#!/bin/bash
#SBATCH --job-name=bridge-ema-train
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40960M
#SBATCH --time=01:30:00
#SBATCH --chdir=/projects/Zeroshot
#SBATCH --output=/projects/Zeroshot/logs/bridge-ema-train-%j.out

set -eo pipefail
source /projects/Zeroshot/scripts/ssd_cache.sh
export TMPDIR="/projects/Zeroshot/.tmp/${SLURM_JOB_ID}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 USE_TF=0 OMP_NUM_THREADS=4
mkdir -p "$TMPDIR" /projects/Zeroshot/logs /projects/Zeroshot/runs
cd /projects/Zeroshot/bridge_ema_pair_v1
sha256sum -c SHA256SUMS
(cd /projects/Zeroshot/bridge_expansion_training_v1 && sha256sum -c SHA256SUMS)

PYTHON=/projects/Zeroshot/envs/multitask-preflight-v1/bin/python
"$PYTHON" - <<'PY'
import subprocess
used = int(subprocess.check_output(['du', '-sb', '/projects/Zeroshot/']).split()[0])
print('Project apparent bytes:', used, flush=True)
if used + 3 * 2**30 > 200_000_000_000:
    raise SystemExit('Stop: less than 3 GiB planning headroom under reported 200 GB allocation')
print('Planning storage check only; this does not query live quota.', flush=True)
PY
"$PYTHON" -u run_bridge_official_unet_training.py \
  --pack /projects/Zeroshot/bridge_expansion_training_v1 \
  --head-report /projects/Zeroshot/runs/bridge-head-train-192651/report.json \
  --window-reference /projects/Zeroshot/runs/bridge-full-chain-191004.json \
  --preflight /projects/Zeroshot/runs/bridge-unet-real-193711.json \
  --baseline-report /projects/Zeroshot/runs/bridge-unet-train-193761/report.json \
  --ema --output "/projects/Zeroshot/runs/bridge-ema-train-${SLURM_JOB_ID}"

#!/bin/bash
#SBATCH --job-name=bridge-dino-only
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=22528M
#SBATCH --time=03:00:00
#SBATCH --chdir=/projects/Zeroshot
#SBATCH --output=/projects/Zeroshot/logs/bridge-dino-only-%j.out
set -euo pipefail
ROOT=/projects/Zeroshot
BUNDLE="$ROOT/bridge_dino_only_v1"
PY="$ROOT/envs/multitask-preflight-v1/bin/python"
source "$ROOT/scripts/ssd_cache.sh"
export TMPDIR="$ROOT/.tmp/${SLURM_JOB_ID}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 USE_TF=0 OMP_NUM_THREADS=4
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
mkdir -p "$TMPDIR" "$ROOT/runs"
cd "$BUNDLE"
sha256sum -c SHA256SUMS
nvidia-smi
STAGE="${BRIDGE_DINO_STAGE:?Required preflight or fit}"
OUT="$ROOT/runs/bridge-dino-only-${STAGE}-${SLURM_JOB_ID}"
if [[ "$STAGE" == preflight ]]; then
  "$PY" -B -u run.py --root "$ROOT" --stage preflight --output "$OUT"
elif [[ "$STAGE" == fit ]]; then
  PREFLIGHT="${BRIDGE_DINO_PREFLIGHT:?Successful new preflight ID required}"
  [[ "$PREFLIGHT" =~ ^[0-9]+$ ]] || exit 2
  "$PY" -B -u run.py --root "$ROOT" --stage fit --output "$OUT"     --preflight "$ROOT/runs/bridge-dino-only-preflight-${PREFLIGHT}"
else
  echo "Unknown stage"; exit 2
fi

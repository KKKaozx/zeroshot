#!/bin/bash
#SBATCH --job-name=oxe-id-cpu
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --partition=cluster02
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:10:00
#SBATCH --chdir=/projects/Zeroshot
#SBATCH --output=/projects/Zeroshot/logs/oxe-id-cpu-%j.out
set -euo pipefail
ROOT=/projects/Zeroshot
PROJECT="$ROOT/envs/multitask-preflight-v1"
cd "$ROOT/oxe_frozen_identity_diag_v1"
export CUDA_VISIBLE_DEVICES=-1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
sha256sum -c SHA256SUMS
LD_LIBRARY_PATH="$PROJECT/lib" "$PROJECT/bin/python" -B run_diagnosis.py --root "$ROOT" --output "$ROOT/runs/oxe-id-cpu-${SLURM_JOB_ID}"

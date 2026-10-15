#!/bin/bash
#SBATCH --job-name=bridge-xyz-train
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40960M
#SBATCH --time=03:00:00
#SBATCH --chdir=/projects/Zeroshot
#SBATCH --output=/projects/Zeroshot/logs/bridge-xyz-train-%j.out
set -eo pipefail
source /projects/Zeroshot/scripts/ssd_cache.sh
export TMPDIR="/projects/Zeroshot/.tmp/${SLURM_JOB_ID}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 USE_TF=0 OMP_NUM_THREADS=4
mkdir -p "$TMPDIR" /projects/Zeroshot/logs /projects/Zeroshot/runs
cd /projects/Zeroshot/bridge_xyz_normalization_v1
sha256sum -c SHA256SUMS
(cd /projects/Zeroshot/bridge_expansion_training_v1 && sha256sum -c SHA256SUMS)
: "${XYZ_PREFLIGHT:?Set XYZ_PREFLIGHT to the passed paired preflight job ID}"
/projects/Zeroshot/envs/multitask-preflight-v1/bin/python -u run_bridge_xyz_pair.py --phase train --output "/projects/Zeroshot/runs/bridge-xyz-train-${SLURM_JOB_ID}" --preflight-report "/projects/Zeroshot/runs/bridge-xyz-preflight-${XYZ_PREFLIGHT}/report.json"

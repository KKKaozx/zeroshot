#!/bin/bash
#SBATCH --job-name=octo-preflight
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --chdir=/projects/Zeroshot
#SBATCH --output=/projects/Zeroshot/logs/octo-preflight-%j.out

set -eo pipefail
source /projects/Zeroshot/scripts/ssd_cache.sh
export TMPDIR="/projects/Zeroshot/.tmp/${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
module load Miniforge3
eval "$(conda shell.bash hook)"
conda activate /projects/Zeroshot/envs/zeroshot
python /projects/Zeroshot/octo_minimal_preflight/octo_preflight.py \
    --plan /projects/Zeroshot/octo_minimal_preflight/octo_minimal_reproduction.json \
    --project /projects/Zeroshot \
    --output "/projects/Zeroshot/baseline_setup/octo-preflight-${SLURM_JOB_ID}.json"

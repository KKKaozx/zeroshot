#!/bin/bash
#SBATCH --job-name=bridge-author-reference
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=22528M
#SBATCH --time=03:00:00
#SBATCH --chdir=/projects/Zeroshot
#SBATCH --output=/projects/Zeroshot/logs/bridge-author-reference-%j.out
set -eo pipefail
source /projects/Zeroshot/scripts/ssd_cache.sh
export TMPDIR="/projects/Zeroshot/.tmp/author-reference-${SLURM_JOB_ID}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export OMP_NUM_THREADS=4 SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy
export MPLCONFIGDIR="/projects/Zeroshot/.tmp/author-reference-mpl"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" /projects/Zeroshot/runs
PROJECT_BYTES=$(du -sb /projects/_ssd/Zeroshot | cut -f1)
if (( PROJECT_BYTES + 4*1024*1024*1024 + 15*1024*1024*1024 > 200000000000 )); then exit 1; fi
SOURCE=/projects/Zeroshot/paper_reproduction/official_pusht_v1/source
AUTHOR_ENV=/projects/Zeroshot/envs/official-pusht-v1
test "$(git -C "$SOURCE" rev-parse HEAD)" = 5ba07ac6661db573af695b419a7947ecb704690f
test -z "$(git -C "$SOURCE" status --porcelain)"
cd /projects/Zeroshot/bridge_author_reference_v1
sha256sum -c SHA256SUMS
nvidia-smi
LD_LIBRARY_PATH="$AUTHOR_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" "$AUTHOR_ENV/bin/python" -u run_bridge_author_reference.py \
  --project /projects/Zeroshot --source "$SOURCE" --output "/projects/Zeroshot/runs/bridge-author-reference-${SLURM_JOB_ID}"

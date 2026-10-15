#!/bin/bash
#SBATCH --job-name=official-pusht-eval
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --chdir=/projects/Zeroshot
#SBATCH --output=/projects/Zeroshot/logs/official-pusht-eval-%j.out
set -eo pipefail
TASK=/projects/Zeroshot/paper_reproduction/official_pusht_v1
ENV_PATH=/projects/Zeroshot/envs/official-pusht-v1
export LD_LIBRARY_PATH="$ENV_PATH/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
OUT="/projects/Zeroshot/runs/official-pusht-eval-${SLURM_JOB_ID}"
source /projects/Zeroshot/scripts/ssd_cache.sh
export TMPDIR="/projects/Zeroshot/.tmp/pusht-eval-${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
export SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy
export WANDB_MODE=offline WANDB_DIR="$TASK/wandb" WANDB_CACHE_DIR="$TASK/wandb-cache"
export MPLCONFIGDIR="$TASK/matplotlib" XDG_CACHE_HOME="$TASK/cache"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export PYTHONUNBUFFERED=1
test -f "$TASK/environment-installed"
test "$(git -C "$TASK/source" rev-parse HEAD)" = 5ba07ac6661db573af695b419a7947ecb704690f
test -z "$(git -C "$TASK/source" status --porcelain)"
sha256sum -c "$TASK/checkpoint.sha256"
PROJECT_BYTES=$(du -sb /projects/_ssd/Zeroshot | cut -f1)
if (( PROJECT_BYTES + 2*1024*1024*1024 > 200000000000 )); then
  echo 'Insufficient planning headroom for evaluation output.'; exit 1
fi
nvidia-smi
"$ENV_PATH/bin/python" -c "import torch; assert torch.cuda.is_available(); print('Torch:', torch.__version__, 'GPU:', torch.cuda.get_device_name(0))"
cd "$TASK/source"
# Author entry point, checkpoint configuration, EMA policy and runner unchanged.
"$ENV_PATH/bin/python" eval.py --checkpoint "$TASK/pusht_image.ckpt" --output_dir "$OUT" --device cuda:0
"$ENV_PATH/bin/python" - "$OUT/eval_log.json" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
for key in ('train/mean_score', 'test/mean_score'):
    assert key in data, f'Missing evaluation metric: {key}'
    print(key, data[key])
print('REPORT', sys.argv[1])
print('These are Push-T overlap scores, not robot grasp success rates.')
PY
cp "$TASK/source-commit.txt" "$OUT/source-commit.txt"
cp "$TASK/checkpoint.sha256" "$OUT/checkpoint.sha256"
echo 'OFFICIAL PUSHT EVALUATION COMPLETE: inspect scores and rollout videos.'

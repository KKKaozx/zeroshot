#!/bin/bash
#SBATCH --job-name=official-pusht-setup
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --partition=cluster02
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --chdir=/projects/Zeroshot
#SBATCH --output=/projects/Zeroshot/logs/official-pusht-setup-%j.out
set -eo pipefail
TASK=/projects/Zeroshot/paper_reproduction/official_pusht_v1
ENV_PATH=/projects/Zeroshot/envs/official-pusht-v1
export LD_LIBRARY_PATH="$ENV_PATH/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
BUNDLE=/projects/Zeroshot/official_pusht_v1
COMMIT=5ba07ac6661db573af695b419a7947ecb704690f
mkdir -p "$TASK" /projects/Zeroshot/baseline_setup
source /projects/Zeroshot/scripts/ssd_cache.sh
export TMPDIR="/projects/Zeroshot/.tmp/pusht-setup-${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
export WANDB_DIR="$TASK/wandb"
export WANDB_CACHE_DIR="$TASK/wandb-cache"
export WANDB_MODE=offline
export MPLCONFIGDIR="$TASK/matplotlib"
export XDG_CACHE_HOME="$TASK/cache"
module load Miniforge3
eval "$(conda shell.bash hook)"
BEFORE_BYTES=$(du -sb /projects/_ssd/Zeroshot | cut -f1)
# 200 GB is the user's reported project allocation, not a live quota query.
if (( BEFORE_BYTES + 20*1024*1024*1024 > 200000000000 )); then
  echo 'Insufficient planning headroom for 20 GiB; stop without installing.'; exit 1
fi
if [ ! -d "$TASK/source/.git" ]; then
  git init "$TASK/source"
  git -C "$TASK/source" remote add origin https://github.com/real-stanford/diffusion_policy.git
  git -C "$TASK/source" fetch --depth=1 origin "$COMMIT"
  git -C "$TASK/source" checkout --detach FETCH_HEAD
fi
test "$(git -C "$TASK/source" rev-parse HEAD)" = "$COMMIT"
test -z "$(git -C "$TASK/source" status --porcelain)"
if [ ! -f "$TASK/environment-installed" ]; then
  if [ -d "$ENV_PATH" ]; then
    conda env update --prefix "$ENV_PATH" --file "$BUNDLE/environment.yaml"
  else
    conda env create --prefix "$ENV_PATH" --file "$BUNDLE/environment.yaml"
  fi
  "$ENV_PATH/bin/python" -m pip install --no-deps -e "$TASK/source"
  "$ENV_PATH/bin/python" -m pip check
  touch "$TASK/environment-installed"
fi
export SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy
cd "$TASK/source"
"$ENV_PATH/bin/python" - <<'PY'
import torch, diffusers, gym, robomimic
from diffusion_policy.workspace.train_diffusion_unet_hybrid_workspace import TrainDiffusionUnetHybridWorkspace
from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv
env = PushTImageEnv()
obs = env.reset()
assert obs['image'].shape == (3, 96, 96)
env.close()
print('IMPORT AND CPU ENVIRONMENT RESET: OK', torch.__version__, diffusers.__version__, gym.__version__, flush=True)
PY
CURRENT_BYTES=$(du -sb /projects/_ssd/Zeroshot | cut -f1)
if (( CURRENT_BYTES + 4293295813 > 200000000000 || CURRENT_BYTES - BEFORE_BYTES + 4293295813 > 20*1024*1024*1024 )); then
  echo 'Planning storage limit reached; stop before checkpoint download.'; exit 1
fi
CKPT="$TASK/pusht_image.ckpt"
if [ ! -f "$CKPT" ] || [ "$(stat -c %s "$CKPT")" != 4293295813 ]; then
  env -u LD_LIBRARY_PATH /usr/bin/curl --http1.1 --fail --location --retry 3 --continue-at - \
    'https://diffusion-policy.cs.columbia.edu/data/experiments/image/pusht/diffusion_policy_cnn/train_0/checkpoints/epoch%3D0500-test_mean_score%3D0.884.ckpt' \
    --output "$CKPT.part"
  test "$(stat -c %s "$CKPT.part")" = 4293295813
  mv "$CKPT.part" "$CKPT"
fi
sha256sum "$CKPT" > "$TASK/checkpoint.sha256"
cp "$BUNDLE/environment.yaml" "$TASK/environment-requested.yaml"
"$ENV_PATH/bin/python" -m pip freeze > "$TASK/pip-freeze.txt"
conda list --prefix "$ENV_PATH" --explicit > "$TASK/conda-explicit.txt"
git -C "$TASK/source" rev-parse HEAD > "$TASK/source-commit.txt"
du -sh "$TASK" "$ENV_PATH" /projects/_ssd/Zeroshot
echo 'OFFICIAL PUSHT SETUP COMPLETE: evaluation has not run.'

#!/bin/bash
#SBATCH --job-name=octo-setup
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --partition=cluster02
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --chdir=/projects/Zeroshot
#SBATCH --output=/projects/Zeroshot/logs/octo-setup-%j.out
set -eo pipefail
source /projects/Zeroshot/scripts/ssd_cache.sh
export TMPDIR="/projects/Zeroshot/.tmp/${SLURM_JOB_ID}"
mkdir -p "$TMPDIR" /projects/Zeroshot/baseline_setup
module load Miniforge3
eval "$(conda shell.bash hook)"
OCTO_ENV=/projects/Zeroshot/envs/octo-small-v1
OCTO_SOURCE=/projects/Zeroshot/baselines/octo-official-v1
OCTO_BUNDLE=/projects/Zeroshot/octo_official_smoke_v1
PROJECT_BYTES=$(du -sb /projects/_ssd/Zeroshot | cut -f1)
if (( PROJECT_BYTES + 15*1024*1024*1024 > 200000000000 )); then
    echo 'Insufficient planning headroom within the reported 200 GB quota'; exit 1
fi
if [ ! -x "$OCTO_ENV/bin/python" ]; then
    conda create --prefix "$OCTO_ENV" python=3.10 pip -y
fi
if [ ! -d "$OCTO_SOURCE" ]; then
    mkdir -p /projects/Zeroshot/baselines
    git clone https://github.com/octo-models/octo.git "$OCTO_SOURCE"
fi
git -C "$OCTO_SOURCE" checkout --detach 241fb3514b7c40957a86d869fecb7c7fc353f540
test "$(git -C "$OCTO_SOURCE" rev-parse HEAD)" = 241fb3514b7c40957a86d869fecb7c7fc353f540
export PIP_CONSTRAINT="$OCTO_BUNDLE/octo_constraints.txt"
"$OCTO_ENV/bin/python" -m pip install -r "$OCTO_SOURCE/requirements.txt" \
    'jax[cuda11_pip]==0.4.20' sentencepiece==0.1.99 \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
"$OCTO_ENV/bin/python" -m pip install --no-deps -e "$OCTO_SOURCE"
"$OCTO_ENV/bin/python" -m pip check
"$OCTO_ENV/bin/python" -m pip freeze > "/projects/Zeroshot/baseline_setup/octo-packages-${SLURM_JOB_ID}.txt"
"$OCTO_ENV/bin/python" "$OCTO_BUNDLE/octo_official_smoke.py" --prepare \
    --plan "$OCTO_BUNDLE/octo_minimal_reproduction.json" \
    --source "$OCTO_SOURCE" --assets /projects/Zeroshot/baseline_setup/octo-assets-v1 \
    --output "/projects/Zeroshot/baseline_setup/octo-setup-${SLURM_JOB_ID}.json"
AFTER_BYTES=$(du -sb /projects/_ssd/Zeroshot | cut -f1)
if (( AFTER_BYTES - PROJECT_BYTES > 15*1024*1024*1024 )); then
    echo 'New project usage exceeds planned 15 GiB; stop before inference'; exit 1
fi
du -sh "$OCTO_ENV" /projects/_ssd/Zeroshot
echo 'OCTO SETUP AND PINNED ASSETS: PASSED; GPU inference is a separate job'

#!/bin/bash
#SBATCH --job-name=pusht-gl-repair
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --partition=cluster02
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --chdir=/projects/Zeroshot
#SBATCH --output=/projects/Zeroshot/logs/official-pusht-setup-%j.out
set -eo pipefail
source /projects/Zeroshot/scripts/ssd_cache.sh
export TMPDIR="/projects/Zeroshot/.tmp/pusht-gl-${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
module load Miniforge3
ENV_PATH=/projects/Zeroshot/envs/official-pusht-v1
# Explicit package URLs bypass dependency solving. Check the external X11
# dependency before installing the three matching libglvnd builds.
"$ENV_PATH/bin/python" - "$ENV_PATH" <<'PY'
import json, pathlib, sys
rows = [json.loads(p.read_text()) for p in pathlib.Path(sys.argv[1], 'conda-meta').glob('xorg-libx11-*.json')]
assert len(rows) == 1, 'Stop: xorg-libx11 metadata missing; no libraries installed'
version = tuple(int(x) for x in rows[0]['version'].split('.'))
assert (1, 8, 9) <= version < (2,), f'Stop: incompatible xorg-libx11 {version}; no libraries installed'
print('Existing X11 dependency:', rows[0]['version'])
PY
conda install --prefix "$ENV_PATH" -y \
  https://conda.anaconda.org/conda-forge/linux-64/libglvnd-1.7.0-ha4b6fd6_0.conda \
  https://conda.anaconda.org/conda-forge/linux-64/libglx-1.7.0-ha4b6fd6_0.conda \
  https://conda.anaconda.org/conda-forge/linux-64/libgl-1.7.0-ha4b6fd6_0.conda
export LD_LIBRARY_PATH="$ENV_PATH/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
"$ENV_PATH/bin/python" -c "import cv2; print('OpenCV import:', cv2.__version__, cv2.__file__)"
bash /projects/Zeroshot/official_pusht_v1/setup.sh

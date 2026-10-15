#!/bin/bash
#SBATCH --job-name=pusht-cv-repair
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
export TMPDIR="/projects/Zeroshot/.tmp/pusht-cv-${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
module load Miniforge3
ENV_PATH=/projects/Zeroshot/envs/official-pusht-v1
mapfile -t CV_PACKAGES < <("$ENV_PATH/bin/python" - "$ENV_PATH" <<'PY'
import json, pathlib, sys
for p in pathlib.Path(sys.argv[1], 'conda-meta').glob('*.json'):
    row = json.loads(p.read_text())
    if row['name'] in ('opencv', 'py-opencv', 'libopencv'):
        print(row['name'])
PY
)
conda list --prefix "$ENV_PATH" --explicit > "$TMPDIR/before-opencv-repair.txt"
if ((${#CV_PACKAGES[@]})); then
  # Force-remove only these explicitly selected OpenCV packages; bypass solver.
  conda remove --prefix "$ENV_PATH" --force -y "${CV_PACKAGES[@]}"
fi
"$ENV_PATH/bin/python" -m pip install --no-deps --only-binary=:all: opencv-python-headless==4.6.0.66
"$ENV_PATH/bin/python" -c "import cv2; print('OpenCV:', cv2.__version__, cv2.__file__); assert any(line.strip().startswith('GUI:') and line.split(':', 1)[1].strip() == 'NONE' for line in cv2.getBuildInformation().splitlines()), 'Expected headless OpenCV'"
"$ENV_PATH/bin/python" -m pip check
bash /projects/Zeroshot/official_pusht_v1/setup.sh

#!/bin/bash
#SBATCH --job-name=diffusion-setup
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --partition=cluster02
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --chdir=/projects/Zeroshot
#SBATCH --output=/projects/Zeroshot/logs/diffusion-setup-%j.out
set -eo pipefail
source /projects/Zeroshot/scripts/ssd_cache.sh
export TMPDIR="/projects/Zeroshot/.tmp/$SLURM_JOB_ID"
mkdir -p "$TMPDIR" /projects/Zeroshot/baseline_setup
module load Miniforge3
eval "$(conda shell.bash hook)"
ENV_PATH=/projects/Zeroshot/envs/bridge-diffusion
if [ ! -x "$ENV_PATH/bin/python" ]; then
  conda create --prefix "$ENV_PATH" python=3.10 pip -y
fi
"$ENV_PATH/bin/python" -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu118
"$ENV_PATH/bin/python" -m pip install numpy==1.26.4
"$ENV_PATH/bin/python" -m pip check
sha256sum -c bridge_diffusion_fit_v1.sha256
test ! -e /projects/Zeroshot/baseline_setup/bridge_diffusion_fit_v1
"$ENV_PATH/bin/python" -m zipfile -e bridge_diffusion_fit_v1.zip /projects/Zeroshot/baseline_setup
cd /projects/Zeroshot/baseline_setup/bridge_diffusion_fit_v1
"$ENV_PATH/bin/python" - <<'PY'
import hashlib, json
from pathlib import Path
import torch, numpy
root = Path.cwd()
for name, expected in json.loads((root/'integrity.json').read_text()).items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest() == expected, name
from cached_diffusion import RobotAdapterModel
bundle = torch.load('fixed_context.pt', map_location='cpu', weights_only=False)
model = RobotAdapterModel(bundle['config'])
model.load_state_dict(bundle['gripper_state_dict'], strict=False)
assert bundle['targets'].shape == (14,16,8)
print('Torch:', torch.__version__, 'CUDA runtime:', torch.version.cuda, 'NumPy:', numpy.__version__)
print('DIFFUSION ENVIRONMENT AND BUNDLE: PASSED')
PY

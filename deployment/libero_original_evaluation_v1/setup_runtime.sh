#!/usr/bin/env bash
set -euo pipefail
ROOT=/projects/Zeroshot
BASE="$ROOT/envs/multitask-preflight-v1/bin/python"
ENV="$ROOT/envs/libero-eval-v1"
cd "$ROOT/libero_original_evaluation_v1"
sha256sum -c SHA256SUMS
"$BASE" -c 'import torch; assert torch.__version__ == "2.5.1+cu118"; print("Training environment stays unchanged:", torch.__version__, torch.__file__)'
if [ ! -x "$ENV/bin/python" ]; then
  "$BASE" -m venv --system-site-packages "$ENV"
fi
# No torch/transformers install, no change to the active OXE training environment.
"$ENV/bin/python" -m pip install --no-cache-dir \
  numpy==1.26.4 mujoco==2.3.7 robosuite==1.4.0 gym==0.25.2 \
  bddl==1.0.1 easydict==1.9 scipy==1.13.1 numba==0.60.0 \
  matplotlib==3.8.4 opencv-python==4.6.0.66 cloudpickle==2.1.0 \
  PyOpenGL==3.1.10 glfw==2.11.0 h5py==3.14.0 PyYAML==6.0.2
"$ENV/bin/python" prepare_runtime.py --project-root "$ROOT"
"$BASE" -c 'import torch; assert torch.__version__ == "2.5.1+cu118"; print("Original training torch retained:", torch.__file__)'
echo 'LIBERO simulator files/environment prepared; no physics, model inference or task result yet.'

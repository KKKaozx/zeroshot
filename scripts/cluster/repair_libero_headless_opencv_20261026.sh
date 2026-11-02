#!/usr/bin/env bash
# Run on the cluster login node, before any completed evaluation episodes exist.
set -euo pipefail
ROOT=/projects/Zeroshot
PY="$ROOT/envs/libero-eval-v1/bin/python"
REMOVE_LOCAL_GUI=$("$PY" - <<'PY'
from pathlib import Path
import importlib.metadata as m
import sys
env=Path('/projects/Zeroshot/envs/libero-eval-v1').resolve()
assert Path(sys.prefix).resolve()==env, 'Only the isolated evaluation venv may be changed'
out=Path('/projects/Zeroshot/runs/libero-original-dino-noaug-s42')
assert not list((out/'episodes').glob('*.json')), 'Completed evaluations exist; keep their bound runtime intact'
try:
    d=m.distribution('opencv-python')
except m.PackageNotFoundError:
    remove=False
else:
    remove=Path(d.locate_file('')).resolve().is_relative_to(env)
print('yes' if remove else 'no')
PY
)
echo 'HEADLESS_FIX_SCOPE_VERIFIED: isolated evaluation venv, no completed episodes'
# OpenCV's four wheel flavours share cv2; remove the GUI wheel before replacing it.
if [ "$REMOVE_LOCAL_GUI" = yes ]; then
  "$PY" -m pip uninstall -y opencv-python
fi
"$PY" -m pip install --no-cache-dir --no-deps --only-binary=:all: opencv-python-headless==4.6.0.66
"$PY" - <<'PY'
from pathlib import Path
import importlib.metadata as m
import json
import ctypes.util
import re
import sys
import cv2
import numpy as np
import torch
assert m.version('opencv-python-headless')=='4.6.0.66'
assert cv2.__version__=='4.6.0' and torch.__version__=='2.5.1+cu118' and np.__version__=='1.26.4'
assert Path(cv2.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
assert re.search(r'GUI:\s+NONE',cv2.getBuildInformation()), 'Expected non-GUI OpenCV build'
assert cv2.cvtColor(np.zeros((2,2,3),dtype=np.uint8),cv2.COLOR_RGB2GRAY).shape==(2,2)
assert not torch.cuda.is_initialized()
out=Path('/projects/Zeroshot/runs/libero-headless-dependency-repair')
out.mkdir(parents=True,exist_ok=True)
r={'passed':True,'opencv_headless':m.version('opencv-python-headless'),'cv2':cv2.__version__,
   'torch':torch.__version__,'numpy':np.__version__,'cv2_path':cv2.__file__,
   'opencv_gui':'NONE','physics_steps':0,'model_forwards':0,'new_jobs_submitted':0,
   'scope':'Login-node headless OpenCV dependency check; native EGL must run on the allocated GPU node'}
r['dependency_versions']={n:m.version(n) for n in ('termcolor','mujoco','robosuite','gym','bddl',
    'easydict','h5py','scipy','numba','PyOpenGL','glfw','transformers','Pillow','PyYAML','matplotlib')}
r['login_node_library_resolution']={n:ctypes.util.find_library(n) for n in ('GL','EGL','OpenGL')}
(out/'report.json').write_text(json.dumps(r,indent=2)+'\n')
print('HEADLESS_OPENCV_IMPORT_VERIFIED',json.dumps(r))
PY
# Existing job body imports native EGL on the actual allocated GPU node. The first
# complete episode still counts toward4000; no extra GPU preflight is submitted.
cd "$ROOT/libero_original_evaluation_v1"
bash submit_once.sh 1

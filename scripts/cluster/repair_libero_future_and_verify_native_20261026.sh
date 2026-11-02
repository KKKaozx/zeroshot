#!/usr/bin/env bash
# Complete the pinned official LIBERO implicit dependency, then test the actual
# native environment import chain before another allocated-GPU episode.
set -euo pipefail
ROOT=/projects/Zeroshot
PY="$ROOT/envs/libero-eval-v1/bin/python"
"$PY" - <<'PY'
from pathlib import Path
import sys
assert Path(sys.prefix).resolve()==Path('/projects/Zeroshot/envs/libero-eval-v1').resolve()
assert not list(Path('/projects/Zeroshot/runs/libero-original-dino-noaug-s42/episodes').glob('*.json'))
print('Verified isolated evaluation environment and no completed episodes')
PY
"$PY" -m pip install --no-cache-dir --no-deps future==0.18.2
"$PY" - <<'PY'
import importlib.metadata as m
from future.utils import with_metaclass
from bddl.backend_abc import BDDLBackend
from bddl.parsing import scan_tokens
import cloudpickle
assert m.version('future')=='0.18.2'
assert m.version('cloudpickle')=='2.1.0'
print('BDDL actual backend/parser imports passed; future0.18.2, cloudpickle2.1.0')
PY
# This existing helper imports OffScreenRenderEnv and every module imported by
# libero.libero.envs itself. It creates no environment/model/EGL context.
"$PY" -B "$ROOT/oxe_resume_and_libero_dependency_repair_v1/verify_eval_import.py" --root "$ROOT"
cd "$ROOT/libero_original_evaluation_v1"
bash submit_once.sh 1

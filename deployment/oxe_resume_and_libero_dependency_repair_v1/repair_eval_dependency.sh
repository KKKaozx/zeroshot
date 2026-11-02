#!/usr/bin/env bash
set -euo pipefail
ROOT=/projects/Zeroshot
REPAIR="$ROOT/oxe_resume_and_libero_dependency_repair_v1"
PY="$ROOT/envs/libero-eval-v1/bin/python"
cd "$REPAIR"
sha256sum -c SHA256SUMS
if find "$ROOT/runs/libero-original-dino-noaug-s42/episodes" -name '*.json' -print -quit 2>/dev/null | grep -q .; then
  echo 'Completed evaluation records exist; do not change that bound environment'; exit 1
fi
# Robosuite1.4.0 imports termcolor but omits it from its declared dependencies.
# Only the isolated evaluation venv receives this missing pure-Python package.
"$PY" -m pip install --no-cache-dir --no-deps termcolor==3.3.0
"$PY" verify_eval_import.py --root "$ROOT"
echo 'Missing logger dependency repaired; no model/physics/episode executed. Existing evaluation package remains unchanged.'

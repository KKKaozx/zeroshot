#!/bin/bash
set -euo pipefail
ROOT=/projects/Zeroshot
BUNDLE="$ROOT/pusht_dino_cache_v1"
OFFICIAL="$ROOT/envs/official-pusht-v1"
cd "$BUNDLE"
sha256sum -c SHA256SUMS
LD_LIBRARY_PATH="$OFFICIAL/lib" "$OFFICIAL/bin/python" -B run_cache.py space --root "$ROOT"
test -s "$ROOT/runs/pusht-dino-resource-220882/tokens.npz"
test -s "$ROOT/runs/pusht-project-cache-213439/text.npz"
test -s "$ROOT/runs/pusht-project-cache-213439/frame_ids.npy"
test -s "$ROOT/dinov2_large_frozen_assets_20261023/model.safetensors"
if ! mkdir .submission; then echo 'Already submitted; inspect saved job before retrying.' >&2; exit 1; fi
if JOB=$(sbatch --parsable cache.sh); then
  printf '%s\n' "$JOB" > .submission/job_id
  echo "Push-T DINO训练缓存与在线接线：$JOB"
else
  echo 'Submission failed; guard retained for inspection.' >&2
  exit 1
fi

#!/bin/bash
set -euo pipefail
ROOT=/projects/Zeroshot
BUNDLE="$ROOT/pusht_dino_resource_v1"
ASSETS="$ROOT/dinov2_large_frozen_assets_20261023"
cd "$BUNDLE"
sha256sum -c SHA256SUMS
test -s "$ASSETS/config.json"
test -s "$ASSETS/preprocessor_config.json"
printf '%s  %s
' 399fba97a95f22c36834418bc69373364a99af3a1153da1c0fb31db567c92e23 "$ASSETS/model.safetensors" | sha256sum -c -
test -s "$ROOT/runs/pusht-front-resource-210763/inputs.npz"
test -s "$ROOT/runs/pusht-project-cache-213439/text.npz"
if ! mkdir .submission; then echo 'Already submitted; inspect the saved job before retrying.' >&2; exit 1; fi
if JOB=$(sbatch --parsable preflight.sh); then
  printf '%s
' "$JOB" > .submission/job_id
  echo "Push-T DINOv2资源预检：$JOB"
else
  echo 'Submission failed; guard retained for inspection.' >&2
  exit 1
fi

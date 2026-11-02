#!/bin/bash
set -euo pipefail
CODE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$CODE"
exec 9>submit_formal.lock
flock -x 9
if [[ -s submitted_formal.job ]]; then
  printf 'Existing spatial formal job: %s\n' "$(cat submitted_formal.job)"
  exit 0
fi
sha256sum -c SHA256SUMS
/projects/Zeroshot/envs/multitask-preflight-v1/bin/python /projects/Zeroshot/bridge_spatial_readout_v1/verify_preflight.py /projects/Zeroshot/runs/bridge-spatial-preflight-210291
SPATIAL_REPLY=$(sbatch --parsable "$CODE/formal.sh")
SPATIAL_NUMBER="${SPATIAL_REPLY%%;*}"
[[ "$SPATIAL_NUMBER" =~ ^[0-9]+$ ]]
printf '%s\n' "$SPATIAL_NUMBER" > submitted_formal.job
printf 'SPATIAL_FORMAL: %s\n' "$SPATIAL_NUMBER"

#!/bin/bash
set -euo pipefail
CODE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$CODE"
exec 9>submit_preflight.lock
flock -x 9
if [[ -s submitted_preflight.job ]]; then
  printf 'Existing spatial preflight: %s\n' "$(cat submitted_preflight.job)"
  exit 0
fi
sha256sum -c SHA256SUMS
SPATIAL_REPLY=$(sbatch --parsable "$CODE/preflight.sh")
SPATIAL_NUMBER="${SPATIAL_REPLY%%;*}"
[[ "$SPATIAL_NUMBER" =~ ^[0-9]+$ ]]
printf '%s\n' "$SPATIAL_NUMBER" > submitted_preflight.job
printf 'SPATIAL_PREFLIGHT: %s\n' "$SPATIAL_NUMBER"

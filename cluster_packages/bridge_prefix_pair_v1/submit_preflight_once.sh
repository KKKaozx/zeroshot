#!/bin/bash
set -euo pipefail
CODE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$CODE"
exec 9>submit_preflight.lock
flock -x 9
if [[ -s submitted_preflight.job ]]; then
  printf 'Existing preflight: %s\n' "$(cat submitted_preflight.job)"
  exit 0
fi
sha256sum -c SHA256SUMS
JOB_REPLY=$(sbatch --parsable "$CODE/preflight.sh")
JOB_NUMBER="${JOB_REPLY%%;*}"
printf '%s\n' "$JOB_NUMBER" > submitted_preflight.job
printf 'PREFIX_PREFLIGHT: %s\n' "$JOB_NUMBER"

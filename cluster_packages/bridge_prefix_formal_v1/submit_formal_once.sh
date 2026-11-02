#!/bin/bash
set -euo pipefail
CODE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$CODE"
exec 9>submit_formal.lock
flock -x 9
if [[ -s submitted_formal.job ]]; then
  printf 'Existing formal pair: %s\n' "$(cat submitted_formal.job)"
  exit 0
fi
sha256sum -c SHA256SUMS
/projects/Zeroshot/envs/multitask-preflight-v1/bin/python \
  /projects/Zeroshot/bridge_prefix_pair_v1/verify_prefix_preflight.py \
  /projects/Zeroshot/runs/bridge-prefix-preflight-209420
JOB_REPLY=$(sbatch --parsable "$CODE/formal.sh")
JOB_NUMBER="${JOB_REPLY%%;*}"
printf '%s\n' "$JOB_NUMBER" > submitted_formal.job
printf 'PREFIX_FORMAL: %s\n' "$JOB_NUMBER"

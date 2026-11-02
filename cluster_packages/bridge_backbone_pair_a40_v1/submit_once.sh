#!/bin/bash
set -euo pipefail
ROOT=/projects/Zeroshot
BUNDLE="$ROOT/bridge_backbone_pair_a40_v1"
PY="$ROOT/envs/multitask-preflight-v1/bin/python"
cd "$BUNDLE"
mkdir -p jobs "$ROOT/logs"
exec 9>jobs/.submit.lock
flock -n 9 || { echo "Another submission is active"; exit 1; }
[[ ! -e jobs/pair.id ]] || { echo "Already submitted: $(cat jobs/pair.id); no duplicate"; exit 1; }
for OLD in 229016 229042; do
  ROW=$(sacct -X -n -P -j "$OLD" --format=JobIDRaw,State%32)
  STATE=$(awk -F'|' -v id="$OLD" '$1==id {print $2; exit}' <<<"$ROW")
  case "$STATE" in FAILED|NODE_FAIL) ;; *) echo "Previous attempt $OLD is '$STATE'; inspect before submitting"; exit 2 ;; esac
 done
sha256sum -c SHA256SUMS
"$PY" -B check_assets.py --root "$ROOT"
"$PY" -B - <<'PY'
from pathlib import Path
from admission import admit
admit(Path('/projects/Zeroshot/runs/bridge-dino-only-preflight-228978'))
print('SOURCE PREFLIGHT228978 BOUND; paired A40 runtime is checked in allocation; no new preflight updates')
PY
JOB=$(sbatch --parsable job.sh)
JOB="${JOB%%;*}"
[[ "$JOB" =~ ^[0-9]+$ ]] || exit 2
printf '%s\n' "$JOB" >jobs/pair.id
echo "BRIDGE A40 BACKBONE PAIR: $JOB"

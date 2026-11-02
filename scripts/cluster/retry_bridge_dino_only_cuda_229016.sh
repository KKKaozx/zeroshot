#!/bin/bash
# Operational retry only: unchanged bundle, CUDA failed before any update.
set -euo pipefail
ROOT=/projects/Zeroshot
BUNDLE="$ROOT/bridge_dino_only_v1"
PY="$ROOT/envs/multitask-preflight-v1/bin/python"
cd "$BUNDLE"
exec 9>jobs/.submit.lock
flock -n 9 || { echo "Another submission is active"; exit 1; }
[[ "$(cat jobs/fit.id)" == 229016 ]] || { echo "Formal marker changed; no duplicate submitted"; exit 1; }
[[ "$(cat jobs/preflight.id)" == 228978 ]] || { echo "Preflight marker differs"; exit 1; }

FAILED=$(sacct -X -n -P -j 229016 --format=JobIDRaw,State%32,NodeList)
PASSED=$(sacct -X -n -P -j 228978 --format=JobIDRaw,State%32,NodeList)
STATE=$(awk -F'|' '$1=="229016" {print $2; exit}' <<<"$FAILED")
BAD=$(awk -F'|' '$1=="229016" {print $3; exit}' <<<"$FAILED")
PSTATE=$(awk -F'|' '$1=="228978" {print $2; exit}' <<<"$PASSED")
GOOD=$(awk -F'|' '$1=="228978" {print $3; exit}' <<<"$PASSED")
case "$STATE" in FAILED|NODE_FAIL) ;; *) echo "229016 is '$STATE'; no retry submitted"; exit 2 ;; esac
[[ "$PSTATE" == COMPLETED ]] || { echo "228978 is '$PSTATE'; inspect accounting first"; exit 2; }
[[ "$BAD" =~ ^[A-Za-z0-9_.-]+$ && "$GOOD" =~ ^[A-Za-z0-9_.-]+$ && "$BAD" != "$GOOD" ]] || {
  echo "No distinct previously successful node confirmed; refusing to guess or change GPU model"; exit 2;
}
[[ ! -e "$ROOT/runs/bridge-dino-only-fit-229016" ]] || {
  echo "Old output exists: inspect before any retry; nothing deleted"; exit 2;
}
"$PY" -B - <<'PY'
from pathlib import Path
text=Path('/projects/Zeroshot/logs/bridge-dino-only-229016.out').read_text()
assert 'ValueError: Allocated working CUDA GPU required' in text
assert not any(x in text for x in ['COVERAGE_UPDATE ', 'DINO_ONLY_INPUT ', 'DINO_ONLY_ENCODE '])
print('229016 startup CUDA failure confirmed; no training/cache-encoding progress logged')
PY
sha256sum -c SHA256SUMS
"$PY" -B check_assets.py --root "$ROOT"
"$PY" -B verify.py "$ROOT/runs/bridge-dino-only-preflight-228978" --stage preflight
if [[ -e jobs/fit.failed-229016.id ]]; then
  [[ "$(cat jobs/fit.failed-229016.id)" == 229016 ]] || exit 2
else
  cp -p jobs/fit.id jobs/fit.failed-229016.id
fi
echo "Retry same A6000 recipe on successful preflight node $GOOD; exclude failed node $BAD"
JOB=$(sbatch --parsable --nodelist="$GOOD" --exclude="$BAD" \
  --export=ALL,BRIDGE_DINO_STAGE=fit,BRIDGE_DINO_PREFLIGHT=228978 job.sh)
JOB="${JOB%%;*}"
[[ "$JOB" =~ ^[0-9]+$ ]] || exit 2
printf '%s\n' "$JOB" >jobs/fit.id
printf 'failed_job=229016\npreflight_job=228978\nexcluded_node=%s\nrequested_node=%s\nretry_job=%s\n' \
  "$BAD" "$GOOD" "$JOB" >"jobs/fit.retry-${JOB}.txt"
echo "BRIDGE DINO-ONLY CUDA retry: $JOB"

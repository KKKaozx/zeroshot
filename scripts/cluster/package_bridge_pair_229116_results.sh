#!/bin/bash
# Collect existing small evidence only; do not include either model weight file.
set -euo pipefail
ROOT=/projects/Zeroshot
DIR="$ROOT/runs/bridge-backbone-pair-a40-229116"
PY="$ROOT/envs/multitask-preflight-v1/bin/python"
cd "$ROOT/bridge_backbone_pair_a40_v1"
"$PY" -B verify_pair.py "$DIR"
cd "$DIR"
ARCHIVE=bridge-backbone-pair-a40-229116-reports.tar.gz
if [[ -e "$ARCHIVE" ]]; then
  sha256sum -c "${ARCHIVE}.sha256"
  echo "Existing archive verified; no repackaging"
  exit 0
fi
sha256sum clip/final_trainable.pt dino/final_trainable.pt >weights.sha256
FILES=(pair_report.json pair_comparison.json weights.sha256)
for ARM in clip dino; do
  for NAME in report.json effective_runner.py entry_binding.json runner_snapshot.py \
    common_updates.jsonl reduced_updates.jsonl reduced_u8192_train.npz reduced_u8192_development.npz; do
    [[ -f "$ARM/$NAME" ]] || { echo "Missing $ARM/$NAME"; exit 2; }
    FILES+=("$ARM/$NAME")
  done
done
tar -czf "$ARCHIVE" "${FILES[@]}"
sha256sum "$ARCHIVE" >"${ARCHIVE}.sha256"
echo "PAIR_REPORTS_PACKAGED $DIR/$ARCHIVE"
du -h "$ARCHIVE"

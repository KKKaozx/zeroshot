#!/usr/bin/env bash
set -euo pipefail
ROOT=/projects/Zeroshot
OUT="$ROOT/runs/oxe-sampling-pair-233531"
BUNDLE="$ROOT/oxe_sampling_pair_v1"
ARCHIVE="$ROOT/oxe_sampling_results_233531.tar.gz"
FILES=(report.json predictions_and_targets.npz paired_zero_update_guard.json
  replacement/receipt.json cycle/receipt.json
  replacement/training_receipt.json cycle/training_receipt.json
  effective_source_loader.py effective_evaluation_inputs.py)
for name in "${FILES[@]}"; do
  [[ -f "$OUT/$name" ]] || { echo "Missing output: $name" >&2; exit 1; }
  [[ $(stat -c %s "$OUT/$name") -lt 10485760 ]] || { echo "Unexpected large output: $name" >&2; exit 1; }
done
for name in protocol.json plans.npz coverage_plan.json; do
  [[ -f "$BUNDLE/$name" ]] || exit 1
done
tar -czf "$ARCHIVE" -C "$OUT" "${FILES[@]}" -C "$BUNDLE" protocol.json plans.npz coverage_plan.json
sha256sum "$ARCHIVE"
du -h "$ARCHIVE"

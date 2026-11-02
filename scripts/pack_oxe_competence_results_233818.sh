#!/usr/bin/env bash
set -euo pipefail
ROOT=/projects/Zeroshot
OUT="$ROOT/runs/oxe-competence-233818"
FILES=(report.json predictions_and_targets.npz zero_update_guard.json moderate_pool/receipt.json)
for name in "${FILES[@]}"; do
  [[ -f "$OUT/$name" ]] || { echo "Missing output: $name" >&2; exit 1; }
  [[ $(stat -c %s "$OUT/$name") -lt 10485760 ]] || { echo "Unexpected large file: $name" >&2; exit 1; }
done
tar -czf "$ROOT/oxe_competence_results_233818.tar.gz" -C "$OUT" "${FILES[@]}"
sha256sum "$ROOT/oxe_competence_results_233818.tar.gz"
du -h "$ROOT/oxe_competence_results_233818.tar.gz"

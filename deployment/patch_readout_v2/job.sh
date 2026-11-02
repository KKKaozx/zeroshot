#!/usr/bin/env bash
#SBATCH --job-name=patch-readout-v2
#SBATCH --partition=cluster02
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:30:00
#SBATCH --no-requeue
#SBATCH --chdir=/projects/Zeroshot
#SBATCH --output=/projects/Zeroshot/logs/patch-readout-v2-%j.out
# 一次作业：建4版本增强缓存(若不存在) -> 依次训练B1-B3(各≤25分钟训练) -> 每个配置独立报告。
# B0 = v1作业239393的A3，不重训。
# 任何一步失败即停止；不自动重提、不加步、不续训。
set -euo pipefail
ROOT=/projects/Zeroshot
PROJECT="$ROOT/envs/multitask-preflight-v1"
PY="$PROJECT/bin/python"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 USE_TF=0 WANDB_MODE=disabled
export LD_LIBRARY_PATH="$PROJECT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "$ROOT/patch_readout_v2"
CACHE="$ROOT/caches/patch_readout_bridge_dinov2_aug4_v2"
RUNS="$ROOT/runs/patch-readout-v2-${SLURM_JOB_ID}"
mkdir -p "$ROOT/caches" "$RUNS"
if [[ ! -e "$CACHE/manifest.json" ]]; then
  "$PY" -B build_cache.py --root "$ROOT" --backbone dinov2 --views 4 --out "$CACHE"
fi
for CFG in B1_vision_only B2_vision_delta B3_vision_delta_aug; do
  "$PY" -B train_cached.py --cache "$CACHE" --config "configs/${CFG}.json" --seed 42 \
      --steps 10000 --batch 64 --time-cap 1500 --out "$RUNS/${CFG}_s42"
done
"$PY" -B summarize_runs.py "$RUNS" | tee "$RUNS/summary.txt"
cd "$RUNS" && tar czf small_results.tgz summary.txt */report.json */history.json */run_config.json
echo "PATCH_READOUT_V2_ALL_COMPLETE $RUNS"

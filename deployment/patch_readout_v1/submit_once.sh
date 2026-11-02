#!/usr/bin/env bash
# 审阅后手动执行一次。先做CPU检查(数值/路径，不加载编码器)，再唯一一次sbatch。
set -euo pipefail
ROOT=/projects/Zeroshot
PROJECT="$ROOT/envs/multitask-preflight-v1"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 USE_TF=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export LD_LIBRARY_PATH="$PROJECT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "$ROOT/patch_readout_v1"
[[ ! -e job_id.txt ]] || { echo "Already submitted: $(cat job_id.txt)"; exit 1; }
"$PROJECT/bin/python" -B build_cache.py --root "$ROOT" --check-only
echo SUBMITTING > job_id.txt
JOB=$(sbatch --parsable job.sh); JOB="${JOB%%;*}"
[[ "$JOB" =~ ^[0-9]+$ ]] || { echo 'Ambiguous submission; inspect squeue before any retry'; exit 1; }
echo "$JOB" > job_id.txt
echo "PATCH READOUT JOB: $JOB"

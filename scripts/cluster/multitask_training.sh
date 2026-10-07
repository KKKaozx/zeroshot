#!/bin/bash
#SBATCH --job-name=multitask-pilot
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=22528M
#SBATCH --time=02:00:00
#SBATCH --chdir=/projects/Zeroshot
#SBATCH --output=/projects/Zeroshot/logs/multitask-training-%j.out
set -eo pipefail
source /projects/Zeroshot/scripts/ssd_cache.sh
export TMPDIR="/projects/Zeroshot/.tmp/${SLURM_JOB_ID}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export USE_TF=0
export OMP_NUM_THREADS=4
mkdir -p "$TMPDIR"
PACK=/projects/Zeroshot/multitask_training_v1
PY=/projects/Zeroshot/envs/multitask-preflight-v1/bin/python
OUT="/projects/Zeroshot/runs/multitask-pilot-${SLURM_JOB_ID}"
cd "$PACK"
sha256sum -c SHA256SUMS
"$PY" -u train.py --dataset-dir "$PACK/data" --bridge-task-plan "$PACK/manifest.json" \
  --model-name /projects/Zeroshot/baseline_setup/clip-vit-large-patch14-v1 \
  --output-dir "$OUT" --sources tfrecord --exclude-path-parts '' --exclude-schemas '' \
  --min-trajectory-steps 17 --chunk-size 16 --stride 4 \
  --bridge-current-gripper continuous --bridge-gripper-policy reverse_scan_valid_steps_v2 \
  --gripper-target-mode state --gripper-head-type legacy --no-balanced-gripper-loss --gripper-change-weight 1 --no-balanced-sampling \
  --adapter-layers 8 --attention-dim 512 --adapter-pooling cls_patch_mean \
  --decoder-type diffusion --diffusion-prediction-type sample --diffusion-steps 100 \
  --beta-schedule squaredcos_cap_v2 --dropout 0.1 \
  --batch-size 2 --epochs 20 --max-steps-per-epoch 0 --learning-rate 0.0001 \
  --lr-schedule cosine --weight-decay 0.0001 --workers 0 --seed 42 \
  --bridge-validation-scope all_windows --max-validation-batches 0 \
  --metric-interval 1 --metric-batches 20 --metric-samples 2 --log-interval 104
"$PY" -u evaluate_multitask_pilot.py --run "$OUT" --manifest "$PACK/manifest.json"

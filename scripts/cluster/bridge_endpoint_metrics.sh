#!/bin/bash
#SBATCH --job-name=bridge-endpoint
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --partition=cluster02
#SBATCH --cpus-per-task=4
#SBATCH --mem=8192M
#SBATCH --time=00:10:00
#SBATCH --chdir=/projects/Zeroshot
#SBATCH --output=/projects/Zeroshot/logs/bridge-endpoint-%j.out
set -eo pipefail
cd /projects/Zeroshot/bridge_endpoint_metrics_v1
sha256sum -c SHA256SUMS
/projects/Zeroshot/envs/multitask-preflight-v1/bin/python -u evaluate_bridge_endpoint_metrics.py \
  --run /projects/Zeroshot/runs/bridge-head-train-192651 \
  --training-report /projects/Zeroshot/runs/bridge-head-train-192651/report.json \
  --window-reference /projects/Zeroshot/runs/bridge-full-chain-191004.json \
  --output /projects/Zeroshot/runs/bridge-endpoint-${SLURM_JOB_ID}.json

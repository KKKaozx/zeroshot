#!/bin/bash
#SBATCH --job-name=decoder-pair-setup
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --partition=cluster02
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --chdir=/projects/Zeroshot
#SBATCH --output=/projects/Zeroshot/logs/decoder-pair-setup-%j.out
set -eo pipefail
source /projects/Zeroshot/scripts/ssd_cache.sh
export TMPDIR="/projects/Zeroshot/.tmp/${SLURM_JOB_ID}"
mkdir -p "$TMPDIR" /projects/Zeroshot/baseline_setup
PYTHON=/projects/Zeroshot/envs/bridge-diffusion/bin/python
test -x "$PYTHON"
sha256sum -c decoder_pair_v1.sha256
test ! -e /projects/Zeroshot/baseline_setup/decoder_pair_v1
"$PYTHON" -m pip install --no-deps einops==0.8.2
"$PYTHON" -m zipfile -e decoder_pair_v1.zip /projects/Zeroshot/baseline_setup
"$PYTHON" - <<'PY'
from pathlib import Path
import hashlib,json
root=Path('/projects/Zeroshot/baseline_setup/decoder_pair_v1')
for name,wanted in json.loads((root/'integrity.json').read_text()).items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==wanted,name
print('DECODER PAIR TRANSFER AND SETUP: PASSED')
PY

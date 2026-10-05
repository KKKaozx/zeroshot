#!/bin/bash
#SBATCH --job-name=bridge-unpack
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --partition=cluster02
#SBATCH --cpus-per-task=2
#SBATCH --time=00:15:00
#SBATCH --chdir=/projects/Zeroshot
#SBATCH --output=/projects/Zeroshot/logs/bridge-unpack-%j.out

set -eo pipefail
source /projects/Zeroshot/scripts/ssd_cache.sh
export TMPDIR="/projects/Zeroshot/.tmp/${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"

/projects/Zeroshot/envs/zeroshot/bin/python - \
  /projects/Zeroshot/bridge_single_step_subset_v1.zip \
  /projects/Zeroshot/data <<'PY'
import hashlib
import json
import sys
import zipfile
from pathlib import Path

archive = Path(sys.argv[1])
base = Path(sys.argv[2]).resolve()
target = base / 'bridge_single_step_subset'
expected = '0668db873a910159cddf7f0142ab7b93bb29f843a21f6c7b41ff4b44d706f1df'

def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

if target.exists():
    raise SystemExit(f'Refusing to overwrite existing directory: {target}')
if sha256(archive) != expected:
    raise SystemExit('Archive SHA256 mismatch; upload the archive again.')
with zipfile.ZipFile(archive) as package:
    for entry in package.infolist():
        parts = entry.filename.split('/')
        if parts[0] != target.name or '..' in parts or '\\' in entry.filename:
            raise SystemExit(f'Unexpected archive member: {entry.filename}')
        (base / entry.filename).resolve().relative_to(base)
    bad = package.testzip()
    if bad:
        raise SystemExit(f'Archive CRC mismatch: {bad}')
    base.mkdir(parents=True, exist_ok=True)
    package.extractall(base)
manifest = json.loads((target / 'export_manifest.json').read_text(encoding='utf-8'))
checks = json.loads((target / 'read_check.json').read_text(encoding='utf-8'))
for relative, metadata in manifest['files'].items():
    path = (target / relative).resolve()
    path.relative_to(target.resolve())
    if path.stat().st_size != metadata['bytes'] or sha256(path) != metadata['sha256']:
        raise SystemExit(f'Extracted file mismatch: {relative}')
if manifest['test_targets_used'] is not False or checks['test_targets_used'] is not False:
    raise SystemExit('Unexpected test partition flag.')
if not checks['passed'] or checks['partitions']['train']['windows'] != 316 or checks['partitions']['validation']['windows'] != 55:
    raise SystemExit('Unexpected local read-check result.')
print(f'Archive and extracted file integrity: PASSED\nData directory: {target}')
print('Saved local author-loader check: train=316, validation=55, test targets unused.')
print('This CPU job checks transfer integrity; it does not run TensorFlow or training.')
PY

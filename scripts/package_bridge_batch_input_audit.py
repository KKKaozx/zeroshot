"""Package CPU diagnostic references without changing the failed experiment."""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ref = ROOT/"reports/bridge_batch_input_reference_20261012"
    local = json.loads((ROOT/"reports/bridge_batch_input_audit_local_20261012/report.json").read_text())
    assert local["comparison_completed"] and local["summary"]["all_raw_and_decoded_inputs_equal"]
    assert local["summary"]["normalized_images_exact_count"] == 72
    assert local["summary"]["max_target_component_difference"] == 0
    pack = ROOT/"cluster_packages/bridge_batch_input_audit_v1"
    pack.mkdir(exist_ok=False)
    for name in ["dataset.py", "audit_bridge_batch_inputs.py"]:
        shutil.copyfile(ROOT/"code"/name, pack/name)
    shutil.copyfile(ROOT/"cluster_packages/bridge_batch_control_v1/manifest.json", pack/"manifest.json")
    shutil.copytree(ref, pack/"reference")
    (pack/"audit.sh").write_text('''#!/bin/bash
#SBATCH --job-name=bridge-input-audit
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --partition=cluster02
#SBATCH --cpus-per-task=4
#SBATCH --mem=8192M
#SBATCH --time=00:05:00
#SBATCH --chdir=/projects/Zeroshot
#SBATCH --output=/projects/Zeroshot/logs/bridge-input-audit-%j.out
set -eo pipefail
source /projects/Zeroshot/scripts/ssd_cache.sh
export TMPDIR="/projects/Zeroshot/.tmp/${SLURM_JOB_ID}"
export CUDA_VISIBLE_DEVICES="" USE_TF=0 OMP_NUM_THREADS=4
mkdir -p "$TMPDIR" /projects/Zeroshot/runs
cd /projects/Zeroshot/bridge_batch_input_audit_v1
sha256sum -c SHA256SUMS
/projects/Zeroshot/envs/multitask-preflight-v1/bin/python -u audit_bridge_batch_inputs.py \\
  --manifest manifest.json --data-root /projects/Zeroshot/bridge_expansion_training_v1/data \\
  --reference reference --output "/projects/Zeroshot/runs/bridge-input-audit-${SLURM_JOB_ID}"
''', newline="\n")
    files = sorted(p for p in pack.rglob("*") if p.is_file())
    (pack/"SHA256SUMS").write_text("".join(f"{sha(p)}  {p.relative_to(pack).as_posix()}\n" for p in files), newline="\n")
    archive = pack.parent/"bridge_batch_input_audit_v1.zip"
    assert not archive.exists()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(pack.rglob("*")):
            if p.is_file():
                z.write(p, pack.name+"/"+p.relative_to(pack).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for p in pack.rglob("*"):
            if p.is_file():
                assert z.read(pack.name+"/"+p.relative_to(pack).as_posix()) == p.read_bytes()
    archive.with_suffix(".sha256").write_text(f"{sha(archive)}  {archive.name}\n", newline="\n")
    print("CPU INPUT AUDIT PACKAGE VERIFIED", archive, "MiB", round(archive.stat().st_size/2**20, 2))


if __name__ == "__main__":
    main()

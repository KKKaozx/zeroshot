"""Freeze reused successful components and release the matched training entry."""
import ast
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "scripts/cluster/pusht_project_training_v1"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, text):
    (BUNDLE / name).write_text(text, encoding="utf-8", newline="\n")


def main():
    # Original files, no rewritten optimizer/loss/restore/CLIP/Adapter operations.
    copies = {
        "training_core.py": "pusht_native_extension_v1/training_core.py",
        "checkpoint_config.yaml": "pusht_native_extension_v1/checkpoint_config.yaml",
        "base_protocol.json": "pusht_native_extension_v1/training_protocol.json",
        "base_verifier.py": "pusht_native_extension_v1/base_verifier.py",
        "native_data.py": "pusht_project_cache_v1/native_data.py",
        "cache_tools.py": "pusht_project_cache_v1/cache_tools.py",
        "project_frontend.py": "pusht_frontend_resource_v1/project_frontend.py",
        "frozen_adapter.py": "pusht_frontend_resource_v1/frozen_adapter.py",
        **{name: "pusht_live_bridge_v1/" + name for name in
           ("live_client.py", "clip_worker.py", "wire.py", "live_contract.json", "resource_probe.py")}}
    for name, relative in copies.items():
        (BUNDLE / name).write_bytes((ROOT / "scripts/cluster" / relative).read_bytes())
    # Exact environment dynamics, sampling seed plan and reward calculation.
    source = (ROOT / "scripts/cluster/pusht_native_extension_v1/effective_train_native.py").read_text()
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "evaluate")
    evaluation = ast.get_source_segment(source, node)
    evaluation = evaluation.replace("PUSHT_NATIVE_DEV", "PUSHT_PROJECT_DEV")
    evaluation = evaluation.replace("Native cap competent; review before preparing matched84168 frozen-CLIP project group",
                                    "Compare fixed endpoint project frontend with admitted native reference; keep final50 sealed")
    evaluation = evaluation.replace("Native cap not competent; stop this module-comparison branch, no extra budget or project attribution",
                                    "Project below reference competence threshold; inspect matched frontend evidence before new experiments")
    write("evaluate_project.py", '"""Original native evaluator with project labels only; live policy wrapper."""\n'
          'import math\nimport numpy as np\nimport torch\nfrom native_data import save_json\n\n' + evaluation + '\n')
    common = '''set -euo pipefail
ROOT=/projects/Zeroshot
BUNDLE="$ROOT/pusht_project_training_v1"
OFFICIAL="$ROOT/envs/official-pusht-v1"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "$BUNDLE"
sha256sum -c SHA256SUMS
'''
    for segment in (1, 2):
        script = '''#!/bin/bash
#SBATCH --job-name=pusht-project%d
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=22G
#SBATCH --time=20:00:00
#SBATCH --chdir=/projects/Zeroshot
#SBATCH --output=/projects/Zeroshot/logs/pusht-project-part%d-%%j.out
''' % (segment, segment)
        script += common
        if segment == 2:
            script += ': "${PUSHT_PROJECT_RESUME:?Set exact verified project segment1 latest.pt}"\n'
        script += 'ARGS=(--root "$ROOT" --output "$ROOT/runs/pusht-project-part%d-${SLURM_JOB_ID}" --segment %d)\n' % (segment, segment)
        script += 'if [[ -n "${PUSHT_PROJECT_RESUME:-}" ]]; then ARGS+=(--resume "$PUSHT_PROJECT_RESUME"); fi\n'
        script += 'LD_LIBRARY_PATH="$OFFICIAL/lib" "$OFFICIAL/bin/python" -B train_project.py "${ARGS[@]}"\n'
        write("part%d.sh" % segment, script)
    write("submit_part1_once.sh", '''#!/bin/bash
set -euo pipefail
cd /projects/Zeroshot/pusht_project_training_v1
sha256sum -c SHA256SUMS
if [[ -n "${PUSHT_PROJECT_RESUME:-}" ]]; then echo 'Use a separately reviewed resume, not the fresh submit wrapper.' >&2; exit 1; fi
LOCK=.part1_submission
if ! mkdir "$LOCK"; then echo 'Submission already attempted. Inspect existing job; no automatic repeat.' >&2; exit 1; fi
if JOB=$(sbatch --parsable part1.sh); then
  printf '%s\n' "$JOB" > "$LOCK/job_id"
  echo "Push-T项目组第一段：$JOB"
else
  echo 'Submission failed; retained guard. Inspect the scheduler error before manual retry.' >&2
  exit 1
fi
''')
    report = ROOT / "reports/pusht_project_cache_213439.json"
    native = json.loads((ROOT / "reports/pusht_native_212158/report.json").read_text())
    contract = {"stage": "matched_project_frontend_training", "native_reference_job": 212158,
                "base_protocol_sha256": sha(BUNDLE / "base_protocol.json"),
                "cache_directory": "runs/pusht-project-cache-213439", "cache_report_sha256": sha(report),
                "required_reports_sha256": {
                    "runs/pusht-project-cache-213439/report.json": sha(report),
                    "runs/pusht-native84k-212158/report.json": sha(ROOT / "reports/pusht_native_212158/report.json")},
                "sample_order_sha256": native["sample_order_sha256"],
                "sample_plan_sha256": native["sample_plan_sha256"],
                "total_updates": 84168, "segment_endpoints": [42084, 84168],
                "clip_vision_and_text_frozen": True, "trained_native_weights_allowed": False,
                "fresh_backend_seed": 42, "adapter_projection_seed": 43,
                "evaluation": "Fixed endpoint EMA; same dev20 seed/sample plan; no segment1 scoring or final50 access",
                "comparison_scope": "Frontend family including preprocessing, not Adapter-only attribution",
                "automatic_following_segment_submission": False,
                "reused_files_byte_identical": copies}
    contract["bundle_sources_sha256"] = {p.name: sha(p) for p in sorted(BUNDLE.iterdir())
                                         if p.is_file() and p.suffix in (".py", ".yaml")}
    write("project_contract.json", json.dumps(contract, indent=2) + "\n")
    files = sorted(p for p in BUNDLE.iterdir() if p.is_file() and p.name != "SHA256SUMS")
    write("SHA256SUMS", "".join(sha(p) + "  " + p.name + "\n" for p in files))
    archive = ROOT / "cluster_packages/pusht_project_training_v1.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for path in sorted(BUNDLE.iterdir()):
            if path.is_file():
                z.write(path, "pusht_project_training_v1/" + path.name)
    release = {"archive": str(archive.relative_to(ROOT)), "bytes": archive.stat().st_size,
               "sha256": sha(archive), "project_contract_sha256": sha(BUNDLE / "project_contract.json"),
               "files": len(files) + 1, "training_submitted": False,
               "real_project_training_executed": False, "no_new_resource_or_wiring_probe": True}
    (ROOT / "reports/pusht_project_training_release_20261020.json").write_text(json.dumps(release, indent=2) + "\n")
    print(json.dumps(release, indent=2))


if __name__ == "__main__":
    main()

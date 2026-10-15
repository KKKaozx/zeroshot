"""Freeze the bounded late-LR intervention, reusing the verified trace package."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT/"cluster_packages/bridge_late_lr_pair_v1"


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--package", action="store_true"); args = ap.parse_args()
    if not args.package:
        assert not PACK.with_suffix(".zip").exists(), "Do not alter a verified bundle"
        old = ROOT/"cluster_packages/bridge_training_trace_v1"
        manifest = copy.deepcopy(json.loads((old/"manifest.json").read_text()))
        PACK.mkdir(exist_ok=True)
        for name, expected in manifest["source_sha256"].items():
            assert sha(old/name) == expected
            shutil.copyfile(old/name, PACK/name)
        name = "check_bridge_late_lr_pair.py"
        shutil.copyfile(ROOT/"code"/name, PACK/name)
        manifest["source_sha256"][name] = sha(PACK/name)
        manifest.update(stage="preregistered_late_lr_pair", initialization_seeds=[43],
            changing_variable="Continuation learning rate after common update1024: 1e-4 versus1e-5",
            evaluation="Common train0/512/1024; branch train1536/1792/1920/2048; development only final2048",
            preflight="Seed43 common4updates, continuous4 versus restored4 at same LR; reducedLR4. All32 train DDPM100; no weights saved",
            storage_policy="No installs/downloads. FourGiB cluster PT cap. Local only small reports/NPZ/JSONL.")
        manifest["late_lr"] = dict(origin_report_sha256=sha(ROOT/"reports/bridge-trace-fit-200039/report.json"),
            origin_manifest_sha256=sha(old/"manifest.json"), branch_update=1024, final_update=2048,
            rates={"constant":1e-4, "reduced":1e-5}, initialization_seed=43,
            common_state="Identical weights and Adam moments/step; no optimizer reset",
            inference="Unchanged native DDPM100, three original sampling seeds",
            control_reproduction_tolerance=dict(position_cm=.1, rotation_deg=.5),
            warmup_must_pass_existing_fit_gate=True, checkpoint_budget_bytes=4*2**30,
            checkpoints="Common1024 withAdam; two final trainable states. Cluster only.",
            hypothesis="Reduced continuation update size may retain an already fitted state; not a unique module diagnosis.")
        # Historical trace metadata is retained for provenance, not executed by this runner.
        (PACK/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
        for stage, limit in [("preflight", "00:20:00"), ("fit", "01:30:00")]:
            dependency = '' if stage == 'preflight' else ': "${LATE_LR_PREFLIGHT_JOB:?Supply successful preflight job ID}"\n'
            reference = '' if stage == 'preflight' else ' \\\n  --preflight-reference "/projects/Zeroshot/runs/bridge-late-lr-preflight-${LATE_LR_PREFLIGHT_JOB}"'
            quota = '' if stage == 'preflight' else '''PROJECT_USED_BYTES=$(du -s -B1 /projects/_ssd/Zeroshot | cut -f1)
echo "Project apparent bytes (planning estimate, not live quota): $PROJECT_USED_BYTES"
if (( PROJECT_USED_BYTES + 4*1024*1024*1024 + 15*1024*1024*1024 > 200*1024*1024*1024 )); then
  echo "Insufficient planning headroom"; exit 1
fi
'''
            shell = f'''#!/bin/bash
#SBATCH --job-name=bridge-late-lr-{stage}
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=22528M
#SBATCH --time={limit}
#SBATCH --chdir=/projects/Zeroshot
#SBATCH --output=/projects/Zeroshot/logs/bridge-late-lr-{stage}-%j.out
set -eo pipefail
source /projects/Zeroshot/scripts/ssd_cache.sh
export TMPDIR="/projects/Zeroshot/.tmp/${{SLURM_JOB_ID}}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 USE_TF=0 OMP_NUM_THREADS=4
mkdir -p "$TMPDIR" /projects/Zeroshot/runs
{dependency}{quota}cd /projects/Zeroshot/bridge_late_lr_pair_v1
sha256sum -c SHA256SUMS
nvidia-smi
/projects/Zeroshot/envs/multitask-preflight-v1/bin/python -u check_bridge_late_lr_pair.py \\
  --manifest manifest.json --data-root /projects/Zeroshot/bridge_expansion_training_v1/data \\
  --clip /projects/Zeroshot/baseline_setup/clip-vit-large-patch14-v1 \\
  --data-reference /projects/Zeroshot/bridge_batch_input_audit_v1/reference \\
  --origin-report /projects/Zeroshot/runs/bridge-trace-fit-200039/report.json \\
  --stage {stage} --output "/projects/Zeroshot/runs/bridge-late-lr-{stage}-${{SLURM_JOB_ID}}"{reference}
'''
            (PACK/f"{stage}.sh").write_text(shell, newline="\n")
        (PACK/"README.md").write_text("# Fitted-state late learning-rate control\n\n"
            "Same seed43/32 train windows and recipe as trace200039. Common warmup1024 then branch with Adam state restored; "
            "only continuation LR differs. Preflight verifies continuous versus restored same-LR exact replay. "
            "Common warmup must fit and match origin. Constant arm must reproduce deterioration before causal interpretation. "
            "No model, dataset, package downloads; no gripper or simulation training. PT files stay on cluster, capped4GiB.\n")
        print("LATE LR PACKAGE PREPARED", PACK)
        return
    preflight = ROOT/"reports/bridge_late_lr_local_preflight_20261012/report.json"
    r = json.loads(preflight.read_text())
    assert r["execution_checks_passed"] and r["branch_restore_passed"] and r["lr_intervention_changes_parameters"]
    assert not r["checkpoints"] and r["manifest_sha256"] == sha(PACK/"manifest.json")
    (PACK/"local_preflight_summary.json").write_text(json.dumps(dict(passed=True,
        report_sha256=sha(preflight), restore=r["branch_restore"], elapsed_seconds=r["elapsed_seconds"],
        peak_reserved_gib=r["peak_reserved_gib"]), indent=2)+"\n")
    files = sorted(p for p in PACK.iterdir() if p.is_file() and p.name != "SHA256SUMS")
    (PACK/"SHA256SUMS").write_text(''.join(f"{sha(p)}  {p.name}\n" for p in files), newline="\n")
    archive = PACK.with_suffix(".zip"); assert not archive.exists()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(PACK.iterdir()):
            if p.is_file(): z.write(p, PACK.name+'/'+p.name)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert all(z.read(PACK.name+'/'+p.name) == p.read_bytes() for p in PACK.iterdir() if p.is_file())
    archive.with_suffix('.sha256').write_text(f"{sha(archive)}  {archive.name}\n", newline="\n")
    print("LATE LR PACKAGE VERIFIED", archive, archive.stat().st_size, 'bytes')


if __name__ == "__main__": main()

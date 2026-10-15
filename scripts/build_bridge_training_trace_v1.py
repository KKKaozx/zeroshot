"""Freeze observer package without touching the previously executed batch package."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT/"cluster_packages/bridge_training_trace_v1"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--package", action="store_true"); args = ap.parse_args()
    if not args.package:
        original = ROOT/"cluster_packages/bridge_batch_control_v2"
        manifest = copy.deepcopy(json.loads((original/"manifest.json").read_text()))
        assert not PACK.with_suffix(".zip").exists(), "Do not rewrite a verified upload bundle"
        PACK.mkdir(exist_ok=True)
        for name, digest in manifest["source_sha256"].items():
            assert sha(original/name) == sha(ROOT/"code"/name) == digest
            shutil.copyfile(original/name, PACK/name)
        shutil.copyfile(ROOT/"code/trace_bridge_batch_training.py", PACK/"trace_bridge_batch_training.py")
        manifest["source_sha256"]["trace_bridge_batch_training.py"] = sha(PACK/"trace_bridge_batch_training.py")
        manifest["arms"] = ["mixed"]
        manifest["stage"] = "preregistered_training_observation"
        manifest["changing_variable"] = "Observation only; no recipe intervention; original mixed arm repeated"
        manifest["preflight"] = "Eight mixed updates with monitoring off/on from identical initial weights; all32 train/all40 final development"
        manifest["evaluation"] = "Train at predefined checkpoints; all40 development only at final; no early stop or checkpoint selection"
        manifest["storage_policy"] = "Reuse assets. Local preflight saves no weights. Formal final trainable states stay on cluster, all PT below4GiB."
        manifest["revision"] = "Training observation only: original mixed recipe, no added updates or model changes"
        manifest["trace"] = dict(origin_manifest_sha256=sha(original/"manifest.json"),
            origin_report_sha256=sha(ROOT/"reports/bridge-batch-fit-v2-198871/report.json"),
            train_monitor_updates=[0, 64, 256, 512, 1024, 1536, 2048],
            development_evaluated_at="Final update only; all40 windows, no checkpoint selection",
            gradient_logging="Every update, before clipping: Adapter, U-Net and history projection",
            observer_guard="Restore NumPy/Torch RNG and per-module mode; parameters/buffers immutable during evaluation",
            checkpoint_budget_bytes=4*2**30, final_checkpoints="Three trainable-only states; seed43 also retains Adam moments",
            local_weights_saved=False, max_updates=2048, inference="Full native DDPM100, same three sampling seeds")
        (PACK/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
        for stage in ("preflight", "fit"):
            source = (original/f"{stage}.sh").read_text()
            source = source.replace("bridge_batch_control_v2", "bridge_training_trace_v1")
            source = source.replace("bridge-batch-preflight-v2", "bridge-trace-preflight")
            source = source.replace("bridge-batch-fit-v2", "bridge-trace-fit")
            source = source.replace("check_bridge_batch_control.py", "trace_bridge_batch_training.py")
            source = source.replace("BATCH_PREFLIGHT_JOB", "TRACE_PREFLIGHT_JOB")
            source = source.replace("--data-reference /projects/Zeroshot/bridge_batch_input_audit_v1/reference --stage fit", 
                "--data-reference /projects/Zeroshot/bridge_batch_input_audit_v1/reference --stage fit --origin-report /projects/Zeroshot/runs/bridge-batch-fit-v2-198871/report.json")
            if stage == "fit":
                source = source.replace("cd /projects/Zeroshot/bridge_training_trace_v1", 
                    'PROJECT_USED_BYTES=$(du -s -B1 /projects/_ssd/Zeroshot | cut -f1)\n'
                    'echo "Project apparent bytes (planning estimate, not live quota): $PROJECT_USED_BYTES"\n'
                    'if (( PROJECT_USED_BYTES + 4*1024*1024*1024 + 15*1024*1024*1024 > 200*1024*1024*1024 )); then\n'
                    '  echo "Insufficient planning headroom; stop before training"; exit 1\nfi\n'
                    'cd /projects/Zeroshot/bridge_training_trace_v1')
            (PACK/f"{stage}.sh").write_text(source, newline="\n")
        (PACK/"README.md").write_text("# Fixed mixed-batch training trace\n\n"
            "Same32 train windows, Adam1e-4, batch8, 2048updates, 512draws/window, pose-only, frozen CLIP, x0/DDPM100. "
            "Observer only; no production or prior runner changes. Train checks at0/64/256/512/1024/1536/2048; "
            "development only at final. Same-package preflight tests monitoring on/off and RNG/mode preservation. "
            "Final trainable checkpoints stay on cluster; only seed43 includes optimizer. Total PT limit4GiB. "
            "No dependency/model/data downloads. Reuse bridge_batch_input_audit_v1/reference and existing dataset/CLIP/env.\n")
        print("TRACE PACKAGE PREPARED", PACK)
        return
    preflight = ROOT/"reports/bridge_trace_local_preflight_20261012/report.json"
    r = json.loads(preflight.read_text())
    assert r["execution_checks_passed"] and r["observer_replay_passed"] and not r["checkpoints_saved"]
    assert r["manifest_sha256"] == sha(PACK/"manifest.json")
    summary = dict(local_preflight_report_sha256=sha(preflight), passed=True, observer_replay=r["observer_replay"],
        elapsed_seconds=r["elapsed_seconds"], peak_reserved_gib=r["runs"][0]["peak_reserved_gib"])
    (PACK/"local_preflight_summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    files = sorted(p for p in PACK.iterdir() if p.is_file() and p.name != "SHA256SUMS")
    (PACK/"SHA256SUMS").write_text("".join(f"{sha(p)}  {p.name}\n" for p in files), newline="\n")
    archive = PACK.with_suffix(".zip"); assert not archive.exists()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(PACK.iterdir()):
            if p.is_file(): z.write(p, PACK.name+"/"+p.name)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert all(z.read(PACK.name+"/"+p.name) == p.read_bytes() for p in PACK.iterdir() if p.is_file())
    archive.with_suffix(".sha256").write_text(f"{sha(archive)}  {archive.name}\n", newline="\n")
    print("TRACE PACKAGE VERIFIED", archive, "bytes", archive.stat().st_size)


if __name__ == "__main__": main()

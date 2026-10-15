"""Revise input guards only; preserve v1 and reuse the uploaded audit reference."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", action="store_true")
    args = ap.parse_args()
    old = ROOT/"cluster_packages/bridge_batch_control_v1"
    pack = ROOT/"cluster_packages/bridge_batch_control_v2"
    if not args.package:
        pack.mkdir(exist_ok=False)
        manifest = json.loads((old/"manifest.json").read_text())
        files = list(manifest["source_sha256"]) + ["bridge_input_contract.py"]
        for name in files:
            if name not in ["check_bridge_batch_control.py", "bridge_input_contract.py"]:
                assert sha(ROOT/"code"/name) == manifest["source_sha256"][name]
            shutil.copyfile(ROOT/"code"/name, pack/name)
        manifest["source_sha256"] = {n:sha(pack/n) for n in files}
        manifest["revision"] = "v2 input validation only; v1 recipe and rows unchanged"
        manifest["input_reference_manifest_sha256"] = sha(old/"manifest.json")
        ref = ROOT/"reports/bridge_batch_input_reference_20261012"
        manifest["input_reference_files_sha256"] = {n:sha(ref/n) for n in ["metadata.json", "arrays.npz"]}
        manifest["input_validation"] = dict(raw_and_decoded_identity="Exact SHA256 and image dimensions",
            absolute_tolerance=dict(images=1e-6, targets=1e-6, currents=0), relative_tolerance=0,
            gripper_labels="Exact", cluster_preflight_to_fit_processed_hashes="Exact in the same environment",
            evidence="User supplied cluster CPU audit 198856: raw equal, image max 4.76837158203125e-7, target max 4.470348358154297e-7")
        (pack/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
        for stage in ["preflight", "fit"]:
            source = (old/f"{stage}.sh").read_text()
            source = source.replace("bridge_batch_control_v1", "bridge_batch_control_v2")
            source = source.replace("bridge-batch-preflight", "bridge-batch-preflight-v2")
            source = source.replace("bridge-batch-fit", "bridge-batch-fit-v2")
            source = source.replace("--data-reference local_data_reference.json", "--data-reference /projects/Zeroshot/bridge_batch_input_audit_v1/reference")
            (pack/f"{stage}.sh").write_text(source, newline="\n")
        (pack/"README.md").write_text("# Batch control v2\n\nInput-check repair only; same 32 train/40 development rows, models and budgets as v1. "
            "Requires the already uploaded bridge_batch_input_audit_v1/reference. Its metadata and array SHA256 are pinned. "
            "Raw image/state/action/instruction and decoded RGB must match exactly. Processed images/pose values have absolute tolerance 1e-6, rtol=0; "
            "gripper labels/current remain exact. Same-cluster preflight-to-fit hashes remain exact. "
            "No packages, weights, data downloads or large checkpoints. Preflight execution does not establish fitting/generalization.\n")
        print("V2 INPUT GUARDS PREPARED", pack)
        return
    result = ROOT/"reports/bridge_batch_preflight_v2_20261012/report.json"
    local = json.loads(result.read_text())
    assert local["execution_checks_passed"] and local["input_contract_passed"]
    assert local["manifest_sha256"] == sha(pack/"manifest.json")
    (pack/"local_preflight_summary.json").write_text(json.dumps(dict(report_sha256=sha(result),
        execution_checks_passed=True, input_contract_passed=True, clip_unchanged=local["clip_unchanged"],
        elapsed_seconds=local["elapsed_seconds"], peak_reserved_gib=max(r["peak_reserved_gib"] for r in local["runs"])), indent=2)+"\n")
    files = sorted(p for p in pack.iterdir() if p.is_file())
    (pack/"SHA256SUMS").write_text("".join(f"{sha(p)}  {p.name}\n" for p in files), newline="\n")
    archive = pack.parent/"bridge_batch_control_v2.zip"
    assert not archive.exists()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(pack.iterdir()):
            z.write(p, pack.name+"/"+p.name)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for p in pack.iterdir():
            assert z.read(pack.name+"/"+p.name) == p.read_bytes()
    archive.with_suffix(".sha256").write_text(f"{sha(archive)}  {archive.name}\n", newline="\n")
    print("V2 PACKAGE VERIFIED", archive, "bytes", archive.stat().st_size)


if __name__ == "__main__":
    main()

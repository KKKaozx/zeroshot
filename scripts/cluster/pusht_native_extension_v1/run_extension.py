"""Continue verified native211309 to the already declared84168 update cap."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import runpy
import sys


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/projects/Zeroshot"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path, help="Only for interruption of this exact extension; default resumes211309")
    args = parser.parse_args()
    root = args.root.resolve()
    bundle = Path(__file__).resolve().parent
    contract_path = bundle / "extension_contract.json"
    contract = json.loads(contract_path.read_text())
    for relative, digest in contract["required_reference_files_sha256"].items():
        if sha(root / relative) != digest:
            raise ValueError("Native20k reference file changed: " + relative)
    checkpoint = root / contract["source_checkpoint"]
    if args.resume:
        checkpoint = args.resume.resolve()
        if root / "runs" not in checkpoint.parents or checkpoint.name != "latest.pt":
            raise ValueError("Explicit extension resume must be latest.pt under root/runs")
        sidecar = json.loads(checkpoint.with_suffix(".json").read_text())
        if sidecar.get("extension_contract_sha256") != sha(contract_path):
            raise ValueError("Resume checkpoint belongs to a different extension")
        if not 20000 < sidecar["completed_updates"] <= 84168:
            raise ValueError("Resume counter is outside the declared extension")
    else:
        sidecar = json.loads(checkpoint.with_suffix(".json").read_text())
        if sidecar["sha256"] != contract["source_checkpoint_sha256"] or sidecar["completed_updates"] != 20000:
            raise ValueError("Default source is not the verified native20k checkpoint")
    if sidecar["protocol_sha256"] != contract["base_protocol_sha256"] or sidecar["sample_plan_sha256"] != contract["sample_plan_sha256"]:
        raise ValueError("Resume changes base protocol or training input/sample plan")
    # Bind the already GPU-validated device repair before loading the runner.
    spec = importlib.util.spec_from_file_location("training_core", bundle / "training_core.py")
    core = importlib.util.module_from_spec(spec)
    sys.modules["training_core"] = core
    spec.loader.exec_module(core)
    sys.path.insert(0, str(bundle))
    namespace = runpy.run_path(str(bundle / "effective_train_native.py"), run_name="native_extension")
    if namespace["restore"] is not core.restore:
        raise ValueError("Runner did not bind the repaired restore helper")
    sys.argv = [str(bundle / "effective_train_native.py"), "--root", str(root),
                "--output", str(args.output), "--resume", str(checkpoint)]
    try:
        namespace["main"]()
    finally:
        output = args.output.resolve()
        if output.is_dir() and root / "runs" in output.parents:
            provenance = {"extension_contract_sha256": sha(contract_path),
                          "source_checkpoint": str(checkpoint),
                          "source_completed_updates": sidecar["completed_updates"],
                          "original20k_source_preserved": True,
                          "restored_device_checks": core.RESTORE_DEVICE_CHECKS}
            (output / "extension_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()

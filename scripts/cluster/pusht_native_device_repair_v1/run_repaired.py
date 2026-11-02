"""Bind the repaired restore helper, execute unchanged native training runner."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/projects/Zeroshot"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    bundle = Path(__file__).resolve().parent
    contract = json.loads((bundle / "repair_contract.json").read_text())
    original = root / "pusht_native_training_v1"
    for name, digest in contract["original_file_sha256"].items():
        if sha(original / name) != digest:
            raise ValueError("Original bundle changed: " + name)
    # Register before run_path; avoid the old helper winning module search.
    spec = importlib.util.spec_from_file_location("training_core", bundle / "training_core_repaired.py")
    repaired = importlib.util.module_from_spec(spec)
    sys.modules["training_core"] = repaired
    spec.loader.exec_module(repaired)
    sys.path.insert(0, str(original))
    namespace = runpy.run_path(str(original / "train_native.py"), run_name="native_device_repaired")
    sys.argv = [str(original / "train_native.py"), "--root", str(root), "--output", str(args.output)]
    # Restart from the same fresh initialization; no failed-audit checkpoint use.
    try:
        namespace["main"]()
    finally:
        output = args.output.resolve()
        if output.is_dir() and root / "runs" in output.parents:
            record = {"repair": "CPU-loaded normalizer tensors returned to destination device",
                      "original_protocol_unchanged": True,
                      "repair_contract_sha256": sha(bundle / "repair_contract.json"),
                      "restored_device_checks": repaired.RESTORE_DEVICE_CHECKS,
                      "failed_job_resumed": False}
            (output / "device_repair.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()

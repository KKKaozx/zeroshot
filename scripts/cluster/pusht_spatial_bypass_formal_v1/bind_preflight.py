"""Validate the existing preflight once and bind its exact small artifacts to this run."""
import argparse
import json
from pathlib import Path
from native_data import sha, save_json
from verify_preflight import validate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/projects/Zeroshot"))
    parser.add_argument("--binding", type=Path, required=True)
    args = parser.parse_args()
    root, bundle = args.root.resolve(), Path(__file__).resolve().parent
    contract = json.loads((bundle / "formal_contract.json").read_text())
    preflight = root / contract["preflight_directory"]
    if args.binding.exists():
        raise ValueError("Binding already exists; do not silently replace an admitted preflight")
    if sha(bundle / "protocol.json") != contract["preflight_protocol_sha256"]:
        raise ValueError("Preflight protocol changed")
    for name, digest in contract["bundle_sources_sha256"].items():
        if sha(bundle / name) != digest:
            raise ValueError("Formal package changed: " + name)
    verified = validate(preflight, bundle)
    report = json.loads((preflight / "report.json").read_text())
    binding = {"format": "pusht-bypass-preflight-binding-v1", "preflight_job": 217778,
               "formal_contract_sha256": sha(bundle / "formal_contract.json"),
               "preflight_protocol_sha256": sha(bundle / "protocol.json"),
               "preflight_report_sha256": sha(preflight / "report.json"),
               "statistics_sha256": sha(preflight / "bypass_statistics.npz"),
               "source_checkpoint_sha256": report["source_checkpoint_sha256"], "verified": verified,
               "preflight_updates_used_as_training_start": False}
    save_json(args.binding, binding)
    print("SPATIAL BYPASS PREFLIGHT BOUND; endpoint and statistics admitted, no task-performance verdict")
    print(json.dumps(verified, indent=2))


if __name__ == "__main__":
    main()

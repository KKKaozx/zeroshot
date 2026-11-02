"""CPU-only verification of the84168 endpoint and original20k preservation."""
import argparse
import hashlib
import json
from pathlib import Path
from base_verifier import validate_metrics


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    folder = args.report.parent
    bundle = Path(__file__).resolve().parent
    contract_path = bundle / "extension_contract.json"
    contract = json.loads(contract_path.read_text())
    protocol_path = bundle / "training_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    report = json.loads(args.report.read_text())
    startup = json.loads((folder / "startup_checks.json").read_text())
    dev = json.loads((folder / "development.json").read_text())
    provenance = json.loads((folder / "extension_provenance.json").read_text())
    assert not report["errors"] and report["training_completed"] and report["development_complete"]
    assert report["completed_updates"] == contract["target_updates"] == 84168
    assert 20000 <= report["initial_completed_updates"] <= 84168
    assert report["protocol_sha256"] == sha(protocol_path) == contract["base_protocol_sha256"]
    assert report["extension_contract_sha256"] == provenance["extension_contract_sha256"] == sha(contract_path)
    assert report["sample_plan_sha256"] == contract["sample_plan_sha256"]
    assert report["sample_order_sha256"] == contract["sample_order_sha256"]
    assert report["data"]["raw_training_images_sha256"] == contract["raw_training_images_sha256"]
    assert report["complete_resume_state_exact"]
    assert startup["resume_checksum_verified"] and all(startup["loss_audit"].values())
    assert provenance["source_completed_updates"] == report["initial_completed_updates"]
    assert provenance["original20k_source_preserved"]
    assert provenance["restored_device_checks"] and all(x["raw_and_ema_all_tensors_on_destination"] for x in provenance["restored_device_checks"])
    assert report["source20k_checkpoint_sha256"] == contract["source_checkpoint_sha256"]
    assert report["lr_horizon_updates"] == protocol["lr_horizon_updates"] == 512400
    for key in ("initial_backend_sha256", "normalizer_sha256"):
        assert report[key] == protocol[key]
    assert report["data"]["resource_64_windows_exact"] and report["data"]["normalizer_inputs_exact"]
    assert not report["project_training_executed"] and not report["final_set_accessed"]
    assert report["development"] == dev
    assert dev["checkpoint_selection"] == "EMA at exactly84168 updates; no best-of-development selection"
    scores = validate_metrics(dev, protocol)
    checkpoint = Path(report["checkpoint"])
    sidecar = json.loads(checkpoint.with_suffix(".json").read_text())
    assert sidecar["completed_updates"] == 84168
    assert sidecar["extension_contract_sha256"] == sha(contract_path)
    assert sidecar["sha256"] == report["checkpoint_sha256"] == sha(checkpoint)
    for row in dev["episodes"][:2]:
        assert (folder / row["video"]).stat().st_size > 0
    print("PUSHT NATIVE CAP OUTPUT VERIFIED; fixed development overlap reward only")
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()

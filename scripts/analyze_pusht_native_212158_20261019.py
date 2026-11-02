"""Review available native cap artifacts, explicitly report a missing summary."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "scripts/cluster/pusht_native_extension_v1"
sys.path.insert(0, str(BUNDLE))
from base_verifier import validate_metrics


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    folder = ROOT / "reports/pusht_native_212158"
    read = lambda name: json.loads((folder / name).read_text())
    sidecar, startup = read("latest.json"), read("startup_checks.json")
    provenance, dev = read("extension_provenance.json"), read("development.json")
    protocol = json.loads((BUNDLE / "training_protocol.json").read_text())
    assert sidecar["completed_updates"] == 84168
    assert sidecar["protocol_sha256"] == sha(BUNDLE / "training_protocol.json")
    assert sidecar["extension_contract_sha256"] == provenance["extension_contract_sha256"] == sha(BUNDLE / "extension_contract.json")
    assert sidecar["startup_checks"] == startup and startup["resume_checksum_verified"]
    assert all(startup["loss_audit"].values())
    original_sidecar = json.loads((ROOT / "reports/pusht_native_211309/latest.json").read_text())
    assert startup["resume_origin_sidecar"] == original_sidecar
    assert provenance["source_completed_updates"] == 20000 and provenance["original20k_source_preserved"]
    assert provenance["restored_device_checks"] and all(x["raw_and_ema_all_tensors_on_destination"] for x in provenance["restored_device_checks"])
    scores = validate_metrics(dev, protocol)
    assert scores["reference_competence_gate_passed"]
    complete = (folder / "report.json").is_file()
    if complete:
        report = read("report.json")
        assert not report["errors"] and report["completed_updates"] == 84168
        assert report["development"] == dev and report["checkpoint_sha256"] == sidecar["sha256"]
        assert report["complete_resume_state_exact"]
        assert report["protocol_sha256"] == sidecar["protocol_sha256"]
        assert report["sample_plan_sha256"] == sidecar["sample_plan_sha256"]
    result = {"available_cap_artifacts_verified_locally": True,
              "full_summary_report_received": complete,
              "missing_files": [] if complete else ["report.json"],
              "cluster_cap_verifier_passed_reported_by_user": True,
              "development_reward_series_recomputed": True, **scores,
              "checkpoint_sha256_from_sidecar": sidecar["sha256"],
              "checkpoint_loaded_or_hashed_locally": False,
              "small_file_sha256": {p.name: sha(p) for p in folder.glob("*.json")},
              "video_frames_decoded_locally": False,
              "project_training_submitted": False}
    (ROOT / "reports/pusht_native_analysis_212158_20261019.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k:v for k,v in result.items() if k != "small_file_sha256"}, indent=2))


if __name__ == "__main__":
    main()

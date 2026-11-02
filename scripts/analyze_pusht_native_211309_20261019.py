"""Review downloaded small native20k artifacts; no checkpoint/model execution."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "scripts/cluster/pusht_native_training_v1"
sys.path.insert(0, str(BUNDLE))
from verify_training import validate_metrics


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    folder = ROOT / "reports/pusht_native_211309"
    read = lambda name: json.loads((folder / name).read_text())
    report, dev, sidecar = read("report.json"), read("development.json"), read("latest.json")
    startup, repair = read("startup_checks.json"), read("device_repair.json")
    protocol = json.loads((BUNDLE / "training_protocol.json").read_text())
    assert report["training_completed"] and report["development_complete"] and not report["errors"]
    assert report["completed_updates"] == sidecar["completed_updates"] == 20000
    assert report["protocol_sha256"] == sidecar["protocol_sha256"] == sha(BUNDLE / "training_protocol.json")
    assert report["checkpoint_sha256"] == sidecar["sha256"]
    assert report["sample_plan_sha256"] == sidecar["sample_plan_sha256"]
    assert startup == sidecar["startup_checks"] and report["resume_replay_exact"]
    assert startup["disk_resume_replay_exact"] and all(startup["loss_audit"].values())
    assert startup["state_digest_before_restore"] == startup["state_digest_after_restore"]
    assert startup["loss_before_restore"] == startup["loss_after_restore"]
    assert repair["original_protocol_unchanged"] and all(x["raw_and_ema_all_tensors_on_destination"] for x in repair["restored_device_checks"])
    assert repair["repair_contract_sha256"] == sha(ROOT / "scripts/cluster/pusht_native_device_repair_v1/repair_contract.json")
    assert report["development"] == dev
    scores = validate_metrics(dev, protocol)
    assert not report["project_training_executed"] and not report["final_set_accessed"]
    for key in ("initial_backend_sha256", "normalizer_sha256", "lr_horizon_updates"):
        assert report[key] == protocol[key]
    assert report["data"]["resource_64_windows_exact"] and report["data"]["normalizer_inputs_exact"]
    videos = []
    for name in ("development_seed_5000000.mp4", "development_seed_5000001.mp4"):
        path = folder / name
        assert path.stat().st_size > 0 and path.read_bytes()[4:8] == b"ftyp"
        videos.append({"name": name, "bytes": path.stat().st_size, "sha256": sha(path)})
    result = {"full_small_artifacts_verified_locally": True,
              "cluster_full_output_verifier_pass_reported_by_user": True,
              "checkpoint_loaded_or_hashed_locally": False,
              "checkpoint_sha256_from_report_and_sidecar": sidecar["sha256"],
              "startup_complete_disk_replay_exact": True, "device_repair_record_matches": True,
              "development_reward_series_recomputed": True, **scores,
              "mean_score_bootstrap_95_interval": dev["mean_score_episode_bootstrap_95_interval"],
              "full_reward_wilson_95_interval": dev["full_reward_wilson_95_interval"],
              "videos": videos, "video_frames_decoded_locally": False,
              "small_file_sha256": {p.name: sha(p) for p in folder.iterdir() if p.suffix == ".json"},
              "next": "Prepare continuation from20000 to predeclared84168 cap; preserve full training state, one endpoint dev eval, no project/final run"}
    (ROOT / "reports/pusht_native_analysis_211309_20261019.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "small_file_sha256"}, indent=2))


if __name__ == "__main__":
    main()

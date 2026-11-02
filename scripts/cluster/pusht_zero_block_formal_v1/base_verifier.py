"""Read completed native output on CPU; no model load, training or rollout."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def validate_metrics(development, protocol):
    rows = development["episodes"]
    expected = list(range(protocol["development"]["start_seed"],
                          protocol["development"]["start_seed"] + protocol["development"]["count"]))
    assert [r["environment_seed"] for r in rows] == expected
    scores = []
    for row in rows:
        rewards = row["reward_series"]
        assert len(rewards) == row["steps"] and 1 <= len(rewards) <= 300
        assert all(0 <= r <= 1 for r in rewards)
        score = max(rewards)
        assert score == row["max_overlap_reward"] and (score == 1) == row["full_reward"]
        scores.append(score)
    mean = statistics.mean(scores)
    full = sum(score == 1 for score in scores) / len(scores)
    assert abs(mean - development["mean_score"]) < 1e-12
    assert abs(full - development["full_reward_fraction"]) < 1e-12
    gate = mean >= protocol["development"]["competence_mean_score"] and full >= protocol["development"]["competence_full_reward_fraction"]
    assert gate == development["reference_competence_gate_passed"]
    assert development["complete"] and not development["final_set_accessed"]
    return {"mean_score": mean, "full_reward_fraction": full,
            "reference_competence_gate_passed": gate}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    protocol_path = Path(__file__).with_name("training_protocol.json")
    protocol = json.loads(protocol_path.read_text())
    assert not report["errors"] and report["training_completed"] and report["development_complete"]
    assert report["completed_updates"] == 20000 and report["protocol_sha256"] == sha(protocol_path)
    assert not report["project_training_executed"] and not report["final_set_accessed"]
    assert report["data"]["resource_64_windows_exact"] and report["data"]["normalizer_inputs_exact"]
    assert report["initial_backend_sha256"] == protocol["initial_backend_sha256"]
    assert report["normalizer_sha256"] == protocol["normalizer_sha256"]
    startup = json.loads((args.report.parent / "startup_checks.json").read_text())
    if "disk_resume_replay_exact" in startup:
        assert startup["disk_resume_replay_exact"] and all(startup["loss_audit"].values())
    else:
        assert startup["resume_checksum_verified"] and all(startup["loss_audit"].values())
    development = json.loads((args.report.parent / "development.json").read_text())
    assert development == report["development"]
    scores = validate_metrics(development, protocol)
    checkpoint = Path(report["checkpoint"])
    sidecar = json.loads(checkpoint.with_suffix(".json").read_text())
    assert sidecar["sha256"] == report["checkpoint_sha256"] == sha(checkpoint)
    assert sidecar["completed_updates"] == 20000
    for row in development["episodes"][:2]:
        assert (args.report.parent / row["video"]).stat().st_size > 0
    print("PUSHT NATIVE OUTPUT VERIFIED; development overlap reward, not robot grasp success")
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()

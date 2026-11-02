"""CPU integrity review, no policy load or training; stage1 is not performance."""
import argparse
import json
from pathlib import Path
from native_data import sha
from base_verifier import validate_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    bundle = Path(__file__).resolve().parent
    contract = json.loads((bundle / "project_contract.json").read_text())
    protocol = json.loads((bundle / "base_protocol.json").read_text())
    assert not report["errors"] and report["segment_completed"]
    assert report["project_contract_sha256"] == sha(bundle / "project_contract.json")
    assert report["protocol_sha256"] == sha(bundle / "base_protocol.json")
    assert report["completed_updates"] == contract["segment_endpoints"][report["segment"] - 1]
    assert report["sample_plan_sha256"] == contract["sample_plan_sha256"]
    assert report["sample_order_sha256"] == contract["sample_order_sha256"]
    assert report["initial_backend_sha256"] == protocol["initial_backend_sha256"]
    assert report["normalizer_sha256"] == protocol["normalizer_sha256"]
    assert report["cache_report_sha256"] == contract["cache_report_sha256"]
    assert report["frozen_clip_cache_verified"] and report["frozen_text_unchanged"]
    assert not report["final_set_accessed"] and not report["clip_training_executed"]
    assert not report["trained_native_weights_loaded"] and not report["automatic_submission"]
    audit = report["startup_checks"]
    assert audit["disk_resume_replay_exact"] and all(audit["loss_audit"].values())
    checkpoint = Path(report["checkpoint"])
    sidecar = json.loads(checkpoint.with_suffix(".json").read_text())
    assert checkpoint.name == "latest.pt" and checkpoint.parent.resolve() == args.report.parent.resolve()
    assert sidecar["sha256"] == report["checkpoint_sha256"] == sha(checkpoint)
    assert sidecar["completed_updates"] == report["completed_updates"]
    assert sidecar["project_contract_sha256"] == report["project_contract_sha256"]
    assert sidecar["protocol_sha256"] == report["protocol_sha256"]
    assert sidecar["sample_plan_sha256"] == report["sample_plan_sha256"]
    assert sidecar["startup_checks"] == audit
    with (args.report.parent / "updates.jsonl").open() as stream:
        count = 0
        for count, line in enumerate(stream, 1):
            row = json.loads(line)
            assert row["completed_updates"] == report["initial_completed_updates"] + count
            assert 0 <= row["loss"] < float("inf")
        assert count == report["completed_updates"] - report["initial_completed_updates"]
    if report["segment"] == 1:
        assert not report["training_completed"] and not report["development_complete"]
        assert not (args.report.parent / "development.json").exists()
        print("PUSHT PROJECT SEGMENT1 VERIFIED; no performance verdict or automatic submission")
        print(json.dumps({"completed_updates": report["completed_updates"], "checkpoint": str(checkpoint)}, indent=2))
    else:
        assert report["training_completed"] and report["development_complete"]
        assert report["complete_resume_state_exact"]
        worker = report["evaluation_worker"]
        assert worker["all_frozen"] and worker["weights_unchanged"]
        dev = json.loads((args.report.parent / "development.json").read_text())
        assert dev == report["development"]
        scores = validate_metrics(dev, protocol)
        for row in dev["episodes"][:2]:
            assert (args.report.parent / row["video"]).stat().st_size > 0
        print("PUSHT PROJECT ENDPOINT VERIFIED; development overlap reward, not robot grasp success")
        print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()

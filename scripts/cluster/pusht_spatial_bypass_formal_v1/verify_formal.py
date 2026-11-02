"""Verify paired endpoint artifacts and recompute predeclared development decision."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from native_data import sha
from base_verifier import validate_metrics
from pair_metrics import admission


def same(a, b):
    if isinstance(a, dict):
        return set(a) == set(b) and all(same(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, float):
        return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
    return a == b


def validate(output, bundle):
    contract = json.loads((bundle / "formal_contract.json").read_text())
    for name, digest in contract["bundle_sources_sha256"].items():
        if sha(bundle / name) != digest:
            raise ValueError("Formal verifier dependency changed: " + name)
    protocol = json.loads((bundle / "protocol.json").read_text())
    base = json.loads((bundle / "base_protocol.json").read_text())
    r = json.loads((output / "report.json").read_text())
    if (not r["formal_complete"] or r["errors"] or r["final_set_accessed"] or r["clip_training_executed"]
            or r["automatic_following_submission"] or not r["matched_initial_loss_exact"] or not r["matched_lr_trace_exact"]
            or r["formal_contract_sha256"] != sha(bundle / "formal_contract.json")
            or r["preflight_protocol_sha256"] != sha(bundle / "protocol.json")
            or r["source_checkpoint_sha256"] != protocol["source_checkpoint_sha256"]
            or r["source_updates"] != 84168 or r["endpoint_updates"] != 92360 or r["new_updates_per_mode"] != 8192):
        raise ValueError("Formal endpoint incomplete, identity changed or scope violated")
    binding = json.loads((output / "preflight_binding.json").read_text())
    if (sha(output / "preflight_binding.json") != r["preflight_binding_sha256"]
            or binding["preflight_job"] != 217778 or not binding["verified"]["preflight_verified"]
            or binding["formal_contract_sha256"] != r["formal_contract_sha256"]
            or binding["preflight_report_sha256"] != r["preflight_report_sha256"]):
        raise ValueError("Preflight binding changed")
    if not (sha(output / "bypass_statistics.npz") == r["statistics_sha256"] == binding["statistics_sha256"]):
        raise ValueError("Statistics changed")
    if r["data"]["raw_training_images_sha256"] != protocol["raw_training_images_sha256"]:
        raise ValueError("Train90 pixels changed")
    if set(r["groups"]) != {"global_control", "spatial_candidate"}:
        raise ValueError("Wrong paired groups")
    developments, lr_traces, first_losses = {}, [], []
    for mode, row in r["groups"].items():
        directory = output / mode
        if json.loads((directory / "report.json").read_text()) != row:
            raise ValueError("Group report differs")
        if (not row["training_completed"] or not row["development_complete"] or not row["old_complete_restore_exact"]
                or row["preflight_weights_used"] or not row["fixed_text_and_statistics_unchanged"]
                or not row["old_learning_rate_horizon_unchanged"] or not row["ema_state_unchanged_during_evaluation"]
                or row["source_updates"] != 84168 or row["completed_updates"] != 92360 or row["new_updates"] != 8192
                or row["new_head_optimizer_step"] != 8192 or row["ema_optimization_step"] != 92360
                or not all(row["first_loss_and_gradient_audit"].values()) or row["attach"]["new_parameters"] != 262144):
            raise ValueError("Invalid group state/audit")
        worker = row["evaluation_worker"]
        if not worker["all_frozen"] or not worker["weights_unchanged"]:
            raise ValueError("Live CLIP freeze/hash failed")
        sidecar = json.loads((directory / "latest.json").read_text())
        if (Path(row["checkpoint"]).resolve() != (directory / "latest.pt").resolve()
                or not sha(directory / "latest.pt") == sidecar["sha256"] == row["checkpoint_sha256"]
                or sidecar["mode"] != mode or sidecar["completed_updates"] != 92360 or sidecar["new_updates"] != 8192
                or sidecar["formal_contract_sha256"] != r["formal_contract_sha256"]
                or sidecar["bypass_protocol_sha256"] != r["preflight_protocol_sha256"]
                or sidecar["preflight_binding_sha256"] != r["preflight_binding_sha256"]
                or sidecar["statistics_sha256"] != r["statistics_sha256"]
                or sidecar["source_checkpoint_sha256"] != r["source_checkpoint_sha256"]
                or sidecar["sample_plan_sha256"] != r["source_sample_plan_sha256"]):
            raise ValueError("Complete checkpoint/sidecar differs")
        lr_hash = hashlib.sha256()
        with (directory / "updates.jsonl").open() as stream:
            count, initial = 0, None
            for count, line in enumerate(stream, 1):
                log = json.loads(line)
                step = 84168 + count
                expected_size = min(64, 10726 - ((step - 1) % 168) * 64)
                if (log["completed_updates"] != step or not math.isfinite(log["loss"]) or log["loss"] < 0
                        or log["logical_batch_size"] != expected_size):
                    raise ValueError("Update trace changed")
                if initial is None:
                    initial = log["loss"]
                lr_hash.update(np.asarray([step], dtype="<i8").tobytes())
                lr_hash.update(np.asarray([log["lr_for_next_update"]], dtype="<f8").tobytes())
        if count != 8192 or lr_hash.hexdigest() != row["lr_trace_sha256"] or row["continuation_order_sha256"] != contract["continuation_order_sha256"]:
            raise ValueError("Continuation count/order/LR differs")
        lr_traces.append(lr_hash.hexdigest())
        first_losses.append(initial)
        dev = json.loads((directory / "development.json").read_text())
        if dev != row["development"]:
            raise ValueError("Endpoint development differs")
        validate_metrics(dev, base)
        if dev["checkpoint_selection"] != contract["endpoint_selection"]:
            raise ValueError("Wrong endpoint metadata")
        for video in dev["episodes"][:2]:
            if (directory / video["video"]).stat().st_size <= 0:
                raise ValueError("Missing fixed first2 videos")
        developments[mode] = dev
    if first_losses[0] != first_losses[1] or lr_traces[0] != lr_traces[1]:
        raise ValueError("Groups did not share initial loss/LR")
    baseline = json.loads((bundle / "unchanged_development.json").read_text())
    recomputed = admission(developments["global_control"], developments["spatial_candidate"], baseline, base, contract)
    if not same(recomputed, r["admission"]) or not same(recomputed, json.loads((output / "admission.json").read_text())):
        raise ValueError("Development decision differs")
    return {"output_verified": True, "scores": recomputed["scores"], "paired_mean_gain": recomputed["paired_mean_gain"],
            "paired_mean_gain_episode_bootstrap95": recomputed["paired_mean_gain_episode_bootstrap95"],
            "gates": recomputed["gates"], "development_admission_passed": recomputed["development_admission_passed"],
            "decision": recomputed["decision"], "scope": recomputed["scope"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = validate(args.output, Path(__file__).resolve().parent)
    print("SPATIAL BYPASS PAIR OUTPUT VERIFIED; development reward only, not a robot grasp success verdict")
    print(json.dumps(result, indent=2))

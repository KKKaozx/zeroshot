"""Verify new bypass integration artifacts; never admit policy performance."""
import argparse
import hashlib
import json
from pathlib import Path
import math
import numpy as np


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def validate(output, bundle):
    protocol = json.loads((bundle / "protocol.json").read_text())
    r = json.loads((output / "report.json").read_text())
    if (not r["preflight_passed"] or r["errors"] or r["formal_training_executed"] or r["policy_rollouts"] != 0
            or r["development_scored"] or r["sealed_final_accessed"] or r["clip_model_loaded_or_updated"]
            or r["automatic_submission"] or r["protocol_sha256"] != sha(bundle / "protocol.json")
            or r["source_checkpoint_sha256"] != protocol["source_checkpoint_sha256"]
            or not r["source_checkpoint_unchanged"] or not r["matched_first_loss_exact"]):
        raise ValueError("Bypass preflight incomplete or scope/identity changed")
    if r["statistics"]["sha256"] != sha(output / "bypass_statistics.npz"):
        raise ValueError("Changed train statistics")
    if r["statistics"]["labels_used"] or r["statistics"]["nontrain_frames_used"] or r["statistics"]["per_frame_normalization"]:
        raise ValueError("Wrong spatial statistic scope")
    with np.load(output / "bypass_statistics.npz", allow_pickle=False) as arrays:
        if len(arrays["frame_ids"]) != 11356 or len(np.unique(arrays["frame_ids"])) != 11356:
            raise ValueError("Incorrect fit frame identities")
        for mode in ("global_control", "spatial_candidate"):
            for field in ("center", "scale"):
                value = arrays[mode + "_" + field]
                if value.shape != (4, 1024) or not np.isfinite(value).all() or (field == "scale" and not (value > 0).all()):
                    raise ValueError("Invalid bypass normalization")
        for field in ("center", "scale"):
            if not np.array_equal(arrays["global_control_" + field], np.repeat(arrays["spatial_candidate_" + field][:1], 4, axis=0)):
                raise ValueError("Control does not repeat the same global statistic")
    lrs, first_losses, summary = [], [], {}
    if set(r["groups"]) != {"global_control", "spatial_candidate"}:
        raise ValueError("Wrong groups")
    for mode, g in r["groups"].items():
        if json.loads((output / mode / "report.json").read_text()) != g:
            raise ValueError("Group side report mismatch")
        for field in ("original_complete_restore_exact", "zero_raw_context_exact", "zero_ema_context_exact",
                      "zero_ddpm100_actions_exact", "rng_unchanged_at_attach", "fixed_text_unchanged",
                      "fixed_spatial_buffers_unchanged", "ema_frozen"):
            if not g[field]:
                raise ValueError("Failed " + mode + "/" + field)
        if not all(g["loss_and_gradient_audit"].values()) or not g["disk_resume"]["exact"]:
            raise ValueError("Gradient/objective/replay failed")
        if not (g["attach"]["new_parameters"] == 262144 and g["attach"]["old_parameter_objects_preserved"]
                and g["attach"]["old_optimizer_history_preserved"] and g["attach"]["new_adam_state_initially_empty"]
                and g["attach"]["new_ema_parameters_frozen"] and g["attach"]["visual_dimensions"] == 64
                and g["attach"]["output_feature_dimensions"] == 66):
            raise ValueError("Unequal or invalid integration boundary")
        replay = g["disk_resume"]
        if (replay["loss_before"] != replay["loss_after"] or replay["state_before_sha256"] != replay["state_after_sha256"]
                or Path(replay["checkpoint"]).resolve() != (output / mode / "replay.pt").resolve()
                or sha(output / mode / "replay.pt") != replay["checkpoint_sha256"]):
            raise ValueError("New complete snapshot/replay changed")
        expected_updates = protocol["preflight_updates_per_mode"]
        rows = g["updates"]
        if (g["source_updates"] != 84168 or len(rows) != expected_updates or g["completed_updates"] != 84168 + expected_updates
                or g["new_head_optimizer_step"] != expected_updates or g["ema_optimization_step"] != g["completed_updates"]
                or not g["new_raw_head_max_abs"] > 0 or not g["new_ema_head_max_abs"] > 0):
            raise ValueError("Wrong update counters or new head not changing")
        for index, row in enumerate(rows, 84169):
            if row["completed_updates"] != index or not math.isfinite(row["loss"]) or row["loss"] < 0:
                raise ValueError("Bad update log")
        lrs.append([row["lr_for_next_update"] for row in rows])
        first_losses.append(rows[0]["loss"])
        if len(g["warm_update_seconds"]) != expected_updates - 3 or not all(v > 0 and math.isfinite(v) for v in g["warm_update_seconds"]):
            raise ValueError("Invalid new-path timing")
        if not math.isclose(g["warm_mean_seconds"], sum(g["warm_update_seconds"]) / len(g["warm_update_seconds"]), rel_tol=1e-12):
            raise ValueError("Incorrect timing summary")
        summary[mode] = {"peak_reserved_gib": g["peak_reserved_gib"], "warm_update_seconds": g["warm_mean_seconds"],
                         "zero_endpoint_replay_exact": True, "new_head_gradients_and_updates_valid": True}
    if lrs[0] != lrs[1] or first_losses[0] != first_losses[1]:
        raise ValueError("Paired LR or initial loss differs")
    seconds = sum(v["warm_update_seconds"] for v in summary.values()) * protocol["planned_formal_updates_per_mode"]
    if not math.isclose(seconds, r["estimated_paired_training_seconds"], rel_tol=1e-12):
        raise ValueError("Incorrect paired runtime projection")
    return {"preflight_verified": True, "groups": summary, "estimated_paired_training_hours_excluding_rollouts_and_io": seconds / 3600,
            "formal_training_submitted": False, "policy_generalization_verdict": None}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("output", type=Path)
    args = p.parse_args()
    result = validate(args.output, Path(__file__).resolve().parent)
    print("SPATIAL BYPASS PREFLIGHT VERIFIED; no generalization verdict or automatic formal submission")
    print(json.dumps(result, indent=2))

"""Audit received frozen diagnostic artifacts; no model load, new head fit or rollout."""
import hashlib
import json
from pathlib import Path
import sys
import tarfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "scripts/cluster/pusht_spatial_diagnostic_v1"
INPUT = ROOT / "reports/pusht_spatial_217660"
OUTPUT = ROOT / "reports/pusht_spatial_analysis_217660_20261021"
sys.path.insert(0, str(BUNDLE))
from verify_output import validate


def sha(data):
    return hashlib.sha256(data).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def audit_archive():
    expected = {"report.json", "predictions.npz", "frame_manifest.json", "alignment_samples.npz", "protocol.json"}
    archive = ROOT / "reports/pusht_spatial_results_217660.tar.gz"
    with tarfile.open(archive, "r:gz") as t:
        members = t.getmembers()
        if len(members) != 5 or {m.name for m in members} != expected or not all(m.isfile() for m in members):
            raise ValueError("Unexpected archive entries")
        payload = {m.name: t.extractfile(m).read() for m in members}
    if payload["protocol.json"] != (BUNDLE / "protocol.json").read_bytes():
        raise ValueError("Received protocol differs from immutable local release")
    INPUT.mkdir(exist_ok=True)
    for name, data in payload.items():
        path = INPUT / name
        if path.exists() and path.read_bytes() != data:
            raise ValueError("Existing extracted file differs: " + name)
        if not path.exists():
            path.write_bytes(data)
    return {"archive_sha256": sha(archive.read_bytes()), "files": {n: {"bytes": len(d), "sha256": sha(d)} for n, d in payload.items()},
            "members_safe_and_exact": True, "protocol_byte_identical_to_release": True}


def comparison_counts(report, left, right):
    a = report["readouts"][left]["heldout"]["per_episode"]
    b = report["readouts"][right]["heldout"]["per_episode"]
    ids = sorted(a, key=int)
    delta = np.array([np.asarray(b[k]) - np.asarray(a[k]) for k in ids])
    return {"episodes": len(ids), "left": left, "right": right,
            "metric_order": ["agent_position_sim_pixels", "block_position_sim_pixels", "block_angle_deg"],
            "right_worse_counts": (delta > 1e-9).sum(0).tolist(),
            "right_better_counts": (delta < -1e-9).sum(0).tolist(),
            "right_equal_counts": (np.abs(delta) <= 1e-9).sum(0).tolist(),
            "right_both_block_metrics_worse_count": int(((delta[:, 1] > 0) & (delta[:, 2] > 0)).sum()),
            "per_episode_delta_right_minus_left": {ep: value.tolist() for ep, value in zip(ids, delta)}}


def render_alignment(samples, predictions, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Polygon
    import cv2
    images, state, frames, episodes, roles = [samples[k] for k in ("raw_images", "state", "frame_ids", "episode_ids", "roles")]
    if images.shape != (24, 96, 96, 3) or state.shape != (24, 5):
        raise ValueError("Alignment sample shape differs")
    all_ids = predictions["frame_ids"]
    frame_lookup = {int(value): i for i, value in enumerate(all_ids)}
    for i, frame in enumerate(frames):
        row = frame_lookup[int(frame)]
        if not (np.array_equal(state[i], predictions["source_state"][row]) and episodes[i] == predictions["episode_ids"][row]
                and (roles[i] == "fit") == predictions["fit_mask"][row]):
            raise ValueError("Alignment sample does not match saved prediction frame")
    vertices = (np.array([[-60, 30], [60, 30], [60, 0], [-60, 0]], dtype=float),
                np.array([[-15, 30], [-15, 120], [15, 120], [15, 30]], dtype=float))
    details = []
    for i in range(len(images)):
        image, pose = images[i], state[i]
        blue = (image[..., 2] > 150) & (image[..., 0] < 130) & ((image[..., 2] - image[..., 0]) > 60) & ((image[..., 2] - image[..., 1]) > 25)
        yy, xx = np.where(blue)
        nearest = float(np.linalg.norm(np.column_stack((xx, yy)) - pose[:2] * 96 / 512, axis=1).min()) if len(xx) else None
        c, s = np.cos(pose[4]), np.sin(pose[4])
        rotation = np.array([[c, -s], [s, c]])
        polys = [(v @ rotation.T + pose[2:4]) * 96 / 512 for v in vertices]
        expected = np.zeros((96, 96), np.uint8)
        cv2.fillPoly(expected, [np.rint(v).astype(np.int32) for v in polys], 1)
        observed = (np.linalg.norm(image - np.array([119, 136, 153]), axis=2) < 42).astype(np.uint8)
        dil_expected = cv2.dilate(expected, np.ones((3, 3), np.uint8))
        dil_observed = cv2.dilate(observed, np.ones((3, 3), np.uint8))
        details.append({"frame_id": int(frames[i]), "episode": int(episodes[i]), "role": str(roles[i]),
                        "agent_label_to_nearest_blue_pixel_image_pixels": nearest,
                        "observed_gray_near_expected_polygon_fraction": float((observed * dil_expected).sum() / max(observed.sum(), 1)),
                        "expected_polygon_near_observed_gray_fraction": float((expected * dil_observed).sum() / max(expected.sum(), 1))})
    for offset, role in ((0, "fit"), (12, "heldout")):
        fig, axes = plt.subplots(3, 4, figsize=(12, 9.5))
        for j, ax in enumerate(axes.flat):
            i = offset + j
            pose = state[i]
            ax.imshow(images[i] / 255., interpolation="nearest")
            ax.add_patch(Circle(pose[:2] * 96 / 512, 2.6, fill=False, color="cyan", linewidth=.9))
            c, s = np.cos(pose[4]), np.sin(pose[4])
            rotation = np.array([[c, -s], [s, c]])
            for v in vertices:
                ax.add_patch(Polygon((v @ rotation.T + pose[2:4]) * 96 / 512, fill=False, edgecolor="magenta", linewidth=.8))
            ax.plot(*(pose[2:4] * 96 / 512), marker="+", color="magenta", markersize=5)
            ax.set(xlim=(-.5, 95.5), ylim=(95.5, -.5), title="ep%d frame%d | %.1f deg" % (episodes[i], frames[i], np.degrees(pose[4])))
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle("Fixed %s alignment samples | cyan: agent label | magenta: T geometry from state" % role)
        fig.tight_layout()
        fig.savefig(output / ("alignment_" + role + ".png"), dpi=140)
        plt.close(fig)
    return {"samples_verified_against_saved_prediction_rows": 24,
            "selection": "All12 preselected frames of first fit and first heldout episode; no outcome selection",
            "geometry": "Pinned author T vertices rotated by saved angle and translated by saved body position;512->96 image coordinates",
            "limits": "Qualitative label-overlay QA on24 sampled frames only; loose color/1pixel-neighborhood checks are descriptive, not full-dataset alignment certification",
            "samples": details}


def render_comparison(report, counts, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    stages = ["clip_spatial", "adapter8_spatial", "clip_pool", "adapter1_pool", "adapter4_pool", "adapter8_pool", "projection64", "native_rgb64"]
    names = ["CLIP spatial", "Adapter8 spatial", "CLIP pool", "Adapter1 pool", "Adapter4 pool", "Adapter8 pool", "Projection64", "Native RGB64"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, metric, title in zip(axes, ("block_position_sim_pixels", "block_angle_deg"), ("T position error (512-canvas pixels)", "T angle error (degrees)")):
        x = np.arange(len(stages))
        fit_values = [report["readouts"][s]["fit"]["episode_equal"][metric] for s in stages]
        held_values = [report["readouts"][s]["heldout"]["episode_equal"][metric] for s in stages]
        ax.bar(x - .18, fit_values, .36, color="#7bb5d8", label="Diagnostic fit90 episodes")
        ax.bar(x + .18, held_values, .36, color="#d78255", label="Diagnostic heldout40 episodes")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("Frozen spatial readouts | independent heads, no policy updates | dimensions differ across families")
    fig.tight_layout()
    fig.savefig(output / "fit_vs_heldout.png", dpi=160)
    plt.close(fig)
    data = counts["clip_spatial__to__adapter8_spatial"]
    delta = np.array(list(data["per_episode_delta_right_minus_left"].values()))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for i, ax in enumerate(axes):
        ax.bar(np.arange(40), delta[:, i + 1], color=np.where(delta[:, i + 1] > 0, "#bf625b", "#438bb2"))
        ax.axhline(0, color="black", linewidth=.7)
        ax.set_xlabel("Heldout episode (numeric order)")
        ax.set_ylabel(("Position delta (sim pixels)", "Angle delta (degrees)")[i])
    fig.suptitle("Adapter8 spatial minus CLIP spatial | same4096 dimensions and readout recipe | positive = worse")
    fig.tight_layout()
    fig.savefig(output / "matched_spatial_episode_deltas.png", dpi=160)
    plt.close(fig)


def main():
    OUTPUT.mkdir(exist_ok=True)
    archive = audit_archive()
    verified = validate(INPUT, BUNDLE / "protocol.json")
    report = json.loads((INPUT / "report.json").read_text())
    terminal = json.loads((ROOT / "reports/pusht_spatial_217660_terminal.json").read_text())
    for name, values in terminal["heldout_readouts"].items():
        actual = [report["readouts"][name]["heldout"]["episode_equal"][k] for k in terminal["columns"]]
        if not np.array_equal(values, actual):
            raise ValueError("Received metrics disagree with reported terminal: " + name)
    counts = {key: comparison_counts(report, *key.split("__to__")) for key in report["comparisons"]}
    tables = {name: {"dimensions": row.get("dimensions"),
                    "fit": row["fit"]["episode_equal"], "heldout": row["heldout"]["episode_equal"],
                    "shuffled_heldout": row.get("shuffled_heldout", {}).get("episode_equal")}
              for name, row in report["readouts"].items()}
    with np.load(INPUT / "predictions.npz", allow_pickle=False) as predictions, np.load(INPUT / "alignment_samples.npz", allow_pickle=False) as samples:
        alignment = render_alignment(samples, predictions, OUTPUT)
    render_comparison(report, counts, OUTPUT)
    result = {"local_output_verified": True, "archive": archive, "original_verifier": verified,
              "reported_terminal_metrics_exact": True, "inputs": report["inputs"], "tables": tables,
              "paired_episode_counts": counts, "paired_bootstrap_intervals": report["comparisons"],
              "alignment_review": alignment, "policy_updates_in_analysis": 0, "new_readout_fits": 0,
              "model_checkpoints_loaded": False, "policy_rollouts": 0, "sealed_final_accessed": False,
              "old_spatial_branch": {"job": 210334, "status": "closed_failed_admission",
                    "intervention": "Post-Adapter scalar softmax weighted pooling with1026 new scoring parameters, collapsed to one1024 context",
                    "difference": "Current frozen diagnostic compares matched spatial moments before/after Adapter and does not retest that weighted pooling",
                    "do_not_repeat_closed_branch": True},
              "interpretation": [
                  "Matched4096-dimensional spatial readouts worsen in position in38/40 episodes and angle in36/40 after Adapter8;34/40 worsen in both.",
                  "Both fit and heldout linear readout errors worsen. This is not solely a heldout-only readout failure; it says nothing new about original policy training-set fit.",
                  "Correct CLIP spatial labels outperform shuffled-fit-label heads and train means. Labels are genuinely linearly accessible under this recipe.",
                  "Within1024-dimensional pooled family, deterioration is concentrated in the sampled first-to-fourth layers; fourth-to-eighth changes are small. Do not claim every layer worsens.",
                  "Output LayerNorm improves mean position readout slightly; it is not the leading suspect from these results.",
                  "Projection64 position readout worsens in40/40 versus Adapter8 pool and is worse than native RGB64 in40/40; dimensionality/probe capacity and training objectives prevent unique causal attribution.",
                  "Native policy has stronger rollout reward despite weaker angle readout than CLIP spatial, so readout rankings cannot substitute for closed-loop competence.",
                  "Evidence supports preserving an explicit spatial route from frozen CLIP alongside semantic fusion as the next intervention question, not unfreezing CLIP or blindly increasing training budget."],
              "limits": report["limitations"]}
    save(OUTPUT / "report.json", result)
    save(OUTPUT / "verification.json", {"passed": True, "received_files": len(archive["files"]),
         "protocol_exact": True, "terminal_metrics_exact": True, "all_original_numerical_checks_passed_locally": True,
         "alignment_sample_identity_verified": True, "figures_generated": 4})
    (OUTPUT / "analysis_source_snapshot.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps({"local_output_verified": True, "analysis": str(OUTPUT),
                      "counts": {k: {field: v[field] for field in ("right_worse_counts", "right_both_block_metrics_worse_count")} for k, v in counts.items()},
                      "alignment": alignment["samples"]}, indent=2))


if __name__ == "__main__":
    main()

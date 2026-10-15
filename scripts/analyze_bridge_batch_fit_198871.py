"""Read saved cluster predictions; independently score in float64 without Torch/GPU."""
import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "reports/bridge-batch-fit-v2-198871"
OUT = ROOT / "reports/bridge_batch_analysis_198871_20261012"
FIELDS = ("path_position_cm", "path_rotation_deg", "endpoint_position_cm", "endpoint_rotation_deg")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def errors(pred, target):
    a, b = pred[..., 3:7].astype(float), target[None, ..., 3:7].astype(float)
    assert np.isfinite(pred).all() and np.isfinite(target).all()
    assert np.linalg.norm(a, axis=-1).min() > 1e-8
    assert np.linalg.norm(b, axis=-1).min() > 1e-8
    a /= np.linalg.norm(a, axis=-1, keepdims=True)
    b /= np.linalg.norm(b, axis=-1, keepdims=True)
    return (np.linalg.norm(pred[..., :3].astype(float) - target[None, ..., :3], axis=-1) * 10,
            np.degrees(2 * np.arccos(np.clip(np.abs((a*b).sum(-1)), 0, 1))))


def score(p, r):
    return dict(zip(FIELDS, map(float, (p.mean(), r.mean(), p[..., -1].mean(), r[..., -1].mean()))))


def episode(row):
    return row["shard"] + "::" + str(row["record_index"])


def write_csv(name, rows):
    with (OUT/name).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(exist_ok=True)
    report = json.loads((DATA/"report.json").read_text())
    manifest = json.loads((DATA/"plan.json").read_text())["manifest"]
    frozen_manifest = ROOT/"cluster_packages/bridge_batch_control_v2/manifest.json"
    assert report["manifest_sha256"] == sha(frozen_manifest)
    assert manifest == json.loads(frozen_manifest.read_text())
    assert report["execution_checks_passed"] and report["input_contract_passed"]
    assert report["clip_unchanged"] and report["production_source_unchanged"]
    assert len(report["runs"]) == 6 and not report["gripper_trained"] and not report["checkpoints_saved"]
    assert report["source_sha256"] == manifest["source_sha256"]
    for name, digest in report["source_sha256"].items():
        assert sha(ROOT/"code"/name) == sha(ROOT/"cluster_packages/bridge_batch_control_v2"/name) == digest
    assert sha(DATA/"runner_snapshot.py") == report["source_sha256"]["check_bridge_batch_control.py"]
    assert not set(map(episode, manifest["train"])) & set(map(episode, manifest["development"]))
    reference_dir = ROOT/"reports/bridge_batch_input_reference_20261012"
    for name, digest in manifest["input_reference_files_sha256"].items():
        assert sha(reference_dir/name) == digest
    reference = np.load(reference_dir/"arrays.npz")["targets"]
    metadata = json.loads((reference_dir/"metadata.json").read_text())
    assert len(report["data_checks"]) == len(metadata["raw_inputs"]) == 72
    assert all(c["raw_identity"] == raw for c, raw in zip(report["data_checks"], metadata["raw_inputs"]))
    paired_hashes, targets_by_split, values = {}, {}, {}
    windows, episodes, tasks, summaries, metric_diffs, artifact_hashes = [], [], [], [], [], {}
    max_reference_difference = 0.0
    for run in report["runs"]:
        init, arm = run["initialization_seed"], run["arm"]
        assert run["updates"] == 2048 and run["actual_draw_slots_per_window"] == [512]*32
        assert run["frozen_other_unchanged"] and all(run["changed"].values()) and all(run["first_gradients"].values())
        if init in paired_hashes:
            assert paired_hashes[init] == run["initial_shared_sha256"]
        paired_hashes[init] = run["initial_shared_sha256"]
        for split in ("train", "development"):
            rows = manifest[split]
            path = DATA/f"seed{init}_{arm}_{split}.npz"
            artifact_hashes[path.name] = sha(path)
            with np.load(path) as z:
                pred, target, indices = z["predictions"], z["targets"], z["row_indices"]
            expected = np.arange(32) if split == "train" else np.arange(32, 72)
            assert np.array_equal(indices, expected) and pred.shape == (3, len(rows), 16, 8)
            assert target.shape == (len(rows), 16, 8)
            if split in targets_by_split:
                assert np.array_equal(target, targets_by_split[split])
            targets_by_split[split] = target
            delta = float(np.abs(target.astype(float)-reference[expected]).max())
            max_reference_difference = max(max_reference_difference, delta)
            assert delta <= 1e-6 and np.array_equal(target[..., 7], reference[expected, ..., 7])
            p, r = errors(pred, target)
            static = np.zeros_like(target); static[..., 6] = 1
            sp, sr = errors(static[None], target)
            s = score(p, r); baseline = score(sp, sr)
            seed_pos, seed_rot = p.mean((1, 2)), r.mean((1, 2))
            good = ((p.mean(-1) <= .5) & (r.mean(-1) <= 5)).all(0)
            gate = bool((seed_pos <= .5).all() and (seed_rot <= 5).all() and good.mean() >= .9
                        and (p[..., -1].mean(1) <= 1).all() and (r[..., -1].mean(1) <= 10).all())
            old = run["final"][split]
            for key in FIELDS:
                diff = abs(s[key]-old[key]); metric_diffs.append((key, diff))
                assert diff < (.02 if "rotation" in key else 1e-5), (init, arm, split, key, diff)
                assert abs(baseline[key]-old["static_"+key]) < (.02 if "rotation" in key else 1e-5)
            if split == "train":
                assert int(good.sum()) == old["fit_gate"]["windows_all_three_seeds_pass"]
                assert gate == old["fit_gate"]["pose_fit_passed"]
            spread_p, spread_r = [], []
            for a, b in itertools.combinations(range(3), 2):
                pp, rr = errors(pred[a:a+1], pred[b]); spread_p.append(pp[0]); spread_r.append(rr[0])
            spread_p, spread_r = np.mean(spread_p, 0), np.mean(spread_r, 0)
            # XYZ MSE decomposes into bias of the mean sample and spread around it.
            # This describes saved samples; it is not a new mean-output policy.
            xyz = pred[..., :3].astype(float)*10
            xyz_mean = xyz.mean(0)
            bias_mse = float(np.square(xyz_mean-target[..., :3]*10).sum(-1).mean())
            spread_mse = float(np.square(xyz-xyz_mean[None]).sum(-1).mean())
            total_mse = float(np.square(xyz-target[None, ..., :3]*10).sum(-1).mean())
            assert np.isclose(bias_mse+spread_mse, total_mse)
            summary = dict(initialization=init, arm=arm, split=split, **s,
                windows_passing_all_sampling_seeds=int(good.sum()), fit_gate=gate if split == "train" else None,
                sampling_position_mean_cm=seed_pos.tolist(), sampling_rotation_mean_deg=seed_rot.tolist(),
                window_position_range_cm=[float(p.mean((0, 2)).min()), float(p.mean((0, 2)).max())],
                mean_pairwise_sample_distance_cm=float(spread_p.mean()),
                mean_pairwise_sample_rotation_deg=float(spread_r.mean()),
                xyz_sample_spread_fraction_of_mse=spread_mse/total_mse,
                xyz_bias_mse_cm2=bias_mse, xyz_spread_mse_cm2=spread_mse,
                mean_signed_xyz_error_cm=(xyz-target[None, ..., :3]*10).mean((0, 1, 2)).tolist(),
                horizon_signed_xyz_error_cm=(xyz-target[None, ..., :3]*10).mean((0, 1)).tolist(),
                static=baseline, horizon_position_cm=p.mean((0, 1)).tolist(),
                horizon_rotation_deg=r.mean((0, 1)).tolist())
            summaries.append(summary)
            values[init, arm, split] = (pred, target, p, r, sp, sr)
            for i, row in enumerate(rows):
                windows.append(dict(initialization=init, arm=arm, split=split, row_index=int(indices[i]),
                    task=row["task"], episode=episode(row), start=row["start_index"], **score(p[:, i], r[:, i]),
                    pass_all_samples=bool(good[i]), static_position_cm=float(sp[:, i].mean()),
                    sample_distance_cm=float(spread_p[i].mean()),
                    sample_rotation_deg=float(spread_r[i].mean()),
                    position_1101_cm=float(p[0, i].mean()), position_1102_cm=float(p[1, i].mean()),
                    position_1103_cm=float(p[2, i].mean())))
            for grouping, output in [(episode, episodes), (lambda row: row["task"], tasks)]:
                for group in dict.fromkeys(map(grouping, rows)):
                    ix = [i for i, row in enumerate(rows) if grouping(row) == group]
                    g, sb = score(p[:, ix], r[:, ix]), score(sp[:, ix], sr[:, ix])
                    entry = dict(initialization=init, arm=arm, split=split, group=group,
                        task=rows[ix[0]]["task"], windows=len(ix), **g,
                        **{"static_"+k: v for k, v in sb.items()},
                        position_beats_static=g["path_position_cm"] < sb["path_position_cm"],
                        rotation_beats_static=g["path_rotation_deg"] < sb["path_rotation_deg"],
                        pass_windows=int(good[ix].sum()))
                    output.append(entry)
                    if output is episodes:
                        previous = next(e for e in old["per_episode"] if e["episode"] == group)
                        for key in FIELDS:
                            diff = abs(g[key]-previous[key]); metric_diffs.append((key, diff))
                            assert diff < (.02 if "rotation" in key else 1e-5)
    assert len(set(paired_hashes.values())) == 3
    write_csv("windows.csv", windows); write_csv("episodes.csv", episodes); write_csv("tasks.csv", tasks)
    comparisons = []
    for split in ("train", "development"):
        pp43, pp44 = values[43, "mixed", split][2], values[44, "mixed", split][2]
        for i, row in enumerate(manifest[split]):
            comparisons.append(dict(split=split, row_index=i if split == "train" else i+32,
                task=row["task"], episode=episode(row), start=row["start_index"],
                seed43_minus44_position_cm=float(pp43[:, i].mean()-pp44[:, i].mean())))
    verification = dict(passed=True, data_dir=str(DATA), no_training=True, no_gpu=True,
        original_report_sha256=sha(DATA/"report.json"), manifest_sha256=report["manifest_sha256"],
        source_snapshot_and_current_source_verified=True, all_12_targets_identical_within_split=True,
        exact_raw_inputs_matched_to_reference=True, max_reference_target_difference=max_reference_difference,
        train_development_episodes_disjoint=True, paired_initializations_equal=True,
        initialization_hashes=paired_hashes, actual_budget_per_window=512, artifact_sha256=artifact_hashes,
        training_windows_seed43_worse_than44=sum(c["seed43_minus44_position_cm"] > 0 for c in comparisons if c["split"] == "train"),
        max_position_metric_difference_cm=max(d for k, d in metric_diffs if "position" in k),
        max_rotation_metric_difference_deg=max(d for k, d in metric_diffs if "rotation" in k),
        runs=summaries, seed43_minus44=comparisons,
        limits=["Only three training initializations and three sampling seeds; no causal layer attribution.",
               "Batch organization changes both diversity and sample temporal ordering.",
               "Only final predictions saved; no trained checkpoints or intermediate learning curves.",
               "Development has 10 previously used episodes; overlapping windows are not independent episodes.",
               "Pose-only native DDPM diagnostic; no gripper training or simulated task success."])
    (OUT/"analysis.json").write_text(json.dumps(verification, indent=2)+"\n")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for seed, color in [(42, "#2c7fb8"), (43, "#d95f0e"), (44, "#238b45")]:
        p, r = values[seed, "mixed", "train"][2:4]
        axes[0].plot(range(1, 17), p.mean((0, 1))*10, marker=".", color=color, label=f"Init {seed}")
        axes[1].plot(range(1, 17), r.mean((0, 1)), marker=".", color=color, label=f"Init {seed}")
    for ax, label in zip(axes, ("Position error (mm)", "Rotation error (degrees)")):
        ax.axhline(5, ls="--", color="gray", label="5 mm" if ax is axes[0] else "5 deg")
        ax.set(xlabel="Prediction horizon (1-16)", ylabel=label); ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle("Mixed batches: train errors across the trajectory (3-sample means)")
    fig.tight_layout(); fig.savefig(OUT/"train_horizon.png", dpi=145); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
    task_names = list(dict.fromkeys(row["task"] for row in manifest["train"]))
    order = sorted(range(32), key=lambda i: (task_names.index(manifest["train"][i]["task"]), i))
    labels = [f"row {i:02d} / t{manifest['train'][i]['start_index']}" for i in order]
    for ax, seed in zip(axes, (42, 43, 44)):
        p = values[seed, "mixed", "train"][2].mean(-1)[:, order]*10
        im = ax.imshow(p.T, vmin=0, vmax=35, cmap="YlOrRd", aspect="auto")
        ax.set_title(f"Init {seed}"); ax.set_xticks(range(3), ("1101", "1102", "1103"))
        ax.set_xlabel("Sampling seed"); ax.set_yticks(range(32), labels, fontsize=7)
    fig.suptitle("Same 32 training windows; tasks grouped, row IDs retained")
    fig.subplots_adjust(left=.11, right=.82, top=.88, bottom=.12, wspace=.1)
    cax = fig.add_axes([.855, .18, .018, .64])
    fig.colorbar(im, cax=cax, label="Window-mean position error (mm), capped at 35")
    fig.savefig(OUT/"train_window_samples.png", dpi=145); plt.close(fig)
    dev_groups = list(dict.fromkeys(map(episode, manifest["development"])))
    fig, axes = plt.subplots(1, 2, figsize=(11, 6), sharey=True)
    for j, field in enumerate(("path_position_cm", "path_rotation_deg")):
        deltas = np.array([[next(e for e in episodes if e["initialization"] == seed and e["arm"] == "mixed"
            and e["split"] == "development" and e["group"] == ep)[field] -
            next(e for e in episodes if e["initialization"] == seed and e["arm"] == "mixed"
            and e["split"] == "development" and e["group"] == ep)["static_"+field]
            for seed in (42, 43, 44)] for ep in dev_groups])
        limit = max(abs(deltas.min()), abs(deltas.max()))
        axes[j].imshow(deltas, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
        for (row, col), val in np.ndenumerate(deltas):
            axes[j].text(col, row, f"{val:+.2f}", ha="center", va="center", fontsize=9,
                color="white" if abs(val) > .65*limit else "black")
        axes[j].set_xticks(range(3), ("Init42", "Init43", "Init44"))
        axes[j].set_title("Model minus static: "+("cm" if j == 0 else "degrees"))
    axes[0].set_yticks(range(10), [f"{e.split('tfrecord-')[1].split('-of-')[0]}::{e.split('::')[1]}" for e in dev_groups])
    fig.suptitle("10 held-out episodes: blue improves, red worsens; all native DDPM100")
    fig.tight_layout(); fig.savefig(OUT/"development_episode_deltas.png", dpi=145); plt.close(fig)
    print("VERIFIED", verification["max_position_metric_difference_cm"], verification["max_rotation_metric_difference_deg"])
    for s in summaries:
        print(s["initialization"], s["arm"], s["split"], "error", round(s["path_position_cm"], 4),
            round(s["path_rotation_deg"], 4), "pass", s["windows_passing_all_sampling_seeds"],
            "spread_cm", round(s["mean_pairwise_sample_distance_cm"], 5),
            "spread_MSE_fraction", round(s["xyz_sample_spread_fraction_of_mse"], 6))
    for e in tasks:
        if e["arm"] == "mixed" and e["split"] == "train":
            print("TASK", e["initialization"], e["task"], round(e["path_position_cm"]*10, 3),
                round(e["path_rotation_deg"], 3), "pass", e["pass_windows"], "/", e["windows"])
    for seed in (42, 43, 44):
        es = [e for e in episodes if e["arm"] == "mixed" and e["split"] == "development" and e["initialization"] == seed]
        print("DEV_EPISODES", seed, "position wins", sum(e["position_beats_static"] for e in es),
            "rotation wins", sum(e["rotation_beats_static"] for e in es),
            "joint wins", sum(e["position_beats_static"] and e["rotation_beats_static"] for e in es))


if __name__ == "__main__":
    main()

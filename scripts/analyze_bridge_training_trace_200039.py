"""Verify and analyze saved training traces without running Torch or training."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "reports/bridge-trace-fit-200039"
OUT = ROOT / "reports/bridge_trace_analysis_200039_20261012"
FIELDS = ("path_position_cm", "path_rotation_deg", "endpoint_position_cm", "endpoint_rotation_deg")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def errors(pred, target):
    assert np.isfinite(pred).all() and np.isfinite(target).all()
    a, b = pred[..., 3:7].astype(float), target[None, ..., 3:7].astype(float)
    assert np.linalg.norm(a, axis=-1).min() > 1e-8
    assert np.linalg.norm(b, axis=-1).min() > 1e-8
    a /= np.linalg.norm(a, axis=-1, keepdims=True)
    b /= np.linalg.norm(b, axis=-1, keepdims=True)
    return (np.linalg.norm(pred[..., :3].astype(float)-target[None, ..., :3], axis=-1)*10,
            np.degrees(2*np.arccos(np.clip(np.abs((a*b).sum(-1)), 0, 1))))


def write_csv(name, rows):
    with (OUT/name).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(exist_ok=True)
    report = json.loads((DATA/"report.json").read_text())
    manifest = json.loads((DATA/"plan.json").read_text())["manifest"]
    package = ROOT/"cluster_packages/bridge_training_trace_v1"
    origin_dir = ROOT/"reports/bridge-batch-fit-v2-198871"
    origin = json.loads((origin_dir/"report.json").read_text())
    assert sha(package/"manifest.json") == report["manifest_sha256"]
    assert json.loads((package/"manifest.json").read_text()) == manifest
    assert all(report[k] for k in ("execution_checks_passed", "input_contract_passed", "clip_unchanged", "production_source_unchanged"))
    assert not report["gripper_trained"] and report["sampling_steps"] == 100
    assert report["sampling_seeds"] == [1101, 1102, 1103]
    assert sha(DATA/"runner_snapshot.py") == report["source_sha256"]["trace_bridge_batch_training.py"]
    assert all(sha(package/n) == h for n, h in report["source_sha256"].items())
    assert origin["data_checks"] == report["data_checks"]
    assert len(report["data_checks"]) == 72
    assert all(row["raw_identity_exact"] for row in report["input_numeric_comparisons"])
    assert all(v["max_abs"] <= v["atol"] for row in report["input_numeric_comparisons"] for v in row["differences"].values())
    identity = lambda row: (row["shard"], row["record_index"])
    assert not set(map(identity, manifest["train"])) & set(map(identity, manifest["development"]))
    history, blocks, windows, runs, hashes = [], [], [], [], {}
    max_metric_delta = {key: 0.0 for key in FIELDS}
    labels = {}
    curves = {}
    for run in report["runs"]:
        seed = run["initialization_seed"]
        old = next(r for r in origin["runs"] if r["initialization_seed"] == seed and r["arm"] == "mixed")
        assert run["initial_shared_sha256"] == old["initial_shared_sha256"]
        assert run["loss_first"] == old["loss_first"]
        assert run["updates"] == 2048 and run["actual_draw_slots_per_window"] == [512]*32
        assert run["frozen_clip_and_other_unchanged"] and all(run["changed"].values())
        assert all(v == 0 for v in run["origin_final_metric_differences"].values())
        log = DATA/f"seed{seed}_observed_updates.jsonl"
        entries = [json.loads(line) for line in log.read_text().splitlines()]
        assert [r["update"] for r in entries] == list(range(1, 2049))
        assert entries[0]["loss"] == run["loss_first"] and entries[-1]["loss"] == run["loss_last"]
        grad_names = list(run["gradient_summary"])
        for prefix in grad_names:
            values = np.asarray([r["unclipped_module_gradient_norms"][prefix] for r in entries])
            assert np.isfinite(values).all() and (values > 0).all()
            assert values.min() == run["gradient_summary"][prefix]["min"]
            assert values.max() == run["gradient_summary"][prefix]["max"]
        assert np.isfinite([r["loss"] for r in entries]).all()
        curves[seed] = entries
        for lo, hi in [(1, 64), (65, 256), (257, 512), (513, 1024), (1025, 1536), (1537, 2048), (1921, 2048)]:
            segment = entries[lo-1:hi]
            blocks.append(dict(seed=seed, first_update=lo, last_update=hi,
                mean_loss=float(np.mean([r["loss"] for r in segment])),
                clipped_fraction=float(np.mean([r["gradient_clip_scale"] < 1 for r in segment])),
                max_global_gradient=max(r["global_unclipped_gradient_norm"] for r in segment),
                median_global_gradient=float(np.median([r["global_unclipped_gradient_norm"] for r in segment])),
                **{prefix.rstrip(".")+"_median_gradient":float(np.median([r["unclipped_module_gradient_norms"][prefix] for r in segment])) for prefix in grad_names}))
        assert [h["update"] for h in run["history"]] == [0, 64, 256, 512, 1024, 1536, 2048]
        pass_updates = []
        means = {}
        for update, split, reported in [(h["update"], "train", h["train"]) for h in run["history"]] + [(2048, "development", run["final"]["development"])]:
            file = DATA/f"seed{seed}_u{update:04d}_{split}.npz"
            hashes[file.name] = sha(file)
            with np.load(file) as z:
                pred, target, indices, contexts = (z[n] for n in ("predictions", "targets", "row_indices", "contexts"))
            count = 32 if split == "train" else 40
            assert pred.shape == (3, count, 16, 8) and contexts.shape == (count, 1024)
            assert np.isfinite(contexts).all()
            assert np.array_equal(indices, np.arange(count) if split == "train" else np.arange(32, 72))
            if split in labels: assert np.array_equal(target, labels[split])
            else: labels[split] = target.copy()
            p, r = errors(pred, target)
            metrics = dict(zip(FIELDS, map(float, (p.mean(), r.mean(), p[..., -1].mean(), r[..., -1].mean()))))
            for key in FIELDS:
                delta = abs(metrics[key]-reported[key])
                max_metric_delta[key] = max(max_metric_delta[key], delta)
                assert delta < (0.02 if "rotation" in key else 1e-4)
            good = ((p.mean(-1) <= .5) & (r.mean(-1) <= 5)).all(0)
            gate = bool(good.mean() >= .9 and (p.mean((1, 2)) <= .5).all() and (r.mean((1, 2)) <= 5).all()
                        and (p[..., -1].mean(1) <= 1).all() and (r[..., -1].mean(1) <= 10).all())
            if split == "train":
                assert good.sum() == reported["fit_gate"]["windows_all_three_seeds_pass"]
                assert gate == reported["fit_gate"]["pose_fit_passed"]
                if gate: pass_updates.append(update)
                means[update] = p.mean((0, 2))
            history.append(dict(seed=seed, update=update, split=split, **metrics,
                                windows_passing=int(good.sum()), pose_fit_passed=gate))
            for i, row in enumerate(manifest[split]):
                windows.append(dict(seed=seed, update=update, split=split, window=i,
                    episode=row["shard"]+"::"+str(row["record_index"]), task=row["task"],
                    path_position_cm=float(p[:, i].mean()), path_rotation_deg=float(r[:, i].mean()),
                    passes_all_sampling_seeds=bool(good[i])))
            if update == 2048:
                with np.load(origin_dir/f"seed{seed}_mixed_{split}.npz") as z:
                    assert np.array_equal(pred, z["predictions"]) and np.array_equal(target, z["targets"])
        runs.append(dict(seed=seed, sampled_updates_passing_gate=pass_updates,
            final_predictions_exactly_match_origin=True,
            windows_position_worse_2048_than_1024=int((means[2048] > means[1024]).sum()),
            final_position_ratio_to_1024=float(means[2048].mean()/means[1024].mean()),
            all_logged_module_gradients_finite_nonzero=True,
            late_updates_clipped=int(sum(r["gradient_clip_scale"] < 1 for r in entries[512:]))))
    write_csv("history.csv", history)
    write_csv("gradient_blocks.csv", blocks)
    write_csv("windows.csv", windows)
    analysis = dict(execution_checks_verified=True, report_sha256=sha(DATA/"report.json"),
        manifest_sha256=report["manifest_sha256"], input_identity_matches_origin=True,
        all_three_pass_at_sampled_update_1024=True, max_float64_metric_difference=max_metric_delta,
        runs=runs, artifact_hashes=hashes,
        limitations=["Monitoring is sparse; no exact onset of deterioration.",
                    "Intermediate checkpoints were not saved; only predictions and contexts.",
                    "Development evaluated only at 2048; no intermediate generalization evidence.",
                    "Nonzero finite gradients do not identify a unique faulty module.",
                    "Gripper frozen, no robot or simulation task success measured."])
    (OUT/"analysis.json").write_text(json.dumps(analysis, indent=2)+"\n")
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.2), constrained_layout=True)
    for seed, color in zip(curves, ("#1565c0", "#d84315", "#2e7d32")):
        rows = [r for r in history if r["seed"] == seed and r["split"] == "train"]
        x = [r["update"] for r in rows]
        axes[0].plot(x, [r["path_position_cm"]*10 for r in rows], "o-", color=color, label=f"init {seed}")
        axes[1].plot(x, [r["windows_passing"] for r in rows], "o-", color=color)
        loss = np.asarray([r["loss"] for r in curves[seed]])
        axes[2].plot(np.arange(64, 2049), np.convolve(loss, np.ones(64)/64, mode="valid"), color=color)
    axes[0].set_yscale("log")
    axes[0].axhline(5, color="gray", linestyle="--", linewidth=1)
    axes[0].set_ylabel("Training mean position error (mm, log scale)")
    axes[0].set_title("All three fit; init 43 deteriorates late")
    axes[0].legend()
    axes[1].axhline(29, color="gray", linestyle="--", linewidth=1)
    axes[1].set_ylim(-1, 33)
    axes[1].set_ylabel("Windows passing all 3 sampling draws / 32")
    axes[1].set_title("Window criterion: mean <=5 mm and <=5 deg")
    axes[2].set_yscale("log")
    axes[2].set_ylabel("Pose training loss (64-update moving mean)")
    axes[2].set_title("Training objective is not rollout accuracy")
    for ax in axes:
        ax.set_xlabel("Optimizer update")
        ax.axvline(1024, color="#999999", linewidth=1, linestyle=":")
        ax.grid(alpha=.2)
    fig.savefig(OUT/"training_trace.png", dpi=160)
    plt.close(fig)
    print(json.dumps({k:v for k,v in analysis.items() if k != "artifact_hashes"}, indent=2))


if __name__ == "__main__":
    main()

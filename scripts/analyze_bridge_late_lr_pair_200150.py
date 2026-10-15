"""Independently score the completed common-state learning-rate intervention."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from analyze_bridge_training_trace_200039 import errors, FIELDS

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/"reports/bridge-late-lr-fit-200150"
OUT = ROOT/"reports/bridge_late_lr_analysis_200150_20261012"


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    global DATA, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, default=DATA)
    ap.add_argument('--output', type=Path, default=OUT)
    ap.add_argument('--package', type=Path, default=ROOT/'cluster_packages/bridge_late_lr_pair_v1')
    ap.add_argument('--initialization-seed', type=int, default=43)
    ap.add_argument('--entry', default='check_bridge_late_lr_pair.py')
    args = ap.parse_args()
    DATA, OUT = args.data, args.output
    init = args.initialization_seed
    OUT.mkdir(exist_ok=True)
    report = json.loads((DATA/"report.json").read_text())
    manifest = json.loads((DATA/"plan.json").read_text())["manifest"]
    package = args.package
    origin_dir = ROOT/"reports/bridge-trace-fit-200039"
    origin = json.loads((origin_dir/"report.json").read_text())
    assert report["manifest_sha256"] == sha(package/"manifest.json")
    assert manifest == json.loads((package/"manifest.json").read_text())
    assert sha(origin_dir/"report.json") == manifest["late_lr"]["origin_report_sha256"]
    assert all(sha(package/n) == h for n,h in report["source_sha256"].items())
    assert sha(DATA/"runner_snapshot.py") == report["source_sha256"][args.entry]
    assert all(report[k] for k in ("execution_checks_passed", "input_contract_passed", "control_reproduces_origin", "clip_unchanged", "other_frozen_unchanged"))
    assert not report["gripper_trained"] and not report["simulation_executed"]
    assert report["initialization_seed"] == init and report["sampling_steps"] == 100
    assert report["sampling_seeds"] == [1101,1102,1103]
    assert report["data_checks"] == origin["data_checks"]
    assert all(v['raw_identity_exact'] for v in report['input_numeric_comparisons'])
    assert all(v['max_abs'] <= v['atol'] for row in report['input_numeric_comparisons'] for v in row['differences'].values())
    assert all(v == 0 for v in report["warmup_origin_differences"].values())
    old = next(r for r in origin["runs"] if r["initialization_seed"] == init)
    assert report["initial_shared_sha256"] == old["initial_shared_sha256"]
    arms = {r["arm"]:r for r in report["runs"]}
    assert set(arms) == {"constant", "reduced"}
    for name,r in arms.items():
        assert r["common_weights_sha256"] == report["common_state_sha256"]
        assert r["common_adam_sha256"] == report["common_adam_sha256"]
        assert r["actual_draw_slots_per_window"] == [256]*32
        assert r["learning_rate"] == manifest["late_lr"]["rates"][name]
    assert arms["constant"]["loss_first"] == arms["reduced"]["loss_first"]
    assert all(v == 0 for v in arms['constant']['origin_final_metric_differences'].values())
    assert arms['constant']['history'][-1]['train']['fit_gate']['pose_fit_passed'] == old['final']['train']['fit_gate']['pose_fit_passed']
    logs = {arm:[json.loads(line) for line in (DATA/f"{arm}_updates.jsonl").read_text().splitlines()]
            for arm in ("common", "constant", "reduced")}
    for arm,entries in logs.items():
        assert [r["update"] for r in entries] == list(range(1,1025) if arm == "common" else range(1025,2049))
        assert np.isfinite([r["loss"] for r in entries]).all()
        assert all(all(np.isfinite(v) and v > 0 for v in r["unclipped_module_gradient_norms"].values()) for r in entries)
        assert all(r["learning_rate"] == (1e-5 if arm == "reduced" else 1e-4) for r in entries)
    assert logs["constant"][0] == {**logs["reduced"][0], "learning_rate":1e-4}
    labels, snapshots, windows, episode_rows, hashes = {}, [], [], [], {}
    delta_max = {k:0.0 for k in FIELDS}
    final_errors = {}
    cases = [("common",h["update"],"train",h["train"]) for h in report["common_history"]]
    for arm,run in arms.items():
        cases += [(arm,h["update"],"train",h["train"]) for h in run["history"]]
        cases += [(arm,2048,"development",run["final_development"])]
    for arm,update,split,reported in cases:
        file = DATA/f"{arm}_u{update:04d}_{split}.npz"; hashes[file.name] = sha(file)
        with np.load(file) as z:
            pred,target,indices = (z[k] for k in ("predictions","targets","row_indices"))
        rows = manifest[split]; count = len(rows)
        assert pred.shape == (3,count,16,8)
        assert np.array_equal(indices,np.arange(count) if split == "train" else np.arange(32,72))
        if split in labels: assert np.array_equal(target, labels[split])
        else: labels[split] = target.copy()
        p,r = errors(pred,target)
        metrics = dict(zip(FIELDS,map(float,(p.mean(),r.mean(),p[...,-1].mean(),r[...,-1].mean()))))
        for k in FIELDS:
            delta_max[k] = max(delta_max[k],abs(metrics[k]-reported[k]))
            assert abs(metrics[k]-reported[k]) < (.02 if "rotation" in k else 1e-4)
        good = ((p.mean(-1) <= .5) & (r.mean(-1) <= 5)).all(0)
        gate = bool(good.mean() >= .9 and (p.mean((1,2)) <= .5).all() and (r.mean((1,2)) <= 5).all()
            and (p[...,-1].mean(1) <= 1).all() and (r[...,-1].mean(1) <= 10).all())
        if split == "train":
            assert int(good.sum()) == reported["fit_gate"]["windows_all_three_seeds_pass"]
            assert gate == reported["fit_gate"]["pose_fit_passed"]
        snapshots.append(dict(arm=arm,update=update,split=split,**metrics,passing_windows=int(good.sum()),gate=gate))
        for i,row in enumerate(rows):
            windows.append(dict(arm=arm,update=update,split=split,window=i,
                episode=row["shard"]+"::"+str(row["record_index"]),task=row["task"],
                path_position_cm=float(p[:,i].mean()),path_rotation_deg=float(r[:,i].mean()),
                passes_all_sampling_seeds=bool(good[i])))
        if update == 2048: final_errors[(arm,split)] = (p,r)
        if arm == "common" or (arm == "constant" and update in (1536,2048)):
            with np.load(origin_dir/f"seed{init}_u{update:04d}_{split}.npz") as z:
                assert np.array_equal(pred,z["predictions"]) and np.array_equal(target,z["targets"])
        if split == "development":
            for ep in sorted(set(row["shard"]+"::"+str(row["record_index"]) for row in rows)):
                ids = [i for i,row in enumerate(rows) if row["shard"]+"::"+str(row["record_index"]) == ep]
                episode_rows.append(dict(arm=arm,episode=ep,windows=len(ids),
                    position_cm=float(p[:,ids].mean()),rotation_deg=float(r[:,ids].mean())))
    for name,rows in [("history.csv",snapshots),("windows.csv",windows),("development_episodes.csv",episode_rows)]:
        with (OUT/name).open("w",encoding="utf-8-sig",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    cp,cr = final_errors[("constant","train")];lp,lr = final_errors[("reduced","train")]
    development = {a:{r["episode"]:r for r in episode_rows if r["arm"] == a} for a in arms}
    result = dict(verified=True,initialization_seed=init,report_sha256=sha(DATA/"report.json"),common_state_and_adam_identical=True,
        common_predictions_and_constant_final_exactly_reproduce_origin=True,
        all_reduced_monitor_points_fit=True,
        reduced_final=dict(position_mm=float(lp.mean()*10),rotation_deg=float(lr.mean()),
            endpoint_position_mm=float(lp[...,-1].mean()*10),endpoint_rotation_deg=float(lr[...,-1].mean()),
            window_mean_position_max_mm=float(lp.mean((0,2)).max()*10),
            individual_target_position_max_mm=float(lp.max()*10),
            passing_windows=next(s['passing_windows'] for s in snapshots if s['arm']=='reduced' and s['update']==2048 and s['split']=='train')),
        training_windows_position_improved=int((lp.mean((0,2)) < cp.mean((0,2))).sum()),
        development_episodes_position_improved_vs_constant=sum(development['reduced'][ep]['position_cm'] < development['constant'][ep]['position_cm'] for ep in development['constant']),
        max_float64_metric_difference=delta_max,artifact_hashes=hashes,
        pt_files_downloaded=bool(list(DATA.glob('*.pt'))),
        checkpoint_total_cluster_bytes=sum(c['bytes'] for c in report['checkpoints']),
        limitations=['One training initialization; three sampling draws are not training replications.',
                     'Window-average pose fitting is not every target error below threshold or robot success.',
                     'Held-out pose performance remains worse than the static baseline; no intermediate development evaluation.',
                     'No unique faulty layer established; broad training/generalization root cause remains open.'])
    assert all(h['gate'] for h in snapshots if h['arm'] == 'reduced' and h['split'] == 'train')
    (OUT/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(10.6,4),constrained_layout=True)
    common=[s for s in snapshots if s['arm']=='common']
    for arm,color,label in [('constant','#d84315','continue 1e-4'),('reduced','#1565c0','reduce to 1e-5')]:
        rows=[common[-1]]+[s for s in snapshots if s['arm']==arm and s['split']=='train']
        axes[0].plot([s['update'] for s in rows],[s['path_position_cm']*10 for s in rows],'o-',color=color,label=label)
        axes[1].plot([s['update'] for s in rows],[s['passing_windows'] for s in rows],'o-',color=color)
    axes[0].set_yscale('log');axes[0].set_ylabel('Training mean position error (mm, log scale)');axes[0].legend()
    axes[0].axhline(5,color='gray',linestyle='--',linewidth=1)
    axes[0].set_title('Same fitted weights and Adam state at update 1024')
    axes[1].set_ylabel('Windows passing all 3 sampling draws / 32');axes[1].set_ylim(-1,33)
    axes[1].axhline(29,color='gray',linestyle='--',linewidth=1)
    axes[1].set_title('Lower continuation LR retains fitting')
    for ax in axes:ax.set_xlabel('Optimizer update');ax.grid(alpha=.2)
    fig.savefig(OUT/'late_lr_training.png',dpi=160);plt.close(fig)
    print(json.dumps({k:v for k,v in result.items() if k!='artifact_hashes'},indent=2))


if __name__ == '__main__':main()

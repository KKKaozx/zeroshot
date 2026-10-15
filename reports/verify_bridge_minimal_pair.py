"""Independent float64 scoring of actual saved native DDPM predictions."""
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"reports/bridge_minimal_pair_fit_20261011"

def metrics(pred,target):
    p=np.linalg.norm(pred[...,:3].astype(float)-target[None,...,:3],axis=-1)*10
    a=pred[...,3:7].astype(float);b=target[None,...,3:7].astype(float)
    a/=np.linalg.norm(a,axis=-1,keepdims=True);b/=np.linalg.norm(b,axis=-1,keepdims=True)
    r=np.degrees(2*np.arccos(np.clip(abs((a*b).sum(-1)),0,1)))
    good=((p.mean(-1)<=.5)&(r.mean(-1)<=5)).all(0)
    passed=bool((p.mean((1,2))<=.5).all() and (r.mean((1,2))<=5).all() and good.mean()>=.9
        and (p[:,:,-1].mean(1)<=1).all() and (r[:,:,-1].mean(1)<=10).all())
    return dict(position_cm=float(p.mean()),rotation_deg=float(r.mean()),
        endpoint_position_cm=float(p[:,:,-1].mean()),endpoint_rotation_deg=float(r[:,:,-1].mean()),
        windows_all_three_seeds_pass=int(good.sum()),pose_fit_passed=passed,
        per_seed_position_cm=p.mean((1,2)).tolist(),per_seed_rotation_deg=r.mean((1,2)).tolist(),
        per_horizon_position_mm=(p.mean((0,1))*10).tolist(),per_horizon_rotation_deg=r.mean((0,1)).tolist())

def main():
    report=json.loads((OUT/"report.json").read_text());plan=json.loads((OUT/"plan.json").read_text())
    assert report["execution_checks_passed"] and len(report["runs"])==6
    sources=report["source_sha256"]
    assert all(hashlib.sha256((ROOT/"code"/n).read_bytes()).hexdigest()==h for n,h in sources.items())
    assert sources["check_bridge_minimal_pair.py"]==hashlib.sha256((OUT/"runner_snapshot.py").read_bytes()).hexdigest()
    initial_hashes={};rows=[];diffs=[]
    retained=np.load(ROOT/"reports/bridge_samecount32_fit_20261011/repeat_current_predictions.npz")
    for run in report["runs"]:
        init=run["initialization_seed"];arm=run["arm"]
        assert run["preflight_initialization_loss_metrics_match"] if init==42 else True
        assert run["frozen_other_unchanged"] and all(run["changed"].values())
        assert run["updates"]==(500 if arm=="early_alone" else 1000)
        assert run["draw_slots"]==([4000,0] if arm=="early_alone" else [4000,4000])
        if init in initial_hashes:assert initial_hashes[init]==run["initial_shared_sha256"]
        initial_hashes[init]=run["initial_shared_sha256"]
        for hist in run["history"][1:]:
            k=hist["updates_per_window"]
            file=OUT/f"seed{init}_{arm}_u{k}.npz"
            z=np.load(file);pred=z["predictions"];target=z["targets"]
            assert pred.shape==(3,2,16,8) and z["contexts"].shape==(2,1024)
            assert np.isfinite(pred).all() and np.array_equal(target,retained["targets"][[26,10]])
            active=1 if arm=="early_alone" else 2
            recomputed={"training_metrics":metrics(pred[:,:active],target[:active]),
                        "early":metrics(pred[:,:1],target[:1]),"late":metrics(pred[:,1:],target[1:])}
            for key,m in recomputed.items():
                old=hist[key]
                for field in ["position_cm","rotation_deg","endpoint_position_cm","endpoint_rotation_deg"]:
                    d=abs(m[field]-old[field]);diffs.append(dict(seed=init,arm=arm,update=k,group=key,field=field,difference=d))
                    assert d<(.02 if "rotation" in field else 3e-6),(field,d)
                assert m["pose_fit_passed"]==old["pose_fit_passed"]
                assert m["windows_all_three_seeds_pass"]==old["windows_all_three_seeds_pass"]
            rows.append(dict(initialization_seed=init,arm=arm,updates_per_window=k,draw_slots=k*8,
                             groups=recomputed,artifact_sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
    assert len(set(initial_hashes.values()))==3
    verdict={arm:all(row["groups"]["training_metrics"]["pose_fit_passed"] for row in rows if row["arm"]==arm and row["updates_per_window"]==500) for arm in plan["arms"]}
    assert verdict==report["fit_verdict_by_arm"]
    verify=dict(execution_checks_passed=True,initializations_distinct=True,paired_initializations_equal=True,
        actual_predictions_recomputed=True,source_snapshot_current_hashes_match=True,source_sha256=sources,
        frozen_clip_unchanged=report["clip_unchanged"],all_retained_targets_equal=True,metrics=rows,
        reported_metric_differences=diffs,verdict=verdict)
    (OUT/"verification.json").write_text(json.dumps(verify,indent=2)+"\n")
    font=FontProperties(fname="C:/Windows/Fonts/msyh.ttc")
    plt.rcParams["axes.unicode_minus"]=False
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for arm,color in [("early_alone","#276e9b"),("early_late_joint","#c45520")]:
        for seed in [42,43,44]:
            subset=sorted([row for row in rows if row["arm"]==arm and row["initialization_seed"]==seed],key=lambda row:row["draw_slots"])
            for ax,field,scale in [(axes[0],"position_cm",10),(axes[1],"rotation_deg",1)]:
                ax.plot([row["draw_slots"] for row in subset],[row["groups"]["early"][field]*scale for row in subset],
                        marker={42:"o",43:"s",44:"^"}[seed],linestyle={42:"-",43:"--",44:":"}[seed],
                        color=color,alpha=.8,label=f"{arm} / init {seed}")
    for ax,ylabel,threshold in [(axes[0],"Early-window position error (mm)",5),(axes[1],"Early-window rotation error (deg)",5)]:
        ax.axhline(threshold,color="black",ls="--",alpha=.6);ax.set_xlabel("Draw slots per trained window");ax.set_ylabel(ylabel)
        ax.set_xscale("log");ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.suptitle("同一早期窗口：单独拟合与加入晚期窗口的三次初始化对照",fontproperties=font)
    fig.tight_layout();fig.savefig(OUT/"early_fit_comparison.png",dpi=140);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for row in rows:
        if row["updates_per_window"]!=500:continue
        color="#276e9b" if row["arm"]=="early_alone" else "#c45520"
        for ax,field in [(axes[0],"per_horizon_position_mm"),(axes[1],"per_horizon_rotation_deg")]:
            ax.plot(range(1,17),row["groups"]["early"][field],color=color,
                    linestyle={42:"-",43:"--",44:":"}[row["initialization_seed"]],alpha=.8,
                    label=f"{row['arm']} / init {row['initialization_seed']}")
    for ax,ylabel in [(axes[0],"Early-window position error (mm)"),(axes[1],"Early-window rotation error (deg)")]:
        ax.set_xlabel("Prediction horizon index");ax.set_ylabel(ylabel);ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.suptitle("4000份样本预算后的完整100步DDPM生成误差",fontproperties=font)
    fig.tight_layout();fig.savefig(OUT/"final_horizon_errors.png",dpi=140);plt.close(fig)
    print("INDEPENDENT VERIFICATION",verdict)
    for row in rows:
        print(row["initialization_seed"],row["arm"],row["draw_slots"],[(key,round(row["groups"][key]["position_cm"]*10,3),round(row["groups"][key]["rotation_deg"],3),row["groups"][key]["pose_fit_passed"]) for key in ["early","late"]])

if __name__=="__main__":main()

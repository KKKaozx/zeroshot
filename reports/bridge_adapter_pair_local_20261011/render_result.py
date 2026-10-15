"""Render this experiment's actual saved predictions; no model execution."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, fontManager

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--directory", type=Path, default=Path(__file__).resolve().parent)
OUT = ap.parse_args().directory.resolve()
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT/"code"))
from evaluate_bridge_endpoint_metrics import pose_errors
report = json.loads((OUT/"report.json").read_text())
assert report["execution_completed"] and report["clip_unchanged"]
fontManager.addfont("C:/Windows/Fonts/msyh.ttc")
font = FontProperties(fname="C:/Windows/Fonts/msyh.ttc")
plt.rcParams.update({"font.family": font.get_name(), "axes.unicode_minus": False,
                    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
dropout_pair = report["plan"].get("comparison") == "dropout"
arms = ["dropout_off", "dropout_on"] if dropout_pair else ["fixed_adapter", "joint_adapter"]
labels = ["联合训练：dropout关闭", "联合训练：dropout开启0.1"] if dropout_pair else ["固定八层 Adapter", "联合更新八层 Adapter"]
colors = ["#248282", "#dc8036"]
errors, records = {}, []
for arm in arms:
    data = np.load(OUT/f"{arm}_predictions.npz")
    p, r = pose_errors(data["predictions"], data["targets"][None])
    errors[arm] = p, r
    actual = report["groups"][arm]["final_metrics"]
    assert abs(float(p.mean())-actual["position_cm"]) < 1e-5
    assert abs(float(r.mean())-actual["rotation_deg"]) < 1e-4
for i, row in enumerate(report["plan"]["selection"]):
    record = dict(row)
    for arm in arms:
        p, r = errors[arm]
        record[arm] = dict(position_cm=float(p[:,i].mean()), rotation_deg=float(r[:,i].mean()),
            seed_position_cm=p[:,i].mean(axis=1).tolist(), seed_rotation_deg=r[:,i].mean(axis=1).tolist(),
            every_seed_pose_pass=bool(np.all(p[:,i].mean(axis=1)<=.5) and np.all(r[:,i].mean(axis=1)<=5)))
    records.append(record)
(OUT/"per_window_metrics.json").write_text(json.dumps(records, indent=2)+"\n")

fig, axes = plt.subplots(1, 3, figsize=(14, 4.7), layout="constrained")
for arm, label, color in zip(arms, labels, colors):
    h = report["groups"][arm]["history"]
    x = [v["update"] for v in h]
    for ax, key in zip(axes[:2], ["position_cm", "rotation_deg"]):
        ax.plot(x, [v[key] for v in h], "o-", label=label, color=color)
        ax.set_yscale("log")
        ax.set_xlabel("优化更新次数")
        ax.grid(alpha=.2)
axes[0].axhline(.5, ls="--", color="#aa4444", label="平均位置阈值 0.5 cm")
axes[1].axhline(5, ls="--", color="#aa4444", label="平均旋转阈值 5°")
axes[0].set_title("纯噪声完整100步生成：位置")
axes[0].set_ylabel("路径位置误差（cm，对数轴）")
axes[1].set_title("纯噪声完整100步生成：旋转")
axes[1].set_ylabel("路径旋转误差（度，对数轴）")
for ax in axes[:2]:
    ax.legend(fontsize=8)
x = np.arange(1,9)
for offset, arm, label, color in zip([-.18,.18], arms, labels, colors):
    values = errors[arm][0].mean(axis=2)
    mean = values.mean(axis=0)
    axes[2].bar(x+offset, mean, width=.36, color=color, label=label,
                yerr=np.stack([mean-values.min(axis=0), values.max(axis=0)-mean]), capsize=3)
axes[2].axhline(.5, ls="--", color="#aa4444")
axes[2].set_title("训练500次后：每窗口生成误差")
axes[2].set_ylabel("位置误差（cm）；误差线为采样种子范围")
axes[2].set_xlabel("预定训练窗口")
axes[2].set_xticks(x)
axes[2].legend(fontsize=8)
subtitle = ("两组均联合训练；仅训练时dropout开关不同；评估均关闭dropout"
            if dropout_pair else "CLIP冻结；两组Adapter均关闭dropout；夹爪头冻结；仅训练位姿")
fig.suptitle("当前模型条件链的实跑对照：同一8窗口、同一起点、每组500更新\n"+subtitle, fontsize=13)
fig.savefig(OUT/("dropout_pair_overview.png" if dropout_pair else "adapter_pair_overview.png"), dpi=150, facecolor="white")
plt.close(fig)

fig, axes = plt.subplots(4, 2, figsize=(12, 11), layout="constrained")
steps = np.arange(1,17)
for i, ax in enumerate(axes.flat):
    for arm, label, color in zip(arms, labels, colors):
        values = errors[arm][0][:,i]
        ax.plot(steps, values.mean(axis=0), color=color, label=label)
        ax.fill_between(steps, values.min(axis=0), values.max(axis=0), color=color, alpha=.18)
    ax.set_title(f"窗口{i+1}：{report['plan']['selection'][i]['episode'].split('::')[-1]}号记录", loc="left")
    ax.set_ylabel("位置误差（cm）")
    ax.set_xlabel("未来目标步")
    ax.set_xticks([1,4,8,12,16])
    ax.grid(alpha=.2)
axes[0,0].legend(fontsize=8)
fig.suptitle("真实目标与生成轨迹的逐步误差\n曲线为三种采样种子均值，阴影为范围；均为训练窗口", fontsize=14)
fig.savefig(OUT/("dropout_pair_eight_windows.png" if dropout_pair else "adapter_pair_eight_windows.png"), dpi=150, facecolor="white")
plt.close(fig)
print("Saved overview, eight-window plots and independently recomputed metrics.")

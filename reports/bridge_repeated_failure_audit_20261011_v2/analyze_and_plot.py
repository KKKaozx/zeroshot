import json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
root=Path("D:/ntu_related/dissertation/Zero_shot")
out=root/"reports/bridge_repeated_failure_audit_20261011_v2"
r=json.loads((out/"report.json").read_text());raw=np.load(out/"raw_episode_arrays.npz");world=np.load(out/"world_predictions.npz")
s=raw["states"];a=raw["actions"]
font=FontProperties(fname="C:/Windows/Fonts/msyh.ttc")
plt.rcParams["axes.unicode_minus"]=False
fig,ax=plt.subplots(2,3,figsize=(12,7))
for row,t in enumerate([0,12]):
    truth=s[t+1:t+17,:3]*100
    pred=world[f"samecount32_t{t}"]*100
    for j,name in enumerate(["X","Y","Z"]):
        axis=ax[row,j]; axis.plot(range(1,17),truth[:,j],color="black",label="Measured target")
        axis.plot(range(1,17),pred[:,:,j].mean(0),color="#c45520",label="Native DDPM prediction")
        axis.fill_between(range(1,17),pred[:,:,j].min(0),pred[:,:,j].max(0),color="#c45520",alpha=.2)
        axis.set_title(f"start {t} / world {name} (cm)")
        axis.set_xlabel("Future observation index relative to input")
        axis.grid(alpha=.25)
ax[0,0].legend(fontsize=8)
fig.suptitle("实际保存的预测与测量目标：各窗口使用自身当前位姿还原世界坐标",fontproperties=font,fontsize=13)
fig.tight_layout();fig.savefig(out/"world_trajectory.png",dpi=140);plt.close(fig)
fig,ax=plt.subplots(1,2,figsize=(11,4))
for group,color in [("samecount32","#c45520"),("timepoint64","#276e9b")]:
    for row in r["saved_prediction_metrics"][group]:
        ax[0].plot(range(1,17),row["per_horizon_position_mm"],label=f"{group} / start {row['start']}",linestyle="-" if row["start"]==0 else "--",color=color)
        ax[1].plot(range(1,17),row["per_horizon_rotation_deg"],label=f"{group} / start {row['start']}",linestyle="-" if row["start"]==0 else "--",color=color)
for axis,label in zip(ax,["Position error (mm)","Rotation error (deg)"]):
    axis.set_ylabel(label);axis.set_xlabel("Prediction horizon index");axis.grid(alpha=.25);axis.legend(fontsize=8)
fig.suptitle("重复失败集中在早期窗口的后续预测；这是整段输出，非仿真逐步执行",fontproperties=font,fontsize=13)
fig.tight_layout();fig.savefig(out/"horizon_errors.png",dpi=140);plt.close(fig)
# Same-world direct metrics independent of production pose_errors.
checks=[]
for group,rows in r["saved_prediction_metrics"].items():
    for row in rows:
        t=row["start"];p=world[f"{group}_t{t}"];e=np.linalg.norm(p-s[t+1:t+17,:3],axis=-1)*1000
        assert abs(e.mean()-row["path_position_mm"])<1e-10
        assert abs(e[:,-1].mean()-row["endpoint_position_mm"])<1e-10
        checks.append(dict(group=group,start=t,position_metrics_recomputed=True))
# Exploratory lag: action[i] compared with observed delta[i+lag].
delta=np.diff(s[:,:3],axis=0)
lags=[]
for lag in range(-2,4):
    indices=[i for i in range(len(a)-1) if 0<=i+lag<len(delta)]
    err=a[indices,:3]-delta[np.array(indices)+lag]
    lags.append(dict(lag=lag,pairs=len(indices),mean_vector_error_cm=float(np.linalg.norm(err,axis=1).mean()*100),
                     rmse_component_cm=float(np.sqrt(np.mean(err**2))*100)))
stats=[]
for t in [0,12]:
    seg=s[t:t+17,:3];diff=seg[-1]-seg[0]
    stats.append(dict(start=t,anchor_xyz_m=seg[0].tolist(),endpoint_delta_cm=(diff*100).tolist(),
                      path_length_cm=float(np.linalg.norm(np.diff(seg,axis=0),axis=1).sum()*100),
                      maximum_distance_from_anchor_cm=float(np.linalg.norm(seg-seg[0],axis=1).max()*100),
                      per_axis_span_cm=((seg.max(0)-seg.min(0))*100).tolist()))
verification=dict(position_metrics=checks,window_geometry=stats,exploratory_action_state_lag=lags,
    lag_limit="Different pair counts and unknown control/timestamp semantics; neither prove misalignment nor authorize relabeling",
    production_source_unchanged=all(hashlib.sha256((root/"code"/name).read_bytes()).hexdigest()==value for name,value in r["source_sha256"].items()),
    geometry_max_xyz_cm=max(v["world_xyz_max_error_cm"] for v in r["independent_geometry"]),
    geometry_max_rotation_deg=max(v["rotation_matrix_max_error_deg"] for v in r["independent_geometry"]),
    focal_features_distinct=not r["feature_pairs"][10]["all_tokens"]["exactly_equal"],
    focal_token_cosine_rank=1+sum(v["all_tokens"]["cosine"]>r["feature_pairs"][10]["all_tokens"]["cosine"] for v in r["feature_pairs"]),
    scope="32 retained training windows; 16 episodes. No training or held-out evaluation.")
assert verification["production_source_unchanged"]
(out/"verification.json").write_text(json.dumps(verification,indent=2)+"\n")
print(json.dumps(verification,indent=2))


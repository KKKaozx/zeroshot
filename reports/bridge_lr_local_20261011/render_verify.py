"""Compare actual generated actions against the preceding 3e-4 control."""
from pathlib import Path
import hashlib
import json
import shutil
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, fontManager

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
BASE = ROOT/'reports/bridge_exposure_local_20261011'
sys.path.insert(0,str(ROOT/'code'))
from evaluate_bridge_endpoint_metrics import pose_errors
new = json.loads((OUT/'report.json').read_text())
old = json.loads((BASE/'report.json').read_text())
for report in [new,old]:
    assert report['execution_completed'] and report['clip_unchanged']
    g = report['groups']['dropout_on']
    assert g['frozen_gripper_unchanged'] and g['adapter_changed'] and g['decoder_changed']
for key in ['initial_hashes','raw_data_check','conditioning']:
    assert new[key]==old[key],key
assert new['plan']['learning_rate']==1e-4 and old['plan']['learning_rate']==3e-4
assert {k:v for k,v in new['plan'].items() if k!='learning_rate'}=={k:v for k,v in old['plan'].items() if k!='learning_rate'}
for name in old['source_sha256']:
    if name!='fit_bridge_adapter_pair.py':
        assert new['source_sha256'][name]==old['source_sha256'][name],name
for k,v in old['groups']['dropout_on']['history'][0].items():
    if k!='elapsed_seconds':
        assert new['groups']['dropout_on']['history'][0][k]==v,k
assert json.loads((OUT/'dropout_on_losses.json').read_text())[0]==json.loads((BASE/'dropout_on_losses.json').read_text())[0]
records=[]; final_errors=[]
for directory,report in [(BASE,old),(OUT,new)]:
    for row in report['groups']['dropout_on']['history']:
        d=np.load(directory/f"update_{row['update']}_predictions.npz")
        p,r=pose_errors(d['predictions'],d['targets'][None])
        assert abs(float(p.mean())-row['position_cm'])<1e-5
        assert abs(float(r.mean())-row['rotation_deg'])<1e-4
        window_pass=(p.mean(axis=2)<=.5)&(r.mean(axis=2)<=5)
        fit=bool(np.all(p.mean(axis=(1,2))<=.5) and np.all(r.mean(axis=(1,2))<=5)
            and window_pass.all(axis=0).mean()>=.9 and np.all(p[:,:,-1].mean(axis=1)<=1)
            and np.all(r[:,:,-1].mean(axis=1)<=10))
        assert fit==row['pose_fit_passed']
        assert int(window_pass.all(axis=0).sum())==row['windows_all_three_seeds_pass']
        records.append(dict(learning_rate=report['plan']['learning_rate'],updates=row['update'],
            position_cm=float(p.mean()),rotation_deg=float(r.mean()),endpoint_position_cm=float(p[:,:,-1].mean()),
            endpoint_rotation_deg=float(r[:,:,-1].mean()),windows_all_three_seeds_pass=int(window_pass.all(axis=0).sum()),
            pose_fit_passed=fit))
        if row['update']==500: final_errors.append(p.mean(axis=2))
(OUT/'independent_metrics.json').write_text(json.dumps(records,indent=2)+'\n')
if not (OUT/'runner_snapshot.py').exists():
    source=ROOT/'code/fit_bridge_adapter_pair.py'
    assert hashlib.sha256(source.read_bytes()).hexdigest()==new['source_sha256'][source.name]
    shutil.copyfile(source,OUT/'runner_snapshot.py')
assert hashlib.sha256((OUT/'runner_snapshot.py').read_bytes()).hexdigest()==new['source_sha256']['fit_bridge_adapter_pair.py']
fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family':FontProperties(fname='C:/Windows/Fonts/msyh.ttc').get_name(),
                    'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,3,figsize=(14,4.7),layout='constrained')
for i,(report,color) in enumerate([(old,'#248282'),(new,'#dc8036')]):
    h=report['groups']['dropout_on']['history']; label=f"学习率 {report['plan']['learning_rate']:g}"
    for ax,key in zip(axes[:2],['position_cm','rotation_deg']):
        ax.plot([x['update'] for x in h],[x[key] for x in h],'o-',color=color,label=label)
    values=final_errors[i]; mean=values.mean(axis=0)
    axes[2].bar(np.arange(1,9)+(-.18 if i==0 else .18),mean,width=.36,color=color,label=label,
        yerr=np.stack([mean-values.min(axis=0),values.max(axis=0)-mean]),capsize=3)
for ax,threshold,title in zip(axes[:2],[.5,5],['路径位置误差（cm）','路径旋转误差（°）']):
    ax.set_yscale('log');ax.axhline(threshold,color='#aa4444',ls='--');ax.grid(alpha=.2)
    ax.set_title(title+'，对数轴');ax.set_xlabel('优化更新次数 / 每窗口暴露量');ax.legend(fontsize=8)
axes[2].set_title('500次更新：每窗口平均位置误差');axes[2].set_ylabel('cm，误差线为采样种子范围')
axes[2].set_xlabel('同一批预定训练窗口');axes[2].set_xticks(np.arange(1,9));axes[2].legend(fontsize=8)
axes[2].axhline(.5,color='#aa4444',ls='--')
fig.suptitle('实际当前模型的学习率单变量对照\n同一初始化与8个训练窗口；Adapter＋扩散位姿联合训练；完整100步纯噪声生成',fontsize=13)
fig.savefig(OUT/'lr_overview.png',dpi=150,facecolor='white');plt.close(fig)
verification=dict(only_plan_difference='learning_rate',same_initial_parameters=True,same_raw_inputs=True,
    same_clip_tokens=True,same_production_source=True,same_initial_generation_and_first_loss=True,
    independent_metrics_match=True,clip_and_gripper_unchanged=True,training_seeds=1,sampling_seeds=3,
    runner_snapshot_matches_executed_source=True,reference_report_sha256=hashlib.sha256((BASE/'report.json').read_bytes()).hexdigest())
verification['files_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.iterdir())
                            if p.is_file() and p.name!='verification.json'}
(OUT/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
print(json.dumps(records,indent=2));print('LR CONTROL INDEPENDENT VERIFICATION PASSED')

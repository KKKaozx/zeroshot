"""Independently measure saved generation and verify the previous training path."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager, FontProperties

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT/'code'))
from evaluate_bridge_endpoint_metrics import pose_errors
r = json.loads((OUT/'report.json').read_text())
old = json.loads((ROOT/'reports/bridge_dropout_pair_local_20261011/report.json').read_text())
assert r['execution_completed'] and r['clip_unchanged']
h = r['groups']['dropout_on']['history']
oldh = old['groups']['dropout_on']['history']
assert r['initial_hashes'] == old['initial_hashes']
assert r['raw_data_check'] == old['raw_data_check']
assert r['conditioning'] == old['conditioning']
assert r['plan']['config'] == old['plan']['config']
metric_keys = [k for k in oldh[0] if k not in ['elapsed_seconds']]
replay = {str(row['update']): all(row[k] == next(x for x in h if x['update']==row['update'])[k]
          for k in metric_keys) for row in oldh}
assert all(replay.values()), replay
assert r['groups']['dropout_on']['adapter_changed'] and r['groups']['dropout_on']['decoder_changed']
assert r['groups']['dropout_on']['frozen_gripper_unchanged']
errors, summaries = {}, []
for row in h:
    step = row['update']
    d = np.load(OUT/f'update_{step}_predictions.npz')
    p, q = pose_errors(d['predictions'], d['targets'][None])
    assert abs(float(p.mean())-row['position_cm']) < 1e-5
    assert abs(float(q.mean())-row['rotation_deg']) < 1e-4
    passing = (p.mean(axis=2)<=.5) & (q.mean(axis=2)<=5)
    assert int(passing.all(axis=0).sum()) == row['windows_all_three_seeds_pass']
    errors[step] = p, q
    summaries.append(dict(updates=step, exposures_per_window=step,
        total_window_draws=step*8, position_cm=float(p.mean()), rotation_deg=float(q.mean()),
        endpoint_position_cm=float(p[:,:,-1].mean()), endpoint_rotation_deg=float(q[:,:,-1].mean()),
        window_position_cm=p.mean(axis=(0,2)).tolist(), window_rotation_deg=q.mean(axis=(0,2)).tolist(),
        every_seed_window_pass=passing.all(axis=0).tolist(), pose_fit_passed=row['pose_fit_passed']))
(OUT/'independent_metrics.json').write_text(json.dumps(summaries,indent=2)+'\n')
runner = ROOT/'code/fit_bridge_adapter_pair.py'
shutil.copyfile(runner, OUT/'runner_snapshot.py')
assert hashlib.sha256(runner.read_bytes()).hexdigest()==r['source_sha256'][runner.name]
verification = dict(execution_completed=True, same_initial_parameters=True, same_raw_inputs=True,
    same_clip_tokens=True, common_updates_replay_previous_exactly=replay,
    independent_saved_prediction_metrics_match=True, clip_unchanged=True, gripper_unchanged=True,
    one_training_path=True, training_seeds=1, sampling_seeds=3,
    runner_snapshot_matches_executed_source=True)
fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
font = FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family':font.get_name(),'axes.unicode_minus':False,
    'axes.spines.top':False,'axes.spines.right':False})
fig, axes = plt.subplots(1,3,figsize=(14,4.7),layout='constrained')
for ax,key,threshold,title in zip(axes[:2],['position_cm','rotation_deg'],[.5,5],
    ['路径位置误差（cm）','路径旋转误差（°）']):
    ax.plot([x['update'] for x in h],[x[key] for x in h],'o-',color='#248282')
    ax.set_yscale('log'); ax.set_xlabel('每个窗口见到的次数 / 优化更新次数')
    ax.set_title(title+'，对数轴'); ax.grid(alpha=.2)
    ax.axhline(threshold,ls='--',color='#aa4444',label='预定位姿门槛'); ax.legend(fontsize=8)
    for x in h:
        ax.annotate(f"{x[key]:.2f}",(x['update'],x[key]),xytext=(5,5),textcoords='offset points',fontsize=9)
for offset,step,color in [(-.18,20,'#dc8036'),(.18,500,'#248282')]:
    values=errors[step][0].mean(axis=2); mean=values.mean(axis=0)
    axes[2].bar(np.arange(1,9)+offset,mean,width=.36,color=color,label=f'{step}次',
        yerr=np.stack([mean-values.min(axis=0),values.max(axis=0)-mean]),capsize=3)
axes[2].set_xticks(np.arange(1,9)); axes[2].set_xlabel('预定训练窗口')
axes[2].set_ylabel('平均位置误差（cm）'); axes[2].set_title('20次与500次：同一8窗口')
axes[2].axhline(.5,ls='--',color='#aa4444'); axes[2].legend()
fig.suptitle('实际当前模型：同一条训练过程的完整100步生成\n八层Adapter联合训练，dropout=0.1；CLIP和夹爪冻结；仅训练集位姿记忆',fontsize=13)
fig.savefig(OUT/'exposure_overview.png',dpi=150,facecolor='white'); plt.close(fig)
verification['files_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
    for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='verification.json'}
(OUT/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
print(json.dumps(summaries,indent=2))
print('PREVIOUS PATH REPLAY AND INDEPENDENT METRICS: PASSED')

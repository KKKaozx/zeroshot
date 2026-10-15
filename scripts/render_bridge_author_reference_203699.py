"""Plot audited development errors, using saved predictions only."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'reports/bridge_author_reference_analysis_203699'
a = json.loads((OUT/'analysis.json').read_text(encoding='utf-8'))
g = a['groups']['ema/development']
plt.rcParams.update({'font.family':'Microsoft YaHei', 'axes.unicode_minus':False, 'font.size':10})
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
h = g['per_horizon']
axes[0].plot([x['target'] for x in h], [x['position_cm'] for x in h], marker='o', label='作者最终EMA')
axes[0].plot([x['target'] for x in h], [x['static_position_cm'] for x in h], linestyle='--', label='静止基线')
axes[0].set(xlabel='未来目标序号（共16个）', ylabel='位置误差（cm）', title='留出窗口：后段误差更大')
axes[0].set_xticks([1, 4, 8, 12, 16]); axes[0].grid(alpha=.2); axes[0].legend()
names = {'close microwave':'关闭微波炉','close the drawer':'关闭抽屉','open the drawer':'打开抽屉',
         'take broccoli out of pan':'从锅中取出西兰花','turn lever vertical to front':'转动控制杆'}
tasks = g['tasks']
delta = np.array([x['path_position_cm']-x['static']['path_position_cm'] for x in tasks])
axes[1].barh([names[x['task']] for x in tasks], delta, color=['#d66a4b' if x>0 else '#3c8b92' for x in delta])
axes[1].axvline(0, color='black', linewidth=.8)
axes[1].set(xlabel='作者误差 − 静止误差（cm）', title='任务拆分：左侧改善，右侧变差')
axes[1].invert_yaxis(); axes[1].grid(axis='x', alpha=.2)
fig.suptitle('203699作者Bridge参照｜固定40留出窗口、最终EMA、3次采样均值', fontsize=12)
fig.savefig(OUT/'development_diagnosis.png', dpi=160)
plt.close(fig)
print(OUT/'development_diagnosis.png')

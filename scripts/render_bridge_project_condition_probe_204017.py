"""Plot actual saved frozen-project outputs; no simulation or model inference."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/bridge_project_condition_analysis_204017'
r = json.loads((OUT / 'analysis.json').read_text(encoding='utf-8'))
assert r['verified']
plt.rcParams.update({'font.family': 'Microsoft YaHei', 'axes.unicode_minus': False,
                     'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.4))
steps = ['99', '74', '49', '24', '0']
colors = plt.cm.tab10(np.linspace(0, .9, 9))
for ax, arm, unit in [(axes[0, 0], 'original', 1), (axes[0, 1], 'donor', 10)]:
    for row, color in zip(r['per_window'], colors):
        values = [row['clean_stages'][s][arm]['position_cm'] * unit for s in steps]
        ax.plot(range(5), values, color=color, alpha=.65, linewidth=1)
    means = [r['stage_metrics'][s]['clean'][arm]['position_cm'] * unit for s in steps]
    ax.plot(range(5), means, color='#172c46', marker='o', linewidth=2.7, label='9窗平均')
    ax.set_xticks(range(5), steps); ax.set_xlabel('原生反向链时间步：第一次 → 最后一次')
    ax.set_ylabel('位置误差（cm）' if arm == 'original' else '位置误差（mm）')
    ax.set_ylim(0, 18 if arm == 'original' else 5)
    ax.grid(axis='y', color='#e6e9ee'); ax.legend(frameon=False, loc='upper left')
axes[0, 0].set_title('A  留出条件：首次估计已存在厘米级误差', loc='left')
axes[0, 1].set_title('B  对应训练条件：各阶段保持毫米级误差', loc='left')
axes[0, 1].text(.98, .94, '所有窗口×采样×记录阶段最大值：4.49mm',
                 ha='right', va='top', transform=axes[0, 1].transAxes, fontsize=9)
ax = axes[1, 0]
x = np.arange(9)
first = np.array([w['clean_stages']['99']['original']['position_cm'] for w in r['per_window']])
last = np.array([w['original']['position_cm'] for w in r['per_window']])
ax.bar(x - .19, first, width=.38, color='#748cac', label='第一次估计')
ax.bar(x + .19, last, width=.38, color='#de9553', label='最终输出')
ax.set_xticks(x, [w['window'] for w in r['per_window']]); ax.set_xlabel('固定留出窗口索引')
ax.set_ylabel('位置误差（cm）'); ax.set_ylim(0, 18)
ax.set_title('C  9/9窗：首次误差 ≥ 最终误差的90%', loc='left')
ax.grid(axis='y', color='#e6e9ee'); ax.set_axisbelow(True); ax.legend(frameon=False)
ax = axes[1, 1]
x = np.arange(2)
final = [e['final']['position_cm'] for e in r['per_episode']]
static = [e['static']['position_cm'] for e in r['per_episode']]
ax.bar(x - .18, final, width=.36, color='#748cac', label='模型最终输出')
ax.bar(x + .18, static, width=.36, color='#c5cbd2', label='静止基线')
for xx, value in zip(x - .18, final): ax.text(xx, value + .2, f'{value:.2f}', ha='center', fontsize=9)
for xx, value in zip(x + .18, static): ax.text(xx, value + .2, f'{value:.2f}', ha='center', fontsize=9)
ax.set_xticks(x, ['演示1（6窗）', '演示2（3窗）']); ax.set_ylim(0, 16)
ax.set_ylabel('位置误差（cm）'); ax.set_title('D  两条演示的错误程度不同', loc='left')
ax.grid(axis='y', color='#e6e9ee'); ax.set_axisbelow(True); ax.legend(frameon=False)
fig.suptitle('项目202278真实状态模型：冻结条件诊断204017', fontsize=16, fontweight='bold', y=.985)
fig.text(.5, .94, '实际保存输出 · 相同噪声配对 · 3次采样平均 · 0次优化更新', ha='center', color='#526176')
fig.text(.5, .012, '9个重叠窗口来自2条西兰花留出演示；此图是离线轨迹误差，不是机器人抓取成功率。',
         ha='center', fontsize=9, color='#526176')
fig.tight_layout(rect=[0, .03, 1, .93], h_pad=2.0, w_pad=2.0)
path = OUT / 'condition_stage_diagnosis.png'
fig.savefig(path, dpi=150, facecolor='white'); plt.close(fig)
print(path)

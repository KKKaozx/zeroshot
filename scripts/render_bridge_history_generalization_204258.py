"""Scientific plot from independently audited saved predictions, not a rollout."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/bridge_history_generalization_analysis_204258'
a = json.loads((OUT / 'analysis.json').read_text(encoding='utf-8')); assert a['verified']
plt.rcParams.update({'font.family': 'Microsoft YaHei', 'axes.unicode_minus': False, 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False})
colors = ['#4779a7', '#d18e51', '#b8c0ca']
fig, ax = plt.subplots(2, 2, figsize=(11.2, 8.1))
train = [a['results'][arm + '/train'] for arm in ['repeat_current', 'real_previous']]
bars = ax[0, 0].bar(range(2), [r['metrics']['path_position_cm'] * 10 for r in train], color=colors[:2], width=.55)
ax[0, 0].set_xticks(range(2), ['重复当前帧', '真实上一帧']); ax[0, 0].set_ylim(0, 3)
ax[0, 0].set_ylabel('训练位置误差（mm）'); ax[0, 0].set_title('A  两组训练拟合均达标', loc='left')
for b, r in zip(bars, train):
    ax[0, 0].text(b.get_x() + b.get_width()/2, b.get_height() + .1,
                   f"{b.get_height():.2f}mm\n逐窗达标{r['passing_windows']}/128", ha='center', fontsize=10)
metrics = [a['results'][arm + '/development']['metrics'] for arm in ['repeat_current', 'real_previous']]
static = a['results']['real_previous/development']['static']
x = np.arange(2)
for j, (r, c, label) in enumerate(zip(metrics + [static], colors, ['重复当前帧', '真实上一帧', '静止基线'])):
    heights = [r['path_position_cm'], r['endpoint_position_cm']]
    bars = ax[0, 1].bar(x + (j-1)*.23, heights, width=.23, color=c, label=label)
    for b in bars: ax[0, 1].text(b.get_x()+b.get_width()/2, b.get_height()+.12, f'{b.get_height():.2f}', ha='center', fontsize=8)
ax[0, 1].set_xticks(x, ['完整轨迹', '第16目标']); ax[0, 1].set_ylim(0, 14)
ax[0, 1].set_ylabel('开发位置误差（cm）'); ax[0, 1].set_title('B  真实历史未改善留出位置预测', loc='left')
ax[0, 1].legend(frameon=False, fontsize=9, loc='upper left', ncol=3)
rows = a['per_episode']; y = np.arange(len(rows))
changes = [r['history_minus_control']['path_position_cm'] for r in rows]
labels = [r['episode'].split('tfrecord-')[1].split('-of')[0]+'::'+r['episode'].split('::')[1] for r in rows]
ax[1, 0].barh(y, changes, color=['#bd655b' if d>0 else '#3b968b' for d in changes], height=.65)
ax[1, 0].set_yticks(y, labels); ax[1, 0].invert_yaxis(); ax[1, 0].axvline(0, color='#637083', linewidth=.8)
ax[1, 0].set_xlabel('真实历史 − 控制，位置误差变化（cm）')
ax[1, 0].set_title('C  4/10演示位置改善；负值为改善', loc='left')
ax[1, 0].set_xlim(-1.5, 3.9)
for i, d in enumerate(changes):
    ax[1, 0].text(d + (.08 if d>=0 else -.08), i, f'{d:+.2f}', va='center', ha='left' if d>=0 else 'right', fontsize=8)
groups = a['subgroups'][1:]; x = np.arange(2)
for j, arm in enumerate(['repeat_current', 'real_previous']):
    heights = [g[arm]['path_position_cm'] for g in groups]
    bars = ax[1, 1].bar(x + (j-.5)*.32, heights, width=.32, color=colors[j], label=['重复当前帧','真实上一帧'][j])
    for b in bars: ax[1, 1].text(b.get_x()+b.get_width()/2, b.get_height()+.1, f'{b.get_height():.2f}', ha='center', fontsize=9)
ax[1, 1].set_xticks(x, ['可用历史（30窗）', '演示起点（10窗）']); ax[1, 1].set_ylim(0, 11)
ax[1, 1].set_ylabel('开发位置误差（cm）'); ax[1, 1].set_title('D  有历史的子组也未获益', loc='left')
ax[1, 1].legend(frameon=False, fontsize=9, loc='upper left')
for axis in ax.flatten():
    axis.set_axisbelow(True); axis.grid(axis='y' if axis is not ax[1,0] else 'x', color='#e8ebef', linewidth=.6)
fig.suptitle('历史输入有界对照204258：独立复算结果', fontsize=16, fontweight='bold', y=.985)
fig.text(.5, .944, '同128训练／40开发窗口 · 单训练初始化43 · 三次采样均值 · 结构、参数量与预算相同', ha='center', color='#596b7d')
fig.text(.5, .013, '这是保存预测的离线位姿误差；不代表机器人抓取成功率。起点组虽输入相同，两组学习后的参数不同。', ha='center', fontsize=9, color='#596b7d')
fig.tight_layout(rect=[0,.035,1,.925], h_pad=2.2, w_pad=2.0)
path = OUT / 'bounded_history_result.png'; fig.savefig(path, dpi=150, facecolor='white'); plt.close(fig)
print(path)

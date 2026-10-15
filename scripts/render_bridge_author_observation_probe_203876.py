"""Render independently audited paired sensitivity and one valid-observation contrast."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/bridge_author_observation_analysis_203876'
a=json.loads((OUT/'analysis.json').read_text())
plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'font.size':10})
fig,axes=plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True)
y=np.arange(9)
axes[0].barh(y-.18,[x['output_changes']['image_only']['position_cm'] for x in a['per_window']],height=.35,label='仅换图像')
axes[0].barh(y+.18,[x['output_changes']['state_only']['position_cm'] for x in a['per_window']],height=.35,label='仅换当前状态')
axes[0].set_yticks(y,[str(x['window']) for x in a['per_window']]);axes[0].invert_yaxis()
axes[0].set(xlabel='预测位置变化（cm）',ylabel='固定窗口索引',title='8/9窗口换图像的输出变化更大')
axes[0].grid(axis='x',alpha=.2);axes[0].legend()
x=next(x for x in a['per_window'] if x['window']==146)
axes[1].bar(['两真实目标\n轨迹差','两真实观察下\n模型输出差'],[x['true_target_difference']['path_position_cm'],
    x['output_changes']['both']['position_cm']],color=['#c65947','#477fbd'],width=.55)
axes[1].set(ylabel='平均位置差（cm）',title='窗口146：两组有效观察，固定配对噪声')
axes[1].set_ylim(0,x['true_target_difference']['path_position_cm']+3)
axes[1].grid(axis='y',alpha=.2)
for i,v in enumerate([x['true_target_difference']['path_position_cm'],x['output_changes']['both']['position_cm']]):
    axes[1].text(i,v+.3,f'{v:.2f}',ha='center')
fig.suptitle('203876作者最终EMA冻结对照｜0次优化更新，原40窗输出精确重放',fontsize=12)
fig.savefig(OUT/'paired_observation_diagnosis.png',dpi=150);plt.close(fig)
print(OUT/'paired_observation_diagnosis.png')

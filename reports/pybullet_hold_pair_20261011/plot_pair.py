"""Plot actual model-controlled object heights in the paired physics runs."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent
summary=json.loads((root/'comparison.json').read_text())
target=json.loads((root/'preparation.json').read_text())['hold_reference_object_height_m']
plt.rcParams['font.family']='Microsoft YaHei'
fig,axes=plt.subplots(1,2,figsize=(11,4.5),sharey=True,layout='constrained')
for ax,name,title in zip(axes,['baseline','hold'],['对照：重复旧上升窗口','处理：新增静止保持窗口']):
    report=json.loads((root/f'{name}_rollout/rollout_report.json').read_text())
    initial=report['results'][0]['initial_height_m'];goal=(target-initial)*100
    ax.axhspan(goal-2,goal+2,color='green',alpha=.10,label='目标高度±2cm')
    ax.axhline(goal,color='black',ls='--',lw=1,label='采集演示保持高度')
    for row in report['results']:
        n=row['sampling_seed'];data=np.load(root/f'{name}_rollout'/f'policy_seed_{n}.npz')
        ax.plot(np.arange(1,113)*.1,(data['object_heights']-initial)*100,label=f'采样种子 {n}')
    ax.axvspan(9.2,11.2,color='gray',alpha=.10)
    group=summary['groups'][name]
    ax.set(title=title+f"\n抓取 {group['grasp_passes']}/3；停位 {group['hold_passes']}/3",
           xlabel='真实仿真时间 / 秒')
    ax.grid(alpha=.2);ax.legend(fontsize=8,loc='upper left')
axes[0].set_ylabel('物体相对初始高度 / cm')
fig.suptitle('仅替换第9个训练窗口：固定训练场景闭环对照',fontsize=13)
fig.savefig(root/'height_pair.png',dpi=160)

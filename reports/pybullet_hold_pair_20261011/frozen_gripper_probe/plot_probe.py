"""Plot actual saved images and frozen same-input counterfactual readouts."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

HERE=Path(__file__).resolve().parent
report=json.loads((HERE/'report.json').read_text(encoding='utf-8'))
trace=np.load(HERE.parent/'hold_rollout/policy_seed_1102.npz')
training=np.load(HERE.parent/'hold/training_examples.npz')
font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
fig=plt.figure(figsize=(11,7),layout='constrained')
gs=fig.add_gridspec(2,3,height_ratios=[1.15,1])
values={x['training_observation']:x['failed_pose_width_first_offset_logit']
        for x in report['training_image_only_comparison_at_failed_first_action']}
for col,start in enumerate([None,38,90]):
    ax=fig.add_subplot(gs[0,col])
    if start is None:
        image=trace['images'][13];title='真实失败观测：第53步之前\n同输入读出：打开（logit +1.037）'
    else:
        idx=int(np.flatnonzero(training['starts']==start)[0])
        image=training['images'][idx]
        title=f'训练画面：第{start}帧\n同位姿、同宽度：关闭（logit {values[start]:.3f}）'
    ax.imshow(image);ax.set_axis_off();ax.set_title(title,fontproperties=font,fontsize=11)
ax=fig.add_subplot(gs[1,:])
keys=['recorded_pose','stationary_pose','training_hold_width_same_pose_image','training_hold_image_same_pose_width']
labels=['原始真实输入','只换原地保持位姿','只换训练保持宽度','只换训练保持图像']
ys=[report['first_failed_open']['logits'][k] for k in keys]
ax.bar(range(4),ys,color=['#c45148' if y>=0 else '#338269' for y in ys],width=.6)
ax.axhline(0,color='#333333',linewidth=1)
ax.set_xticks(range(4),labels,fontproperties=font,fontsize=10)
ax.set_ylabel('夹爪 logit（≥0 打开；<0 关闭）',fontproperties=font)
ax.set_ylim(-11,3)
for i,y in enumerate(ys): ax.text(i,y+(0.3 if y>=0 else -0.7),f'{y:+.3f}',ha='center')
ax.set_title('同一冻结权重、同一指令、同一动作时间索引；每列仅替换标注的输入',fontproperties=font,fontsize=11)
fig.suptitle('实际错误打开的输入对照：替换图像改变决定，替换位姿或宽度未改变\n仅为离线读出诊断，替换结果未控制机器人',fontproperties=font,fontsize=13)
fig.savefig(HERE/'conditioning_comparison.png',dpi=150)
plt.close(fig)

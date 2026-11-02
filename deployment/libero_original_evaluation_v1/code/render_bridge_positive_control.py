"""Render actual train-only predictions from the local positive control."""
from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'reports/bridge_positive_local_20261011'
r=json.loads((out/'report.json').read_text())
font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family':font.get_name(),'axes.unicode_minus':False,'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
# Explicit font property used for every text item by font name resolved on this host.
from matplotlib.font_manager import fontManager
fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
z=np.load(out/'fresh_multiscale_predictions.npz');p=z['predictions'];t=z['targets'];xyz=np.linalg.norm(p[...,:3]-t[None,...,:3],axis=-1)*10
pq=p[...,3:7]/np.maximum(np.linalg.norm(p[...,3:7],axis=-1,keepdims=True),1e-8)
tq=t[...,3:7]/np.maximum(np.linalg.norm(t[...,3:7],axis=-1,keepdims=True),1e-8)
rot=2*np.rad2deg(np.arccos(np.clip(abs((pq*tq).sum(axis=-1)),0,1)))
per_window=[]
for i in range(8):
 per_window.append({**r['selection'][i],'position_cm':float(xyz[:,i].mean()),'rotation_deg':float(rot[:,i].mean()),'endpoint_position_cm':float(xyz[:,i,-1].mean()),'all_seeds_position_cm':xyz[:,i].mean(axis=1).tolist(),'all_seeds_rotation_deg':rot[:,i].mean(axis=1).tolist()})
(out/'per_window_metrics.json').write_text(json.dumps(per_window,indent=2)+'\n')
fig,axs=plt.subplots(4,2,figsize=(13,12),layout='constrained')
colors=['#1f77b4','#ed7d31','#399761'];steps=np.arange(1,17)
for i,ax in enumerate(axs.flat):
 for d,color in enumerate(colors):
  ax.plot(steps,t[i,:,d]*10,color=color,lw=2,label='XYZ'[d]+' 目标')
  ax.plot(steps,p[0,i,:,d]*10,color=color,lw=1.5,ls='--',label='XYZ'[d]+' 预测')
 ax.set_title(f"窗口 {i+1}｜三种子平均位置误差 {xyz[:,i].mean():.3f} cm",loc='left',fontsize=11)
 ax.set_xlabel('未来目标步');ax.set_ylabel('工具系相对位移（cm）');ax.set_xticks([1,4,8,12,16]);ax.grid(alpha=.2)
axs[0,0].legend(ncol=3,fontsize=8,loc='best')
fig.suptitle('新训练的当前多尺度 U-Net：8 个真实训练窗口\n实线为真实目标，虚线为纯噪声开始的完整 100 步生成（种子 1101）',fontsize=15)
fig.savefig(out/'fresh_multiscale_8_windows.png',dpi=150);plt.close(fig)
# Overview: recorded real learning curve plus newly executed generation checks.
oldroot=ROOT.parent/'GPU cluster/diffusion-full-x0-186047';history=json.loads((oldroot/'history.json').read_text())['evaluations']
old=np.load(out/'replay_train_predictions.npz');op=np.linalg.norm(old['predictions'][...,:3]-old['targets'][None,...,:3],axis=-1)*10
fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
a=axs[0,0];xs=[x['step'] for x in history];ys=[np.mean([s['partitions']['train']['position_error_cm'] for s in x['samples']]) for x in history]
a.plot(xs,ys,'o-',color='#1f77b4');a.axhline(.5,ls='--',color='#b64040',label='拟合阈值 0.5 cm');a.set_yscale('log');a.set_title('已有 316 窗口实验：完整生成误差曲线');a.set_xlabel('优化更新次数');a.set_ylabel('训练位置误差（cm，对数轴）');a.legend(fontsize=8);a.grid(alpha=.2)
a=axs[0,1];windowpos=op.mean(axis=(0,2));a.hist(windowpos,bins=22,color='#348b84',edgecolor='white');a.axvline(.5,color='#b64040',ls='--');a.set_title('今天用当前采样代码重放：316 个窗口');a.set_xlabel('每窗口三种子平均位置误差（cm）');a.set_ylabel('窗口数');a.text(.98,.95,f"均值 {op.mean():.3f} cm\n311/316 窗口通过全部种子位姿阈值",ha='right',va='top',transform=a.transAxes,fontsize=9)
a=axs[1,0];hist=json.loads((out/'fit_history.json').read_text());xx=[h['update'] for h in hist];yy=[h['position_cm'] for h in hist];a.plot(xx,yy,'o-',color='#ed7d31');a.set_yscale('log');a.axhline(.5,color='#b64040',ls='--');a.set_title('今天从头训练当前多尺度 U-Net：8 个窗口');a.set_xlabel('优化更新次数');a.set_ylabel('完整生成位置误差（cm，对数轴）');a.grid(alpha=.2)
a=axs[1,1];a.axis('off');f=r['fresh_current_multiscale_fit'];a.text(0,1,'新的实测结果',fontsize=16,fontweight='bold',va='top');a.text(0,.84,f"{r['updates']} 次更新；每窗口 {r['draws_per_window']} 次训练\n位置：{f['position_cm']:.3f} cm（静止：{f['static_position_cm']:.2f} cm）\n旋转：{f['rotation_deg']:.3f}°（保持姿态：{f['static_rotation_deg']:.2f}°）\n终点位置：{f['endpoint_position_cm']:.3f} cm\n全部采样种子通过：8/8 窗口\n\n固定真实缓存上下文，仅训练位姿解码器。\n未测试原始 dataloader、CLIP、Adapter 联合训练。\n同训练窗口评估，无泛化或抓取成功声明。",fontsize=12,va='top',linespacing=1.65)
fig.suptitle('训练集拟合的直接证据：旧结果重放 + 当前多尺度解码器新训练',fontsize=16)
fig.savefig(out/'positive_control_overview.png',dpi=150);plt.close(fig)
summary=f'''# 本地训练集拟合阳性验证（2026-10-11）

本次真实运行 GPU 为 {r['gpu']}，PyTorch {r['torch']}。不是从历史报告抄取新的成绩。

## 结果

| 本次执行 | 窗口 | 位置误差 | 旋转误差 | 全部三种子位姿阈值通过 |
|---|---:|---:|---:|---:|
| 旧最终权重 + 当前生产采样方法重放 | 316 | {r['retained_compact_replay']['position_cm']:.6f} cm | {r['retained_compact_replay']['rotation_deg']:.6f}° | 311/316 |
| 当前多尺度 U-Net 从头训练 | 8 | {f['position_cm']:.6f} cm | {f['rotation_deg']:.6f}° | 8/8 |

新训练使用当前 code/diffusion_decoder.py 的实际 [256,512,1024] 网络，{r['decoder_parameters']} 个参数。真实缓存来自早期 sweep into pile 训练划分，按缓存顺序取前八条不同训练演示的第一个窗口，不按模型表现选样本。训练中只使用真实位姿目标，推理仅输入固定上下文与纯高斯动作噪声，使用当前 code/models.py 的 diffusion_loss、sample 和相应预测解释。构造时绕过下载 CLIP，用现有缓存上下文；这是有意隔离头部的测试。

Adam lr=3e-4、无 weight decay、固定学习率、batch=8、梯度裁剪1；最多2000更新，预先规定检查点500/1000/2000。在500更新时达到预设生成阈值并提前停止。每个窗口实际训练500次。三种采样种子为1101/1102/1103，只代表采样重复，不代表三个训练种子。

阈值：每种采样种子的总体位置≤0.5cm、旋转≤5°；至少90%窗口在全部种子下满足上述逐窗口均值阈值；每种子的总体终点位置≤1cm、旋转≤10°。不是每个动作点的最大误差阈值。

## 可以支持的结论

- 当前多尺度条件 U-Net 能在这八个真实缓存条件/目标上，从初始化开始学到有效位姿生成。
- 当前已执行的 x0 位姿训练路径与完整100步采样组合，不存在导致所有训练记忆都不可能的阻断。
- 保留的紧凑网络权重与当前实际紧凑解码器、采样方法兼容，重放误差接近原始316窗口结果。
- 非解码器参数在新训练中未变化；源代码、数据和权重哈希保存在report.json。

## 不能支持的结论

- 原始 dataloader 本次未重新执行，输入是已保留缓存；CLIP、Adapter 本次未前向或联合训练，不能据此判定其正确。
- 旧模型的 Adapter 为2层/256维，本次上下文来自它；不是当前8层Adapter有效性的证明。
- 夹爪头冻结，未建立新的夹爪切换学习、定位物体或抓取成功结论。
- 新实验使用8个训练窗口与记忆配方；不能直接解释五任务1836窗口联合训练为何失败，也不能据此单独归因于训练量。
- 没有使用验证/测试目标，没有泛化或闭环任务成功结论。
- 为节省磁盘未保留新训练的大权重；保留脚本、哈希、设置、逐窗口预测和结果，可按固定种子重新执行约一分钟的测试。

## 下一项唯一建议

下一步用当前原始图像/语言数据、同一批预定训练窗口和相同初始化，对比固定Adapter上下文与Adapter联合训练，保持解码器、优化预算和生成判据一致。不要再次改预测目标、U-Net宽度或归一化。该对照尚未运行。
'''
(ROOT/'docs/BRIDGE_POSITIVE_CONTROL_LOCAL_20261011.md').write_text(summary,encoding='utf-8')
print('Rendered two figures and saved verified local positive-control record.')

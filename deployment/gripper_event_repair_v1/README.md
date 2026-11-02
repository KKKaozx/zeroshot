# 保留位姿和开闭识别，增加事件监督：一个候选，待审阅

2026-10-31。未提交新训练。依赖现有`/projects/Zeroshot/dino_gripper_sidecar_v1`与235928的gripper.pt，不复制原DINO/位姿模型、原数据或9.54MB头训练输入，不重新采样。

下一步只尝试改善夹爪切换。已有数组的一次新问题分析显示，3个旧draw的开发事件中：张开24次有13次所在窗口没有同方向预测，闭合60次均有同方向预测但仅11次精确对齐，34次晚、15次早。允许±1步时张开/闭合F1仅.308/.352，不能解释为全体统一偏移一小步。±1/2/3等匹配只作描述，原精确步门槛保持，不用于新策略纠偏或准入。

## 唯一干预

保留同一个约27万参数夹爪结构，原BCE开闭状态损失不变，候选额外加一个事件损失：根据相邻预测概率构造“闭→开”“开→闭”概率，明确惩罚漏切换、多切换和切换位置错误。事件标签仍从原未来16步命令序列直接构造，没有新专家信息，没有从当前测量猜先前命令。

每个方向的正/负权重只用完整训练窗事件频率固定：正类`.5/f`、负类`.5/(1-f)`；两个方向取平均，再以固定系数1加到原状态BCE。没有按开发结果选系数或扫结构/时间偏移/阈值。额外强调已有边界信息是待验证候选，不能宣称它已解释或修复唯一根因。

两组均从235928同一完整头权重起跑，各新建空Adam，共用旧2048×64批次/生成draw计划、seed42、LR1e-4/clip1。控制组仅继续原状态BCE，候选组用状态+事件损失；每组新增2048，总4096头更新。这是匹配的延续阶段，不重跑235928原fresh实验，不从头重训位姿。

**229116 DINO位姿网络仍不加载、不更新，235928原头与原位姿检查点均只读保留。** 新候选不得覆盖它们。

## 预算与保护门槛

单A40、预计1–4分钟，Slurm硬上限10分钟，进程480秒；每512保存各组头/Adam/RNG，256MiB输出余量。不自动续训、改损失权重或阈值，不自动仿真。

两组固定终点评价，训练和开发分别要求：

1. 平均开闭召回率≥80%，且相比235928同角色下降不超过2个百分点（开发下限约85.62%，训练下限约88.67%）。原87.62%和90.67%结果永久保留，并非重新评分旧模型。
2. 张开、闭合精确步事件F1均≥0.50，保留原规则，不能用±步容差分数替代。

候选还须在开发两项事件F1上都优于同期控制，才优先选择候选。若控制达到上述绝对保护门槛而候选没有相对优势，保留控制的改善；两组均未过则本修复候选结束。选择规则在训练前固定，不中途选checkpoint。通过只表示可考虑下一阶段工程，不自动进入LIBERO或13模型矩阵。

## 已完成的本地工作

`gripper_event_objective.py`已实现稳定的相邻事件损失，3项CPU数学/梯度/数值饱和和训练频率绑定测试通过。状态输出结构、阈值、数据与位姿冻结方式均不改；原同输入位姿/RNG保护测试直接复用，不再重跑已关闭实验。没有新模型前向、真实头训练、位姿采样或物理回合。新目标的3项合成测试不是候选能力结果，实际A40入口尚未执行。

## 审阅后手动上传／提交／日志

本机PowerShell：

```powershell
scp "D:\ntu_related\dissertation\Zero_shot\cluster_packages\gripper_event_repair_v1_20261031.zip" "zixiao005@10.97.216.128:/projects/Zeroshot/"
```

集群终端：

```bash
cd /projects/Zeroshot
test ! -e gripper_event_repair_v1
unzip -n gripper_event_repair_v1_20261031.zip
cd gripper_event_repair_v1
sha256sum -c SHA256SUMS
bash submit_once.sh
```

脚本先核对既有v1包、原位姿检查点、235928头身份，再提交唯一10分钟作业，输出`GRIPPER EVENT REPAIR: JOB`。已有job_id不重交。

```bash
JOB=$(cat /projects/Zeroshot/gripper_event_repair_v1/job_id.txt)
tail -n 60 -F /projects/Zeroshot/logs/gripper-event-repair-${JOB}.out
```

完成后仅保存结果复核：

```bash
/projects/Zeroshot/envs/multitask-preflight-v1/bin/python -B /projects/Zeroshot/gripper_event_repair_v1/verify_pair.py /projects/Zeroshot/runs/gripper-event-repair-${JOB}
```

小结果可取回（按实际JOB替换）：

```powershell
scp "zixiao005@10.97.216.128:/projects/Zeroshot/runs/gripper-event-repair-JOB/small_results.zip" "D:\ntu_related\dissertation\Zero_shot\reports\"
```

新头采用独立`gripper-event-repair-head-v1`格式，源头2048+本阶段2048=4096总头更新，不能伪装为旧235928原头。当前包只准备该受控训练和结果验证，不会自动调用旧LIBERO加载器放行新头；若有头达到门槛，再依据已保存选择回执接入后续工程验证。

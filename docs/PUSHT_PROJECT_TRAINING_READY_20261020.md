# 冻结CLIP + 项目Adapter的Push-T正式对照

213439原CPU验证器已通过，小报告本地已读。11356帧视觉缓存与原128帧参考完全一致（最大差0），文本差0，CLIP前后权重未变、全冻结；11.133GiB数值数组留集群，未在本地重复散列。212158完整summary也已核对。0个项目正式训练更新、没有新性能结论。

## 已准备的唯一训练组

冻结CLIP + 原八层Adapter + 1024→64映射，接作者原生扩散后端。从seed42相同初始后端开始，不继承原生已训练权重；Adapter/映射seed43。相同90演示/10726窗、两个观测、16步动作、训练统计、取样和独立扩散随机流、原epsilon损失/100步cosineDDPM/AdamW/EMA；原LR总计划512400/warmup500保留。

总84168次更新，分42084、84168两段手动提交，原始/EMA/Adam/LR/EMA计数及所有RNG完整保存并验证恢复，不是两次独立训练。只到最终固定EMA才评同20个开发场景，使用原采样种子和奖励算法，第一段不评分，不访问最终50。比较包含预处理的前端家族，不能把效果全部归给Adapter单独部件。第一段预计约12h纯训练，加磁盘写入和排队；每段作业上限20h。

复用已完成资源和在线接线证据，不另提交预检。正式入口自带首损失/梯度检查、第二次更新的完整磁盘恢复一致性检查；未通过即停止。新缓存索引绑定CPU检查使用真实作者损失与EMA、小型替身网络：8/38/64批更新与直接输入完全一致，梯度有效、磁盘恢复下一步一致、错误微批/可训练特征/错误项目与样本合同拒绝；原两段循环计数和评价AST、复用文件字节、3.9语法/CLI/Bash/ZIP通过。未在本地运行完整新模型训练或仿真，不将替身检查当成泛化成绩。

包：[pusht_project_training_v1.zip](../cluster_packages/pusht_project_training_v1.zip)。
检查：[本地记录](../reports/pusht_project_training_checks_20261020.json)。
发布SHA及大小：[发布记录](../reports/pusht_project_training_release_20261020.json)。

## 本地PowerShell上传

```powershell
scp "D:\ntu_related\dissertation\Zero_shot\cluster_packages\pusht_project_training_v1.zip" zixiao005@10.97.216.128:/projects/Zeroshot/
```

## 集群只提交第一段

```bash
cd /projects/Zeroshot
if [ ! -e pusht_project_training_v1 ]; then
  /projects/Zeroshot/envs/official-pusht-v1/bin/python -m zipfile -e pusht_project_training_v1.zip . &&
  bash /projects/Zeroshot/pusht_project_training_v1/submit_part1_once.sh
else
  echo '目录已存在，请检查已有作业，避免重复提交。'
fi
```

日志`logs/pusht-project-part1-JOB.out`，报告`runs/pusht-project-part1-JOB/report.json`。约5GB检查点和11GB缓存均留集群。结束后：

```bash
/projects/Zeroshot/envs/official-pusht-v1/bin/python \
 /projects/Zeroshot/pusht_project_training_v1/verify_segment.py \
 /projects/Zeroshot/runs/pusht-project-part1-JOB/report.json
```

核对后手动提交第二段（将JOB替换为第一段实际编号）：

```bash
sbatch --export=ALL,PUSHT_PROJECT_RESUME=/projects/Zeroshot/runs/pusht-project-part1-JOB/latest.pt \
 /projects/Zeroshot/pusht_project_training_v1/part2.sh
```

不是现在同时提交两段。中断重启须用新作业输出目录和本合同完整检查点，不改源目录、不重置优化器、学习率或EMA。启动前要求12GiB磁盘空余以支持原子检查点写入。

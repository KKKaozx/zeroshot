# 冻结DINO同预算正式训练包

221055六份小输出已收齐：原DINO128帧差异0，新worker1/2/4批量token/context/action差异0，固定文本与模型状态一致，缓存编码239.54秒、完整390.26秒。保存的两窗完整16步与执行8步动作数组本地逐值相等；完整11.13GiB缓存散列验收依据原集群验证器，不冒称本地取得大缓存，也未收到Slurm账目回执。

## 本次唯一实验

新冻结DINO-L/14视觉组，从共同seed42作者后端和seed43原八层前端初始化。原CLIP215999与原生212158直接复用；数据、训练帧、归一化、样本序列、八层Adapter、池化/投影、U-Net、损失、优化器、EMA、微批和完整LR进度保持。总84168有效更新分42084/84168两段，第一段不跑开发评价、不选模型；第二段接续完整raw/EMA/Adam/LR/EMA计数及全部RNG，最终一次同dev20评价，final50封存。

原CLIP主模型仍保留。DINO原生ImageNet归一化和最终LayerNorm属于骨干表示差异；保留旧224resize/224crop画面范围，偏离DINO默认256resize/crop224已公开。旧CLIP训练A6000、新DINO申请A40，单训练seed及复用开发场景的限制保留。结果不能拆成Adapter单独效果，也不能代表机器人抓取或最终泛化确认。

## 入口和验收

34文件ZIP约88KB，13个原模块字节不变；既有DINO worker/client仅协议文件改名，不重复实际IPC或资源实验。新正式入口的逐处替换列于`entry_changes.json`，不改原更新函数/噪声目标/训练循环预算。独立DINO checkpoint格式绑定cache/backbone合同；旧CLIP/原生/零块或不同缓存不可恢复。原完整首两步磁盘恢复audit直接嵌入正式起始阶段，不另加资源作业。

本地35项检查及4个Bash脚本语法通过。新完整格式用极小八块替身模型做真实磁盘恢复/下一步逐状态精确重放；输出验证器使用临时小文件与合成元数据，不能说真实DINO正式首步已跑。实际大模型首步loss/梯度/完整恢复必须在新正式作业启动时通过。未提交任何新集群作业。

两段各A40/4CPU/40GiB/20小时上限，按220882估算纯训练每段12.73小时、合计25.46小时，另有I/O/保存/最终闭环开销。每2000步保存完整latest；不requeue、不自动续段或重投。提交前和作业内分别检查项目200十进制GB保守上限及18GiB余量，非权威quota查询，不自动删旧资产。新完整checkpoint实际大小仍待正式保存时观察。

## 第一段上传和提交

PowerShell：

```powershell
scp "D:\ntu_related\dissertation\Zero_shot\cluster_packages\pusht_dino_training_v1.zip" zixiao005@10.97.216.128:/projects/Zeroshot/
```

集群：

```bash
cd /projects/Zeroshot
unzip pusht_dino_training_v1.zip
bash /projects/Zeroshot/pusht_dino_training_v1/submit_part1_once.sh
```

第一段日志`logs/pusht-dino-part1-作业号.out`，输出`runs/pusht-dino-part1-作业号`。应先出现原损失/梯度及完整磁盘恢复通过，再正常更新；通过表示训练入口可继续，不是泛化提升。

完成后核验（替换真实编号）：

```bash
DINO_PART1_JOB=实际编号
LD_LIBRARY_PATH=/projects/Zeroshot/envs/official-pusht-v1/lib \
  /projects/Zeroshot/envs/official-pusht-v1/bin/python -B \
  /projects/Zeroshot/pusht_dino_training_v1/verify_dino.py \
  "/projects/Zeroshot/runs/pusht-dino-part1-${DINO_PART1_JOB}"
```

核验后再手工接续，不设置自动依赖：

```bash
bash /projects/Zeroshot/pusht_dino_training_v1/submit_part2_once.sh \
  "/projects/Zeroshot/runs/pusht-dino-part1-${DINO_PART1_JOB}/latest.pt"
```

第二段同样核验输出目录；最终三组奖励及DINO减CLIP的配对场景区间自动计算。不会据dev成绩自动开新seed/结构分支、访问final50或替换主CLIP模型。若训练中断，先收错误和latest侧车，再按完整状态审阅恢复，不能直接重新训练或重置优化器。

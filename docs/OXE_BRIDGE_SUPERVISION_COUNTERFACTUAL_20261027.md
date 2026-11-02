# 固定 Bridge 曝光，检验其他来源监督的影响

用户提出“四个数据集一起训练可能导致问题”。已有只读梯度检查发现部分来源方向冲突，但合并梯度仍与 Bridge 正相关，因此尚不能据此认定混训是主要原因。旧 Bridge 配方仅换 DINO 的收益也保留，不再重复该实验。

本次保持**当前主模型配方**：冻结 DINO-L/14、冻结 CLIP 文本、8 层 Adapter/CLS、多尺度 epsilon DDPM100、原动作和名义时间规则。复用 seed42 混训最终模型及全部 65,536 步原批次记录作为基线。候选只去掉非 Bridge 的监督损失分子，保留原分母、全部输入、随机过程和优化器时钟。

| 固定项 | 数值/规则 |
|---|---|
| 总更新 | 65,536，分 32,768 + 32,768 两段 |
| Bridge 窗口呈现 | 131,244 次，与原混训相同 |
| 全部输入呈现 | 524,276 次，仍保留非 Bridge 行 |
| 不含 Bridge 的批次 | 6,617 步，监督梯度为零，仍执行 AdamW 和学习率步骤 |
| 损失归一化 | 保留原整个批次有效维度/夹爪分母，避免 Bridge 梯度被额外放大 |
| 验证损失 | 原损失原样保留，训练过滤不进入验证 |
| 评价 | 固定 64 个不同 Bridge 验证组，每组一窗；两模型同一 A40、同三次 DDPM 随机种子 |

普通“全换成 Bridge、仍跑 65,536 步”会增加约四倍 Bridge 曝光，不能区分数据量与其他来源监督的影响。本次保留非 Bridge 的输入，是为了固定 token padding、dropout、噪声和批次行为；它是**去掉其他来源监督的诊断**，不是常规 Bridge 单数据集训练。

原模型没有保存初始学习权重摘要。新脚本不伪造这个历史证据：先要求原代码、配置、环境和合同全部一致，再在任何优化步骤之前精确回放原首批的总损失、位姿损失、夹爪损失及梯度范数。出现未解释差异就停止。新预检另存完整初始摘要，绑定之后的正式候选。首批标量重放是复现证据，不能表述成旧完整初始张量已被存档。

短预检只做**两次实际更新**：更新 1 后保存，重新构造并恢复原优化器/批次/随机数状态，再更新至 2。它不评分、不传递预检学习权重，也不自动提交长训练。正式总耗时参考旧主模型约 **5–6 小时**，分两段，每段 4 小时墙钟上限。已有混训模型不重训；评价单独提交。

原先诊断的 8 个验证组保留作辅助锚点，新增 56 组在训练前按元数据 SHA 排名、组内选中间窗口；没有按目标或误差挑样。主要比较固定 64 组的平均轨迹位置误差，同时列旋转、终点、逐组差和组重采样区间。三次采样不等于三个训练种子；开发集比较也不变成正式 LIBERO 泛化结论。位置仍是记录单位经原 0.1 除数归一化，不新增物理单位声明。

本地已完成损失数值/梯度、全日程曝光计数、零梯度 AdamW、断点 RNG/优化器续跑、合同拒绝和评价维度的 CPU 夹具检查；原模型/编码器没有在这些检查中加载。Shell 仅做静态与 LF 检查，未冒称执行集群 bash。实际 A40 首批重放和预检尚待用户运行。

若候选明显改善，说明其他来源监督在这套配置和固定 Bridge 曝光下有伤害；仍需区分语义不一致与分布/优化冲突，不能直接推广为所有混训无效。若仍差，混训不是充分解释，应回到当前配方中尚未隔离的状态条件、训练目标及读出变化，不能通过追加预算掩盖结果。

上传包：`cluster_packages/oxe_bridge_supervision_v1.zip`，只含代码和小型元数据，没有原始数据或模型。原模型和历史结果均保留，主 LIBERO 大评估继续暂停。

上传（本地 PowerShell）：

```powershell
scp "D:\ntu_related\dissertation\Zero_shot\cluster_packages\oxe_bridge_supervision_v1.zip" zixiao005@10.97.216.128:/projects/Zeroshot/
```

集群先提交短预检：

```bash
(
set -euo pipefail
cd /projects/Zeroshot
unzip -n oxe_bridge_supervision_v1.zip
cd oxe_bridge_supervision_v1
bash submit_once.sh preflight
)
```

查看日志，把 `JOB_ID` 替换为新编号：

```bash
tail -n 60 -f /projects/Zeroshot/logs/oxe-bridge-supervision-JOB_ID.out
```

**尚未提交任何新作业。** 正式训练命令在包内 README，待实际短预检结果审阅后使用。

## 229597 初始化故障后的独立修复入口

第一段229597在0前向/0更新时冻结身份守卫失败，不是拟合/泛化结果。新包漏接此前已用于seed43的两项常量精确恢复，原失败证据保留。实际整体错误摘要与旧227815一致，旧故障只差`posterior_mean_x0`/`posterior_mean_xt`；当前逐张量差异仍须先读集群现存`actual_profile.json`确认。

独立修复包`cluster_packages/oxe_bridge_supervision_frozen_repair_v1.zip`保留原主包不改。运行`bash submit_once.sh 1`会先在登录节点比较当前既有失败文件；任何额外编码器/缓冲区变化、顺序差或未识别常量字节都会停止。确认后只恢复已存原始常量，仍必须通过原完整冻结摘要/初始学习摘要/首批损失和梯度精确重放，然后重提原零更新的第一段，无重复短训练。修复协议和源哈希额外绑定到诊断checkpoint。

```powershell
scp "D:\ntu_related\dissertation\Zero_shot\cluster_packages\oxe_bridge_supervision_frozen_repair_v1.zip" zixiao005@10.97.216.128:/projects/Zeroshot/
```

```bash
(
set -euo pipefail
cd /projects/Zeroshot
unzip -n oxe_bridge_supervision_frozen_repair_v1.zip
cd oxe_bridge_supervision_frozen_repair_v1
bash submit_once.sh 1
)
```

**后续第二段也在新修复目录执行`bash submit_once.sh 2`，评价用`bash submit_once.sh evaluate`，不再用上文原目录的fit命令。** 原失败marker不覆盖，新修复目录另存job ID。修复日志为`/projects/Zeroshot/logs/oxe-bridge-supervision-repaired-JOB_ID.out`。本地仅CPU夹具/字节核对通过，实际集群逐张量比较及修复运行尚未执行，未提交任何重试。

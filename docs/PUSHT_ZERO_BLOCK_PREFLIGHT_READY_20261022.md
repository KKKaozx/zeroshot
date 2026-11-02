# 无八层堆叠控制组：短预检包就绪

2026-10-22。用户同意先做同CLIP前端有无Adapter的直接对照。现有八层215999固定端点保留、完整训练和闭环分数不重跑。本轮只准备无交叉注意力块的预检包，未运行真实数据、未加载训练权重、未提交集群作业。

## 实际改变与比较边界

只移除八个跨注意力/FFN残差块（连同块内dropout），保留旧`.5*(CLS+mean patch)`、输出LayerNorm、1024→64投影和agent_pos两列。零块前端仍有67648个LayerNorm/投影参数。因而“无Adapter”是无八层融合堆叠，不是连输出归一化一起删除。

CLIP图文权重始终冻结。零块分支不再利用语言；Push-T始终只有同一句指令，适合作此任务内部机制对照，不能把零块分支直接替代正式多指令/零样本语言策略。当前生产八层默认不变。

从头构建时固定原作者后端seed42及初始SHA；用seed43按原构造器完整生成八层前端，再删除八层，保持旧输出归一化/投影的参数对象和值，以及初始化后的CPU随机状态。直接从零层构造会改变随机数消耗并可能连带改变投影初始化，所以本包不采用该做法。没有截断已训练215999，也没有载入已训练原生U-Net。

保持原90演示/10726窗/11356帧、训练专用统计、冻结CLIP缓存、64逻辑批次/8微批、末批38、原epsilon目标/DDPM100/AdamW/EMA、原512400学习率时域/warmup500和样本/噪声计划。后续若正式训练，零块控制仍需从头完成相同84168更新，而不是用预检权重接续。去掉Adapter没有去掉大U-Net，不能保证正式对照很短。

层数与参数量、块内dropout同时减少是该堆叠消融的组成部分；结果即使有差距，也不能单独归因某个注意力算子。历史八层训练在A6000，本预检默认A40；硬件/数值与单训练seed限制必须在后续结果解释中保留。正式采用判据及执行分段尚未制定，不能自动启动。

## 预检执行范围

仅零块模型做8个独立更新，加一次保存恢复后的同一步重算、一次单独的真实38样本探针，共10次更新函数执行；不重训八层。核对原loss/噪声/时间步/代表梯度，剩余LayerNorm和投影能更新，EMA冻结并跟随，raw/EMA/Adam/LR/计数/Python/NumPy/CPU/CUDA RNG恢复一致。最后只在两个训练输入上核对100步采样形状与1:9执行切片，不输出开发集分数或环境成功率。

所有预检权重都属于短测/一次性恢复状态，不是正式起点。replay.pt保留服务器用于SHA验收，不必下载。需要8GiB可用盘，不删除旧文件。

资源A40×1、4CPU、40GiB、15分钟硬上限，内部800秒、不requeue。沿用近期219129成功的卡型；A6000裸CUDA失败具体原因仍未隔离。实际资源与零块每更新耗时需服务器短测，84168更新的估算不含完整检查点I/O、闭环、排队；不承诺正式结束时间。

## 本地验收

47项新入口检查通过：实际原生产前端随机初始化逐字节重建、相同投影/LayerNorm值与对象、RNG保持、实际零层前向与独立公式、状态列/文本独立性、训练CLIP输入拒绝；原作者loss与EMA配小后端，8/38/64批次梯度及完整磁盘恢复一致；错误架构/协议/计划/报告/计数/文件等拒绝；Python3.9、两Bash入口、20文件ZIP与10个复用文件逐字节相同。没有加载真实训练权重或运行完整U-Net，不把本地合成验收称作泛化或集群运行通过。

包：`cluster_packages/pusht_zero_block_preflight_v1.zip`，20文件/50768字节。

ZIP SHA256：`f942c1723586758bd7a8e7d86ff3b98c7ae011f8dfec95e39473728d7f0dbfeb`。

protocol SHA256：`0a0a709878b1d65b1a6adff42548afd252da18d2fbf9c452fc95393d15daff1c`。

PowerShell上传：

```powershell
scp "D:\ntu_related\dissertation\Zero_shot\cluster_packages\pusht_zero_block_preflight_v1.zip" zixiao005@10.97.216.128:/projects/Zeroshot/
```

服务器提交：

```bash
cd /projects/Zeroshot
unzip -n pusht_zero_block_preflight_v1.zip
bash /projects/Zeroshot/pusht_zero_block_preflight_v1/submit_once.sh
```

日志`/projects/Zeroshot/logs/pusht-zero-preflight-编号.out`；输出`/projects/Zeroshot/runs/pusht-zero-preflight-编号`。

完成验证：

```bash
/projects/Zeroshot/envs/official-pusht-v1/bin/python \
  /projects/Zeroshot/pusht_zero_block_preflight_v1/verify_preflight.py \
  /projects/Zeroshot/runs/pusht-zero-preflight-实际编号
```

只新增必要的零块控制，不重开217820旁路，不重查已有FiLM/归一化/调度，不解冻CLIP，不读final50，不自动提交正式训练。

构建：`scripts/build_pusht_zero_block_preflight_20261022.py`。
检查：`scripts/check_pusht_zero_block_preflight_20261022.py`。
检查报告：`reports/pusht_zero_block_preflight_checks_20261022.json`。

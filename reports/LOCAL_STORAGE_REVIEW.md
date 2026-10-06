# 本地占用与归档候选

日期：2026-10-06。按用户说明，本地可用空间约 30 GB。此次只统计，没有移动或删除文件。

results 合计 **29.22 GiB**，其中 139 个检查点合计 **28.72 GiB**。同名或同大小文件不能据此判定为重复。

## 占用最大的实验目录

| 目录 | 总 GiB | 权重 GiB | 检查点数 | 处理状态 |
|---|---:|---:|---:|---|
| bcz_generalization_x0_v1 | 1.342 | 1.341 | 2 | 旧分支待归档核对 |
| bcz_overfit_cosine_v1 | 1.341 | 1.341 | 2 | 旧分支待归档核对 |
| bcz_overfit_cosine_v2 | 1.341 | 1.341 | 2 | 旧分支待归档核对 |
| bcz_overfit_x0_v1 | 1.341 | 1.341 | 2 | 旧分支待归档核对 |
| oxe_core_8shard_v1 | 1.341 | 1.341 | 2 | 旧分支待归档核对 |
| oxe_core_pilot | 1.341 | 1.341 | 2 | 旧分支待归档核对 |
| bcz_native_pool_adapter_lr_split_v1 | 1.214 | 1.213 | 2 | 旧分支待归档核对 |
| bcz_native_adapter_fit_lr_low_v1 | 1.213 | 1.213 | 2 | 旧分支待归档核对 |
| bcz_native_adapter_fit_lr_split_v1 | 1.213 | 1.213 | 2 | 旧分支待归档核对 |
| bcz_native_adapter_fit_v1 | 1.213 | 1.213 | 2 | 旧分支待归档核对 |
| bcz_native_pool_adapter_smoke_v1 | 1.213 | 1.213 | 2 | 旧分支待归档核对 |
| bcz_native_pool_adapter_v1 | 1.213 | 1.213 | 2 | 旧分支待归档核对 |
| bcz_pool_seen_probe_fit_v1 | 1.213 | 1.213 | 2 | 旧分支待归档核对 |
| bcz_pool_seen_probe_rotation4_v1 | 1.213 | 1.213 | 2 | 旧分支待归档核对 |
| bcz_cls_regression_fit_control_v1 | 1.211 | 1.211 | 2 | 旧分支待归档核对 |

## 保留与处理原则

保留原始完整 Bridge 训练权重、上下文缓存、隔离动作头、扩散导出输入/初始权重和诊断证据。这些文件由当前结果的 provenance 引用，不应仅因旧或大而删除。

其余实验目录先作为归档候选，不代表可直接删。优先将大权重转存至有容量的外部存储或集群，并核验 SHA256 和目标文件可读后，再决定本地副本处理；报告、配置、数据划分和来源指纹继续留在项目。当前没有执行转存。

完整逐文件清单：[local_checkpoint_inventory.csv](local_checkpoint_inventory.csv)。逐目录统计：[local_storage_inventory.json](local_storage_inventory.json)。

后续学习效果对照放在集群；本地不新增训练数据、模型大权重或完整结果包。每轮默认只下载文本/JSON/CSV和少量图例，总预算不超过 100 MiB。若需要大检查点，先核算容量和保存位置。

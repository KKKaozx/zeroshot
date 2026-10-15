# 本地文件清理候选 2026年10月15日

本次只盘点，未删除任何文件。D 盘当前空余约 65.44 GiB。仅检查本地工程，未查询集群配额。

| 候选 | 文件数 | 可释放 GiB | 代价与处理建议 |
|---|---:|---:|---|
| 已解压且逐成员 SHA256 一致的压缩传输包 | 9 | 0.81 | 优先候选；保留解压目录、原校验和与本次清单，未来传输需重新打包 |
| 历史实验 latest.pt，旁边保留 best.pt | 34 | 12.01 | 可选；失去最后状态的直接重放/恢复，不声称 best 与 latest 内容相同 |
| 上一项以外的旧 BC-Z 权重 | 29 | 11.96 | 进一步清理候选；旧 BC-Z 直接重放将需要重新训练，保留所有数值报告、划分、配置与代码 |

这些是互不重叠的候选分类，不是删除指令。第一批若只处理已验证传输包与旧 latest，预计可释放 12.82 GiB。

## 保留项目

- 不删除 training_cache/exports/bridge_expansion_training_v1/data 或其他原始分片；当前逻辑核对仍需要它们。
- 不删除 reports/bridge_joint_both_events_1000_local_20261011/final.pt；老师三项接口核对脚本明确加载它。
- 保留 PyBullet 抓取、保持配对的权重、轨迹、视频和报告，它们是已有正反对照的直接证据。
- 保留 202278、204017、204258 的原始预测 NPZ、协议、报告和日志，以支持独立复算。
- 不删除 CLIP 缓存、Python 环境或冻结的 cluster_packages 解压目录；本次没有证明它们可由其他完整副本替代。
- GitHub 只保存代码和选定证据，不能作为权重或数据已经备份的证明。

## 完整清单

[逐文件候选和保留原因](../reports/local_storage_review_20261015/checkpoint_candidates.csv)；[空间、归档逐成员核验和文件指纹](../reports/local_storage_review_20261015/inventory.json)。

## 压缩包核验结果

| 压缩包 | GiB | 匹配成员 | 所有成员已解压且一致 |
|---|---:|---:|---|
| `reports/bridge_author_obs_results_203876.tar.gz` | 0.00 | 2 | True |
| `reports/bridge_author_preflight_results_203498.tar.gz` | 0.00 | 6 | True |
| `reports/bridge_author_reference_results_203699.tar.gz` | 0.00 | 8 | True |
| `reports/bridge_history_results_204258.tar.gz` | 0.00 | 17 | True |
| `reports/bridge_project_condition_results_204017.tar.gz` | 0.00 | 2 | True |
| `reports/bridge_state_rep_results_202463.tar.gz` | 0.01 | 64 | True |
| `reports/bridge_state_results_202278.tar.gz` | 0.01 | 64 | True |
| `training_cache/exports/bridge_expansion_training_v1.zip` | 0.68 | 221 | True |
| `training_cache/exports/multitask_training_v1.zip` | 0.09 | 36 | True |

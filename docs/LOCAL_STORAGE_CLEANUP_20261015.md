# 本地文件清理执行结果 2026年10月15日

按用户授权删除41个文件，文件总大小12.58 GiB。D盘清理后空余约77.94 GiB。没有递归删除目录，没有操作集群文件。

| 删除类别 | 数量 | GiB |
|---|---:|---:|
| verified redundant archive | 9 | 0.81 |
| old latest checkpoint | 32 | 11.77 |

32份对应best.pt在删除后重新计算SHA256，全部与删除前一致。保留两份被多个Bridge脚本直接引用的latest.pt、老师核对依赖、PyBullet权重、原始分片以及202278/204017/204258预测证据。其余旧BC-Z best权重未删除。

删除latest不会删除best，但不能再直接恢复被删的最后训练状态。9个压缩包删除前重新逐成员核验与解压副本一致，解压内容保留。

[删除前计划与保留文件指纹](../reports/local_storage_cleanup_20261015/plan.json)；[逐文件删除记录](../reports/local_storage_cleanup_20261015/execution.json)；[删除后核验](../reports/local_storage_cleanup_20261015/verification.json)。

| 删除文件 | 字节数 |
|---|---:|
| `results/bcz_balanced_fit_v1/latest.pt` | 28519981 |
| `results/bcz_cls_regression_fit_control_v1/latest.pt` | 649942357 |
| `results/bcz_first_command_fit_v1/latest.pt` | 649572437 |
| `results/bcz_first_command_pilot_v1/latest.pt` | 649590357 |
| `results/bcz_fusion_adapter_v1/latest.pt` | 649593301 |
| `results/bcz_fusion_simple_v1/latest.pt` | 28534893 |
| `results/bcz_generalization_regression_v1/latest.pt` | 649960725 |
| `results/bcz_generalization_x0_v1/latest.pt` | 719947237 |
| `results/bcz_gripper_head_legacy_400_v1/latest.pt` | 11748413 |
| `results/bcz_gripper_head_legacy_v1/latest.pt` | 11742077 |
| `results/bcz_gripper_head_mlp2_400_v1/latest.pt` | 13331133 |
| `results/bcz_gripper_head_mlp2_v1/latest.pt` | 13324797 |
| `results/bcz_gripper_only_fit_v1/latest.pt` | 11740861 |
| `results/bcz_native_adapter_fit_lr_low_v1/latest.pt` | 651385765 |
| `results/bcz_native_adapter_fit_lr_split_v1/latest.pt` | 651385765 |
| `results/bcz_native_adapter_fit_v1/latest.pt` | 651385701 |
| `results/bcz_native_pool_adapter_lr_split_v1/latest.pt` | 651443429 |
| `results/bcz_native_pool_adapter_smoke_v1/latest.pt` | 651398629 |
| `results/bcz_native_pool_adapter_v1/latest.pt` | 651443365 |
| `results/bcz_native_pool_baseline_v1/latest.pt` | 30384373 |
| `results/bcz_native_pool_smoke_v1/latest.pt` | 30339573 |
| `results/bcz_native_sequence_fit_v1/latest.pt` | 28743277 |
| `results/bcz_overfit_cosine_v2/latest.pt` | 719930917 |
| `results/bcz_overfit_regression_v1/latest.pt` | 649944533 |
| `results/bcz_overfit_x0_v1/latest.pt` | 719931109 |
| `results/bcz_patch_pool_regression_fit_v1/latest.pt` | 649942357 |
| `results/bcz_pool_seen_probe_fit_v1/latest.pt` | 651385829 |
| `results/bcz_pool_seen_probe_rotation4_v1/latest.pt` | 651385893 |
| `results/bridge_fixed_fit_constant_lr_v1/latest.pt` | 128044469 |
| `results/bridge_fixed_fit_v1/latest.pt` | 128044469 |
| `results/bridge_single_step_v1/training/latest.pt` | 127845813 |
| `results/bridge_single_task_pilot_v1/latest.pt` | 128023221 |
| `reports/bridge_author_obs_results_203876.tar.gz` | 106850 |
| `reports/bridge_author_preflight_results_203498.tar.gz` | 52456 |
| `reports/bridge_author_reference_results_203699.tar.gz` | 974946 |
| `reports/bridge_history_results_204258.tar.gz` | 4785430 |
| `reports/bridge_project_condition_results_204017.tar.gz` | 371736 |
| `reports/bridge_state_rep_results_202463.tar.gz` | 14665124 |
| `reports/bridge_state_results_202278.tar.gz` | 14678062 |
| `training_cache/exports/bridge_expansion_training_v1.zip` | 727974763 |
| `training_cache/exports/multitask_training_v1.zip` | 101015610 |

# 首个完整学习策略的LIBERO工程执行

seed42无增强已完成65,536更新。下载模型526,910,896字节的文件SHA为`33df7eb93ff0d0f7d4ed470643f45ccabb3c8d2c001dc81396f4b66f2c6516d3`，全部321个学习张量SHA和完整更新记录核验通过。

本地Torch2.11/cu128与训练Torch2.5.1/cu118重建的冻结摘要不同。逐项清单证明639个张量相同、编码器没有差异，只有两组DDPM后验均值系数末位差，最大1.192e-7。该现象证明派生常量跨运行环境重建不逐位一致，不能据此认定旧两次集群启动失败也是同一原因。

新增`code/frozen_profile_restore.py`与推理加载的显式`frozen_profile_path`参数，使用已有实际预检清单恢复这两组原始常量。它不替换编码器或学习参数；配置、源码、分词器、逐张量和最终原始冻结摘要均精确核验，异常回滚，未放宽任何身份容差。没有修改训练代码、模型算法、训练预算或导出权重。跨环境推理需与模型一起保留`reports/oxe_noaug_seed42_227223/cluster/preflight.json`。

真实学习策略在固定且已排除正式评分的工程任务中完成8次决策和32个20Hz物理步，按原约定每次只执行0.2秒的第一个目标，随后重新观测。命令输入从未知`[0,0]`变为已知显式命令；没有用测量夹爪宽度代替命令，没有读取专家动作或最终测试初态。完整过程约30.49秒、无错误，未统计成功率。初末图像均已目视查看。

这完成了已训练模型→图像/语言→扩散采样→原生OSC控制→重新观测的工程连通。它不是LIBERO正式成绩，也未证明源单位、基座或TCP已经校准。正式任务、固定初始状态和统计协议必须先固定，之后才能做公平的成功率评价；当前评分准入仍为false。

证据：

- `reports/oxe_noaug_seed42_227223/download_verification.json`
- `reports/oxe_noaug_seed42_227223/frozen_runtime_comparison.json`
- `reports/oxe_noaug_seed42_227223/frozen_portability_checks.json`
- `reports/oxe_seed42_libero_engineering_20261026/attempt1/report.json`：原严格检查拒绝，0前向/物理步。
- `reports/oxe_seed42_libero_engineering_20261026/attempt2/report.json`、`chunks.json`及初末图像：真实策略接入通过，无评分。

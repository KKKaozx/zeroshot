# 231297：只恢复评价，不重训

错误是原`evaluate.py`读取不存在的`sampling_steps`。原不可变`selection.json`实际字段为`generation_steps:100`、`generation_batch_size:8`。退出前没有生成指标。原`run_pair.py`已经在启动评价前完成两组检查点验收；新入口仍会原样重验，不依据这份说明跳过任何守卫。

新包独立部署，只将原评价源码在内存中修正字段访问和输出目录/验收目录。旧训练目录及源码、SHA256SUMS、pair_protocol、120窗、3采样seed、模型权重、冻结身份、checkpoint/首批重放/噪声配对守卫和指标算法全部保留。仅加载231297两个已训练的8192步终点，不调用训练入口、优化器、初始化训练权重或LIBERO。未删除原空`evaluation`目录；新结果独立写入`runs/oxe-target-short-eval-fixed-新作业号/`，失败证据保留。

本地三项CPU回归通过：真实选择字段/拒绝错误值、原48文件及固定源码绑定、执行同一评价main先复现KeyError，再用CPU模型/编码器替身和真实选定数字目标走完120窗/两角色/三采样的90次批采样调用、保存及独立复算，并核对旧空评价目录和检查点未写入。替身检查不代表真实模型推理/泛化已经通过；没有使用真实编码器、CUDA或优化器。此前六项训练包装检查未覆盖这一main入口，是本次字段错误漏检的原因。

先验证原48个包文件哈希和sample终点用户报告的权重/合同SHA；推理前再运行原完整`verify_pair`并与231297已保存验收记录逐项比较。任何不一致就停止。采样和特征重新生成只因原入口在开始前退出，不是增加新评价窗口或采样种子。45分钟A40上限，无自动重提。

本地PowerShell上传：

```powershell
scp "D:\ntu_related\dissertation\Zero_shot\cluster_packages\oxe_target_pair_eval_repair_v1.zip" zixiao005@10.97.216.128:/projects/Zeroshot/
```

集群仅提交新评价入口：

```bash
(
set -euo pipefail
cd /projects/Zeroshot
unzip -n oxe_target_pair_eval_repair_v1.zip
cd oxe_target_pair_eval_repair_v1
bash submit_once.sh
)
```

不要重提`oxe_target_pair_short_v1/submit_once.sh`，它是fresh双训练入口。

```bash
JOB=$(cat /projects/Zeroshot/oxe_target_pair_eval_repair_v1/job_id.txt)
tail -n 60 -f "/projects/Zeroshot/logs/oxe-target-eval-fix-${JOB}.out"
```

完成后仅读现有预测并复算指标：

```bash
JOB=$(cat /projects/Zeroshot/oxe_target_pair_eval_repair_v1/job_id.txt)
PYTHONNOUSERSITE=1 /projects/Zeroshot/envs/multitask-preflight-v1/bin/python -B \
 /projects/Zeroshot/oxe_target_pair_eval_repair_v1/verify_repair.py \
 "/projects/Zeroshot/runs/oxe-target-short-eval-fixed-${JOB}"
```

新目录含`report.json`、`predictions_and_targets.npz`、`repair_receipt.json`、`effective_evaluator.py`、原两组`training_pair_verification.json`副本。评价成功与训练成功分别记录；231297原作业的评价失败状态不伪造为成功。本轮尚无生成误差，不推断x0改善。

# 234100原生采样旋转只读追踪：代码与操作

日期2026-10-30。承接用户“下一步要干什么你直接干吧”，已把下一项证据需求落实成独立诊断代码。助手没有SSH或集群提交；不是新训练或继续234100预算。当前仅代码/CPU测试完成，真实端点追踪结果尚未取得。

## 要回答的唯一问题

**234100原生生成过程中，旋转误差在原始x0四元数上是否已形成，四元数范数、分量裁剪、最终单位化又各呈现什么状态？**

现有六文件只保存最终单位化轨迹，无法还原这些中间量。非零四元数的单位化不会改变其代表的旋转，因此不能把“末尾做了单位化”直接当作旋转被改坏的原因。追踪应核实原始范数、分量MSE、方向角距及identity fallback，而非预设修复。

## 固定边界

- 仅加载234100的`final_policy.pt`，CPU先验证最终base/pose-port参数SHA、65536步完成身份及原小结果SHA。不会读取或修改`latest_state.pt`，不恢复Adam续训。
- 复用原`bridge_full_capability_v1`冻结源码、标签/输入和编码器。新包不改写原部署文件，也不带模型/图片/数据副本。
- 固定原评价顺序的训练前8窗、开发前8窗，即全局索引0–7及128–135，使用原采样seed1101/1102/1103。没有按旋转误差或目标选择窗口。
- 对原生100步sample函数只插入CPU张量拷贝。保留全部计算公式和随机调用；开发批前跳过15个原批次的随机数调用，且先以实际训练批验证这种推进与原生采样的RNG消耗一致。
- 16窗×3seed最终输出必须与234100原预测**逐值完全相同**，否则停止，不扩大容差或换种子。此处的GPU前向用于重放和观测，不能算0前向；优化更新严格为0。
- 记录全部100步的原始x0四元数、带噪当前四元数、由原公式重建的clean、裁剪后的clean和最终单位化前轨迹。真值仅在追踪结束后用于CPU分析，不输入当前去噪器。
- 检查参数摘要、冻结身份、梯度None、空Adam、源文件大小/mtime均不变。原评分/gate不替换，不新建策略端点。

这不同于已关闭的2026-10-27 epsilon/条件置换五t检查：本次是234100完整单源x0端点的原生生成链，没有人为构造带噪专家输入，没有条件置换，也不测试新骨干/目标/损失变体。

## 结果怎样解释

| 观测 | 支持进一步检查的方向 | 不能直接推出 |
|---|---|---|
| 原始q范数很小，分量误差与单位化角误差脱节 | 原始去噪输出的范数与方向质量及监督接口 | 单位化改变了非零q的旋转，或加一个正则必然修复 |
| 范数正常，但原始q方向已经差；后处理前后基本一致 | 问题已存在于去噪预测，不能仅归罪最终输出包装 | 已确定是图像、语言、数据或某个网络层 |
| raw x0与按原公式重建clean差异明显 | x0→epsilon→clean数值路径是否有放大 | 只凭某一高噪声步就改采样公式或全部权重 |
| 裁剪频繁且方向指标发生变化 | 报告原过程的分量裁剪作用 | 可以直接关闭clip或换采样器并重训 |
| 重放或身份守卫失败 | 工程准入未成立 | 有了新的性能结论，或可以放宽guard继续 |

原始四元数分量MSE另作范数差与方向差的精确几何分解；这是现有采样路径上的描述量，使用窗口等权和有效尾部mask。不是实际训练批次归一化损失或梯度占比。16窗不能代表192窗或完整10352窗分布。

## 预算

预计含加载2–4分钟，单A40作业硬上限10分钟；构造完成后的记录器上限300秒。没有任何optimizer.step/backward、训练预算、后续任务或自动续跑。

拟预留256MiB输出空间；准入采用保守project du和文件系统free，仍不等于权威quota。保存小trace/report，不复制权重或大缓存，不清理既有资产。

## ZIP与操作

包：`D:\ntu_related\dissertation\Zero_shot\cluster_packages\bridge_rotation_trace_v1.zip`，14481字节，8成员。

SHA256：`06a5e80a94b60f1e7252f989fb6e7c755a3581d6816d328c0d02ab2293085f84`。

本机PowerShell上传：

```powershell
scp "D:\ntu_related\dissertation\Zero_shot\cluster_packages\bridge_rotation_trace_v1.zip" zixiao005@10.97.216.128:/projects/Zeroshot/
```

用户已登录的集群终端：

```bash
cd /projects/Zeroshot
unzip -n bridge_rotation_trace_v1.zip
cd bridge_rotation_trace_v1
bash submit_once.sh
```

脚本先做登录节点CPU元数据/真实策略张量身份检查，再申请一次typed A40。成功回执`BRIDGE ROTATION TRACE: 编号`只说明提交，不表示已开始或诊断通过。防重复记录及锁不能删除以绕过准入。

查看日志：

```bash
JOB=$(cat /projects/Zeroshot/bridge_rotation_trace_v1/job_id.txt)
tail -n 80 -F "/projects/Zeroshot/logs/bridge-rotation-trace-${JOB}.out"
```

正常完成应有六条`ROTATION_TRACE_REPLAY_EXACT`、`ROTATION_TRACE_RESULT`的0更新/源不变，以及`ROTATION_TRACE_SMALL_RESULTS`和`ROTATION_TRACE_COMPLETE`。

## 下载与CPU复核

输出`/projects/Zeroshot/runs/bridge-rotation-trace-编号/small_results.zip`只含report.json、trace.npz和effective_sample.py，不含权重。下载时核对日志打印的SHA。

本机PowerShell，用实际编号替换：

```powershell
$jobId = "替换为实际作业编号"
$resultDir = "D:\ntu_related\dissertation\Zero_shot\reports\bridge_rotation_trace_$jobId"
New-Item -ItemType Directory -Force -Path $resultDir
scp "zixiao005@10.97.216.128:/projects/Zeroshot/runs/bridge-rotation-trace-$jobId/small_results.zip" "$resultDir\small_results.zip"
Get-FileHash -Algorithm SHA256 -LiteralPath "$resultDir\small_results.zip"
Expand-Archive -LiteralPath "$resultDir\small_results.zip" -DestinationPath "$resultDir\cluster"
& "D:\ntu_related\dissertation\Zero_shot\training_cache\libero_native_env_v1\Scripts\python.exe" -B "D:\ntu_related\dissertation\Zero_shot\deployment\bridge_rotation_trace_v1\verify_trace.py" "$resultDir\cluster" --source-results "D:\ntu_related\dissertation\Zero_shot\reports\bridge_full_capability_234100\cluster"
```

该复核对照已经下载的234100原预测/目标/mask和新trace，重新计算全部100步摘要；源权重没有本地下载，不能将集群报告的权重不变身份冒称本地字节验收。

## 本地验证范围

`tests/test_bridge_rotation_trace.py`三项CPU测试通过（原生100步公式、真实数值标签、CPU替身网络）。覆盖精确原输出重放和随机数跳过、错误预测/模型身份拒绝、参数/梯度/空Adam不变、零范数处理、保存数组复算和被篡改指标拒绝。

Python AST及Shell LF/shebang、A40/10分钟/no-requeue文字检查通过；未实际执行Bash、DINO/CUDA、SSH或提交。机器回执见`reports/bridge_rotation_trace_validation_20261030.json`。旧234100及旧部署包均保持冻结。

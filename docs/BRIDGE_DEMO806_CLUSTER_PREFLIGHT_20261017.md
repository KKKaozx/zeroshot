#806条演示：集群资源与一致性预检

本轮已完成：正式作业208207的7文件本地复核通过，但开发集四项联合验收未通过；保留405参照，不自动追加训练或评分预留25条。见[完整结果](BRIDGE_DEMO806_RESULT_208207_20261017.md)。以下提交说明为历史记录。

用户已报告预检作业208194完成（COMPLETED 0:0，6分49秒，A6000）；日志显示旧四窗上下文/预测重放差异均为0。完整报告与资源门槛仍需validate_preflight.py核验；正式训练尚未收到提交编号。日志：`/projects/Zeroshot/logs/bridge-demo806-preflight-208194.out`。提交记录见`reports/bridge_demo806_preflight_submission_208194.json`。

2026-10-17：传输包和入口准备完成，未提交任何新作业。只提交一次预检；收到完整通过证据后才提交正式训练。

## 本轮内容

原405条演示/1836窗 →806条/3624窗；新增401条已通过CPU源检查。原10条开发演示/40窗不变。五任务各更新槽位、网络、普通x0损失、seed42、8192次更新、batch8及学习率策略不变。实际每窗平均抽样18.08次，属于固定预算覆盖对照。

仅新增401条完整记录按原编号导出到独立目录，空记录保留索引。原数据、旧子集和1876行缓存不覆盖。13份数据zip和1份代码zip合计约658.5MiB，无权重。每份数据包小于64MiB；所有记录、源payload SHA、压缩成员、编号和采样已检查，中间未压缩副本在校验后删除。本地没有生成完整token缓存，没有加载或运行CLIP/Adapter/动作头。

新25条预留演示只有元数据，未导出其记录或目标；已使用208056的13条也未导出。不要重新用208056调参或评分。

## 本地PowerShell上传

```powershell
$demo806Files = @((Get-ChildItem "D:/ntu_related/dissertation/Zero_shot/cluster_packages/bridge_demo806_*.zip").FullName)
$demo806Files += "D:/ntu_related/dissertation/Zero_shot/cluster_packages/bridge_demo806_v1.sha256"
scp @demo806Files zixiao005@10.97.216.128:/projects/Zeroshot/
```

## 集群只提交预检

```bash
cd /projects/Zeroshot
sha256sum -c bridge_demo806_v1.sha256 &&
unzip -n bridge_demo806_code_v1.zip &&
/projects/Zeroshot/envs/multitask-preflight-v1/bin/python bridge_demonstration_expansion_v1/unpack_data.py &&
DEMO806_CHECK=$(sbatch --parsable bridge_demonstration_expansion_v1/preflight.sh) &&
echo "806演示预检：$DEMO806_CHECK"
```

解包只写新的bridge_demonstration_expansion_v1/data目录，不覆盖原bridge_expansion_training_v1/data。检查已有文件时仅接受相同哈希，不自动覆盖冲突文件。

日志与报告：`/projects/Zeroshot/logs/bridge-demo806-preflight-JOBID.out`、`/projects/Zeroshot/runs/bridge-demo806-preflight-JOBID/report.json`。可用sacct查看状态，tail -n20查看末尾，不需贴全部日志。

## 预检检查什么

- 集群完整验证816条源序列/3664个输入窗，复用旧几何核验和401条新增源检查，检查旧输入身份和新增完整数值目标/当前状态。
- 新图像以原始编码SHA、冻结预处理代码和96个固定归一化像素容差1e-6作本地/集群交叉核对，不把它描述成完整图像逐元素比较。
- 原1876行CLIP token逐元素复用，只编码1788行新增训练窗口；新缓存预计约3.67GiB，位于集群。
- 用既有207558检查点对旧4窗重放：上下文与预测最大差门槛均维持1e-4，保存实际数组供核对；恢复原上下文执行范围及参数梯度标志，不放宽门槛。
- 恢复相同初始化后做16次运行检查，验证梯度、冻结参数和优化器分支恢复。它不以8次更新的拟合好坏判断正式效果，不评分新25条或候选完整开发集。
- 资源门槛：保留显存小于12GiB，新缓存小于5GiB，预计正式耗时低于3.5小时；原7.5GiB门槛随缓存行数扩充明确调整，不代表放宽科学成功标准。项目200GiB及15GiB余量是保守规划检查，不是实时配额查询。

## 执行入口与可追溯性

原207558入口文件不修改。包装器只调整1836→3624、1876→3664、资源门槛、输入helper和预检重放挂钩。advance()优化循环和context()函数AST完全一致；模型、噪声、损失及优化器参数保持原值。实际执行的是经过尺寸适配的源码，effective_runner_snapshot.py和wrapper_snapshot.py另存；runner_snapshot.py仍是未改的原入口，不将两者混淆。

本地通过的是实际401条传输记录/305个稀疏子分片、13数据压缩包、完整数值参考、任务槽位、代码结构、CPU索引映射及Bash语法检查；不声称实际模型/GPU预检已通过。

预检完成后先取报告，用validate_preflight.py核对全部条件，再允许fit.sh；需明确提供DEMO806_PREFLIGHT_JOB。不要现在提交正式训练，也不要同时排多个正式分支。正式只跑一个806池、8192次更新、最终开发集验收；不重训旧对照，不加步数或额外种子。

[准备记录](../reports/bridge_demo806_cluster_preparation_20261017.json)、[本地检查](../cluster_packages/bridge_demonstration_expansion_v1/local_validation.json)、[数据盘点方案](BRIDGE_DEMONSTRATION_EXPANSION_20261017.md)。

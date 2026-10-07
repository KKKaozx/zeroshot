# 固定模型的步间噪声检查

复用probe_bridge_conditioning.py，增加--chain-probe模式；不更改生产models.py。
加载190189 latest，严格匹配190853的checkpoint、归档模块、80窗口、种子和参数摘要。
训练40窗口（每任务8条演示各1窗）、开发40窗口，均沿用上一轮顺序。

两组都直接调用归档model.sample：

- native_stochastic：原posterior_variance，100步生产采样。
- no_step_noise：仅暂时将posterior_variance置零，保留原均值公式和100步，结束后恢复。

初始高斯噪声仍存在，三组种子均为0/1/2；两条链使用相同种子，仍消费同样数量的
随机抽样。检查必须确认t99输入与初次x0预测逐元素相同，之后才允许比较。
参数不更新，无优化器/反传；方差buffer在finally恢复，并检查可训练参数哈希及无梯度。

通过前向hook观察生产去噪器，记录t=99/89/74/49/24/9/0的原始x0预测误差、
带噪状态分量MSE，以及最终输出的逐任务位置/旋转误差。不向模型提供真实未来动作。
目标仅在前向结束后用于评分。中间x0尚未经过最终位移裁剪/四元数归一化，
最终完成动作使用生产处理，比较时保留这一差异。
原随机组还与190853正确条件的完整采样指标核对，记录重放差值，不隐藏硬件差异。

这只判断删除步间随机量是否改善固定样本的目标误差、在哪些步骤出现变化。
改变采样噪声会改变输出分布；即使误差降低也不是采样公式错误或任务成功的证明。
后续不能据此直接修改训练或作为新演示泛化结果。实验GPU部分尚未执行。

CPU三项检查通过：hook不改变生产输出/RNG、移除步间噪声保持初始状态与初次预测、
采样异常后hook仍能移除。另复读真实80窗口并匹配190853清单。无权重下载。

本地PowerShell上传：

```powershell
scp "D:/ntu_related/dissertation/Zero_shot/training_cache/exports/bridge_chain_probe_v1.zip" "D:/ntu_related/dissertation/Zero_shot/training_cache/exports/bridge_chain_probe_v1.sha256" zixiao005@10.97.216.128:/projects/Zeroshot/
```

集群先校验解压：

```bash
cd /projects/Zeroshot
sha256sum -c bridge_chain_probe_v1.sha256 && unzip -n bridge_chain_probe_v1.zip
```

成功后提交（默认A40，复用已通过的环境）：

```bash
CHAIN_JOB=$(sbatch --parsable bridge_chain_probe_v1/bridge_chain_probe.sh)
echo "采样检查作业：$CHAIN_JOB"
squeue --me
```

日志/projects/Zeroshot/logs/bridge-chain-作业编号.out，
结果/projects/Zeroshot/runs/bridge-chain-作业编号.json。
读取190853留在集群的报告用于身份核对，不需要重新上传它或下载权重。

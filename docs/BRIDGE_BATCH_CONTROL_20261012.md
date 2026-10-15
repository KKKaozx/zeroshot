# 固定32窗口：训练批次组织对照

日期：2026-10-12。状态：修正版集群对照198871已完成执行检查，见[六组结果](BRIDGE_BATCH_RESULT_198871.md)。

后续：集群预检198833在归一化图像SHA256检查失败，尚未训练。
CPU诊断198856已证实原始数据一致、处理后差异小于4.8e-7，原浮点哈希检查误报。
现在使用[修正版v2操作](BRIDGE_BATCH_CONTROL_V2_20261012.md)；本文下方v1命令保留作历史记录。

此前，32条演示各取一个窗口的拟合通过；同样32个窗口改为16条演示各取两个窗口时，只有25/32通过。随后，反复失败的微波炉早期窗口单独学习或与晚期窗口一起学习，在较大固定预算下都能拟合。它不是一个已经证明无法学习的坏样本，但较大训练集合的问题仍未定位。

这一步固定此前失败的32个窗口，仅比较训练批次的组织方式。没有更换数据标签、CLIP、Adapter、U-Net、噪声表或采样器。

| 条件 | 混合组 mixed | 重复组 homogeneous |
|---|---|---|
| 每批8份样本 | 8个不同窗口 | 同一个窗口重复8份，各份独立噪声/dropout |
| 更新次数 | 2048 | 2048 |
| 每个窗口抽样份数 | 恰好512 | 恰好512 |
| 独立初始化 | 42、43、44 | 与混合组逐一相同 |
| 训练目标 | 位姿 x0 MSE | 相同 |
| 优化器 | Adam，固定学习率1e-4，梯度裁剪1 | 相同 |
| 生成评估 | 原生100步余弦DDPM，3个采样种子 | 相同 |

两组共同使用此前诊断中的“当前图像重复两次，经共享Adapter及零初始化历史投影融合”的实验上下文结构。它与早期完整训练的单次Adapter结构不同；本次不能仅凭与旧报告的差异归因。

噪声/dropout随机键按全局更新配对，因为批次中窗口不同，不能声称每个窗口得到逐元素相同的训练噪声。时间顺序也随批次组织改变，属于这一干预的组成部分。

正式运行同时评估全部32个训练窗口，以及固定10条独立留出演示的全部40个窗口。留出标签不参与优化。报告分别给出窗口平均、每条演示结果和演示平均；重叠窗口不是40个独立样本。这套留出数据已被多次用于开发，不能当作未经接触的最终测试集，也不代表未见任务泛化。

训练拟合门槛沿用已有协议：每个采样种子的整体路径误差≤5mm/5°、整体终点误差≤10mm/10°，至少90%的窗口在全部采样种子下通过路径门槛（32个窗口需29个通过）。夹爪冻结，不汇报抓取成功或夹爪效果。

结果的解释提前固定：

- 如果重复组在多个初始化下训练拟合明显更好，说明批次组织影响当前配方的优化；这还不能定位唯一故障层。
- 如果两组训练都好而留出都差，后续重点转向独立演示泛化。
- 如果两组训练都差，512份/窗口的这项干预未解决共同训练问题，不继续宣称仅是泛化失败。
- 留出改善要结合位置、旋转、终点及逐演示结果判断，不能只选一个更好看的指标。

## 已完成的预检

本地两组各8次更新，仅评估前4个训练窗口和前4个留出窗口。全部72个输入/标签均已读取核对；32个训练窗口与旧报告的图像、标签逐字节一致。初始化相同，Adapter、U-Net和历史投影均有梯度且参数变化；CLIP及其他冻结参数保持不变，重放指标一致。

用独立NumPy浮点64计算核对四份保存预测的路径与终点指标，结果在容差内一致。Python语法、Bash语法、LF换行、内部SHA256及ZIP逐文件内容核对均通过。

耗时71.95秒，峰值保留显存3.873GiB，上传包60,534字节。预检通过只表示可以执行，**不表示训练拟合或泛化通过**。没有安装包、下载模型/数据、保存大检查点或运行仿真。

证据：`reports/bridge_batch_preflight_20261012/report.json`、`verification.json`。固定协议与源码快照：`cluster_packages/bridge_batch_control_v1/manifest.json`。

## 集群操作

在本地 PowerShell 上传：

```powershell
scp "D:/ntu_related/dissertation/Zero_shot/cluster_packages/bridge_batch_control_v1.zip" "D:/ntu_related/dissertation/Zero_shot/cluster_packages/bridge_batch_control_v1.sha256" zixiao005@10.97.216.128:/projects/Zeroshot/
```

在集群 Bash 中校验、解压并提交预检：

```bash
cd /projects/Zeroshot
mkdir -p logs
sha256sum -c bridge_batch_control_v1.sha256 && unzip -n bridge_batch_control_v1.zip
BATCH_CHECK=$(sbatch --parsable /projects/Zeroshot/bridge_batch_control_v1/preflight.sh)
echo "批次对照预检作业：$BATCH_CHECK"
squeue --start -j "$BATCH_CHECK"
```

完成后检查实际作业状态和报告末尾；排队时日志尚不存在是正常的。

```bash
sacct -j "$BATCH_CHECK" --format=JobID,State,ExitCode,Elapsed
tail -n 12 "/projects/Zeroshot/logs/bridge-batch-preflight-$BATCH_CHECK.out"
```

预检完成并确认报告 `execution_checks_passed=true` 后，正式训练可用以下命令提交。它会再次核对预检、数据、初始化和首步损失。断连后将 `$BATCH_CHECK` 设置为真实预检编号，避免空变量。

```bash
BATCH_FIT=$(sbatch --parsable --dependency="afterok:$BATCH_CHECK" --export="ALL,BATCH_PREFLIGHT_JOB=$BATCH_CHECK" /projects/Zeroshot/bridge_batch_control_v1/fit.sh)
echo "批次对照正式作业：$BATCH_FIT"
squeue --start -j "$BATCH_FIT"
sacct -j "$BATCH_FIT" --format=JobID,State,ExitCode,Elapsed
tail -n 12 "/projects/Zeroshot/logs/bridge-batch-fit-$BATCH_FIT.out"
```

复用 `/projects/Zeroshot/envs/multitask-preflight-v1`、`bridge_expansion_training_v1/data` 和 `baseline_setup/clip-vit-large-patch14-v1`。默认申请一张A40。正式任务上限2小时，排队时间另算。只保存小报告/预测，不复制数据或下载权重。

完成后下载 `/projects/Zeroshot/runs/bridge-batch-fit-实际编号/` 中的报告和NPZ。正式结果尚未产生。

# 批次对照v2：修正跨环境浮点校验

集群CPU作业198856核对全部72个窗口：原始图片、解码RGB、state/action/指令完全一致。处理后图像最大差异4.76837158203125e-7，目标分量最大差异4.470348358154297e-7，当前夹爪差异0。此前198833被跨环境浮点SHA256检查误拦截，尚未训练。

v2只修正输入检查：严格固定原始输入及参考文件SHA256，浮点图像/位姿采用绝对容差1e-6、相对容差0，夹爪类别及当前夹爪仍严格相同。没有修改生产数据处理、模型、训练数据、更新次数、批次顺序、优化器或评估协议。v1失败包保留，v2使用独立目录。

受控测试接受实际测得的末位差异，并拒绝原始图片/像素/state/action/指令/尺寸变化、1e-4级数值变化、非有限值，以及微小的夹爪标签变化。本地v2预检通过；两组的初始化哈希、首步和末步损失、全部预检评估指标与v1本地结果精确相同。CLIP保持冻结。

本地预检91.11秒，峰值保留显存3.873GiB。新ZIP为56,016字节，复用已上传的35MiB参考文件，无需重传它。源码/协议比对、Python/Bash语法、内部校验和及ZIP逐文件核对通过。

证据：`reports/bridge_batch_preflight_v2_20261012/verification.json`、`reports/bridge_batch_v2_guard_verification_20261012.json`。集群198856证据为用户日志摘录，完整JSON尚未下载。这一修复解释了预检退出，没有确定之前拟合/泛化差的原因。

## 上传及预检

本地PowerShell：

```powershell
scp "D:/ntu_related/dissertation/Zero_shot/cluster_packages/bridge_batch_control_v2.zip" "D:/ntu_related/dissertation/Zero_shot/cluster_packages/bridge_batch_control_v2.sha256" zixiao005@10.97.216.128:/projects/Zeroshot/
```

集群Bash：

```bash
cd /projects/Zeroshot
mkdir -p logs
sha256sum -c bridge_batch_control_v2.sha256 && unzip -n bridge_batch_control_v2.zip
BATCH_CHECK=$(sbatch --parsable /projects/Zeroshot/bridge_batch_control_v2/preflight.sh)
echo "修正版预检编号：$BATCH_CHECK"
```

检查：

```bash
sacct -j "$BATCH_CHECK" --format=JobID,State,ExitCode,Elapsed
tail -n 12 "/projects/Zeroshot/logs/bridge-batch-preflight-v2-$BATCH_CHECK.out"
```

报告：`/projects/Zeroshot/runs/bridge-batch-preflight-v2-实际编号/report.json`。须完成且 `execution_checks_passed=true`。预检不代表拟合/泛化成功。

## 正式训练（集群修正版预检通过后）

```bash
BATCH_FIT=$(sbatch --parsable --dependency="afterok:$BATCH_CHECK" --export="ALL,BATCH_PREFLIGHT_JOB=$BATCH_CHECK" /projects/Zeroshot/bridge_batch_control_v2/fit.sh)
echo "正式批次对照编号：$BATCH_FIT"
squeue --start -j "$BATCH_FIT"
```

两组各2048次更新、每个训练窗口512份抽样，3个独立初始化。最终评估32个训练窗口与10条留出演示的全部40个窗口，使用原生100步DDPM和3个采样种子。作业198871已完成执行检查；见[六组结果和下一步](BRIDGE_BATCH_RESULT_198871.md)。数据集、环境和模型权重沿用现有资产，不保存大检查点。

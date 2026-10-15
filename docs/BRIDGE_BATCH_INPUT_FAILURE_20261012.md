# 作业198833：输入一致性检查失败

集群作业198833在46秒后失败，位置为 `check_bridge_batch_control.py` 中的归一化图像SHA256断言。它尚未加载CLIP或执行优化更新。TensorFlow的cuDNN/cuFFT/cuBLAS注册日志不是这次退出的直接异常。

原始断言要求不同环境的浮点图像逐字节一致，但失败时没有输出窗口身份、差异大小或原始像素核对结果。随后作业198856提供了差异证据，见下方更新。

没有删除断言或直接放宽阈值，没有修改生产dataset/Adapter/U-Net，也没有改变两组训练预算。原始上传包和失败配方保留。

## CPU诊断内容

新诊断核对原来的全部72个窗口：

- 选中图片的原始编码字节及解码RGB字节、尺寸；
- 演示完整的原始state/action浮点字段，以及当前指令字节；
- 最终归一化图像、16步动作标签、当前夹爪输入的最大差异和平均差异；
- 第一处归一化图像哈希不同的窗口，以及本地与集群的Python/Torch/NumPy/Pillow版本。

仅测量差异，不跳过训练检查，也不自动给出“数值差异无影响”的结论。

在本地同一环境重新读取72个窗口，原始输入、处理后的图像及标签全部一致。另用受控数值变化确认，诊断能分别报告1e-7的浮点变化与0.1的明显像素变化。Python和Bash语法、内部校验和、ZIP逐文件内容校验通过。

## 集群198856结果及修正

用户贴出的集群日志显示诊断完成：72个窗口的编码图片、解码RGB、原始state/action/指令均完全一致；图像最大归一化数值差异4.76837158203125e-7，目标分量最大差异4.470348358154297e-7，当前夹爪差异0。

本地为Windows、Torch2.11.0、NumPy2.2.6；集群为Linux、Torch2.5.1、NumPy1.26.4。这些证据足以解释逐字节哈希检查为何拦住预检，但没有隔离到底哪个依赖/算子造成末位差异，也没有证明之前厘米级拟合误差的根因。

修正版v2保留原始图片、解码像素、状态、动作和指令的严格校验；图像/位姿浮点值采用绝对容差1e-6、相对容差0，夹爪类别与当前夹爪仍要求完全一致。参考文件本身的SHA256也固定。集群预检与正式训练之间仍要求处理后哈希完全一致，以避免同一环境发生无提示变化。

数据窗口、模型、批次顺序、学习率和训练预算保持v1协议。原v1包与失败记录保留。v2复用已上传诊断参考文件，不需要重新上传35MiB参考。

集群证据摘录：`reports/bridge_batch_input_cluster_198856_observed_summary.json`，明确标为用户日志摘录，尚未下载完整集群报告。修正检查的受控变化验证：`reports/bridge_batch_v2_guard_verification_20261012.json`。

下一步使用 `bridge_batch_control_v2` 重新预检；不要再次提交旧v1脚本。

参考数组包含72幅完整的已处理图像，上传包34.84MiB；它不是新的数据集或模型权重。复用已有环境与原始数据，CPU作业上限5分钟，不申请GPU，不安装包，不加载CLIP。

## 操作

本地PowerShell：

```powershell
scp "D:/ntu_related/dissertation/Zero_shot/cluster_packages/bridge_batch_input_audit_v1.zip" "D:/ntu_related/dissertation/Zero_shot/cluster_packages/bridge_batch_input_audit_v1.sha256" zixiao005@10.97.216.128:/projects/Zeroshot/
```

集群Bash：

```bash
cd /projects/Zeroshot
mkdir -p logs
sha256sum -c bridge_batch_input_audit_v1.sha256 && unzip -n bridge_batch_input_audit_v1.zip
INPUT_AUDIT=$(sbatch --parsable /projects/Zeroshot/bridge_batch_input_audit_v1/audit.sh)
echo "CPU输入诊断编号：$INPUT_AUDIT"
```

完成后：

```bash
sacct -j "$INPUT_AUDIT" --format=JobID,State,ExitCode,Elapsed
tail -n 8 "/projects/Zeroshot/logs/bridge-input-audit-$INPUT_AUDIT.out"
```

报告为 `/projects/Zeroshot/runs/bridge-input-audit-实际编号/report.json`。先审查原始输入是否一致及数值差异大小，再决定怎样修正跨环境检查。正式批次对照训练仍未提交。

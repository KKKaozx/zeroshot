# Octo 最小官方复现

2026-10-07。目标是运行已发布完整策略的官方第一项推理示例，作为正对照。暂停继续调整 CLIPort 专家回放或旧动作头。**预检完成；安装与推理包已准备，尚未运行安装或模型。**

用户提供的集群作业 188824 报告确认 x86_64、RTX A6000 可见，项目 apparent use 23.18GiB，inventory_passed=true；octo_inference_ready=false。旧环境 JAX0.4.13/Flax0.7.0/TF2.13 与官方依赖不同，另建 `/projects/Zeroshot/envs/octo-small-v1`。200GB 配额按 200,000,000,000 bytes 做规划；原预检用 200GiB，余量略高估，新脚本已纠正。仍未读取实际配额。

安装作业 188827 在 pip 依赖解析阶段失败：未锁定的新版 wandb 要求 protobuf≥5，与 TF2.15/protobuf4.23.4 冲突；尚未进入模型下载或推理。修复为 wandb==0.16.6，其 PyPI 元数据支持 Linux/Python3.10 下 protobuf≥3.19,<5。补锁后仍需集群安装验证，不能称修复已运行通过。旧依赖等待任务 188828 应取消，再按 afterok 重新提交安装和推理；复用已创建的独立环境。

重试 188832 仍在依赖解析阶段失败：tensorflow-metadata1.14.0 要求 protobuf≥3.20.3,<4.21，与人为固定的4.23.4冲突。保留 wandb0.16.6，将 protobuf 改为3.20.3；检查其满足日志列出的 TF、TFDS、TensorBoard、wandb、TFHub、Orbax、TFMetadata 全部 protobuf 范围。此检查不等于完整依赖安装或运行通过。取消旧等待任务188833，再复用安装环境提交作业。

固定代码 `241fb3514b7c40957a86d869fecb7c7fc353f540`，模型 `rail-berkeley/octo-small-1.5` revision `dc9aa3019f764726c770814b27e4ab0fc6e32a58`，T5 资源 revision `a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1`。checkpoint 546,696,551 bytes。模型和环境只在集群下载；T5 只下载配置及 tokenizer，语言编码器参数来自 Octo checkpoint。官方代码未替换为本项目网络。

CPU 安装作业使用官方 requirements 加关键兼容约束，固定 JAX0.4.20 CUDA11 GPU wheel、NumPy1.24.3/Flax0.7.5/TF2.15；记录 pip freeze 并执行 pip check。其余传递依赖仍由 pip 解析，不能称所有依赖均已锁定或安装验证通过。安装前要求 15GiB 规划余量，安装后检查新增占用不超过 15GiB；此检查不限制安装期间的峰值占用。保留原有环境，不下载完整 OXE 数据。本地包只有脚本和配置。

GPU 作业复现 [官方 notebook 第一项](https://github.com/octo-models/octo/blob/241fb3514b7c40957a86d869fecb7c7fc353f540/examples/01_inference_pretrained.ipynb)：Bridge 示例 JPEG、RGB256×256、`pick up the fork`、历史长度1、有效时间掩码、seed0，使用 `bridge_dataset` 动作统计反归一化。GPU 阶段离线读取缓存；JAX 回退 CPU 则失败。检查输出 `[1,4,7]` 且全部有限，保存原生动作和统计。未运行 notebook 下载整条远程数据的第二项，不使用项目 `[16,8]` 或 CLIP 预处理。

在本地 PowerShell 上传：

```powershell
scp "D:/ntu_related/dissertation/Zero_shot/results/octo_preparation/octo_official_smoke_v1.zip" "D:/ntu_related/dissertation/Zero_shot/results/octo_preparation/octo_official_smoke_v1.sha256" zixiao005@10.97.216.128:/projects/Zeroshot/
```

在 VS Code 集群终端执行；安装成功后 GPU 推理才具备运行条件：

```bash
cd /projects/Zeroshot
mkdir -p logs baseline_setup
sha256sum -c octo_official_smoke_v1.sha256 && unzip -n octo_official_smoke_v1.zip &&
OCTO_SETUP=$(sbatch --parsable octo_official_smoke_v1/octo_setup.sh) &&
OCTO_RUN=$(sbatch --parsable --dependency=afterok:"$OCTO_SETUP" octo_official_smoke_v1/octo_inference.sh) &&
printf '安装作业：%s\n推理作业：%s\n' "$OCTO_SETUP" "$OCTO_RUN"
```

在同一终端查询：

```bash
sacct -j "$OCTO_SETUP,$OCTO_RUN" --format=JobID,State,ExitCode
tail -n 60 "/projects/Zeroshot/logs/octo-setup-$OCTO_SETUP.out"
tail -n 60 "/projects/Zeroshot/logs/octo-inference-$OCTO_RUN.out"
```

排队时日志可能不存在，正常。安装失败时推理可能保持依赖等待，先检查安装日志；不要重复提交。完成后取回 `baseline_setup/octo-inference-JOBID.json` 小报告，无需下载权重。CPU 下载成功与 GPU 加载成功分别验收，运行前不宣称复现通过。

推理通过仅证明官方模型加载、GPU执行和有限值动作输出，没有动作数值参考对照，也没有机器人任务成功率。Octo 预训练包含 Bridge、BC-Z、Fractal、Language Table，在本地 Bridge 上表现好不能证明未见数据泛化。后续比较需统一划分、动作合同和评价指标。来源：[官方代码](https://github.com/octo-models/octo)、[权重](https://huggingface.co/rail-berkeley/octo-small-1.5)、[论文项目](https://octo-models.github.io/)。

# Octo 最小官方复现

2026-10-07。目标是建立一个完整已发布策略的正对照，停止继续改同一条 CLIPort 专家回放。当前只完成版本固定与预检脚本准备，**未运行 Octo，未安装依赖或下载权重**。本地非交互 SSH 连接集群地址超时，需在用户已有 VS Code 远程终端提交。

固定代码提交 `241fb3514b7c40957a86d869fecb7c7fc353f540`，模型 `rail-berkeley/octo-small-1.5` 的 revision 为 `dc9aa3019f764726c770814b27e4ab0fc6e32a58`。权重 checkpoint 文件 546,696,551 bytes，不能用“27M 参数”推断所有保存文件大小。模型、语言资源、独立环境与缓存全部放集群项目 SSD；本地仅小型脚本和报告，不下载完整 OXE 数据。

第一阶段复现 [官方 notebook](https://github.com/octo-models/octo/blob/241fb3514b7c40957a86d869fecb7c7fc353f540/examples/01_inference_pretrained.ipynb) 的第一张 Bridge 示例图与 `pick up the fork`：历史长度 1、有效时间掩码、seed=0，使用 `bridge_dataset` 的动作统计反归一化，期望输出 `[1,4,7]`。不运行 notebook 中下载整条远程数据的第二段，不沿用项目 `[16,8]` 或 CLIP 图像预处理。先检查可加载、GPU 实际执行、输出有限值、动作统计及形状，再进行现有 Bridge 训练分区的原生合同核查。

官方 requirements 固定 NumPy1.24.3/JAX0.4.20/Flax0.7.5/TF2.15，与旧环境不能假定兼容；部分依赖没有上界。预检先查 x86_64、分配 GPU、现有环境版本及项目空间，之后另建独立环境，固定兼容的传递依赖和 GPU wheel，保留旧环境。预检通过不等于依赖安装或 Octo 推理通过。

约束：用户报告项目配额 200GB，预检以 du apparent bytes 做保守规划并要求至少 15GiB 余量；它不读取实际配额，TB 级文件系统余量不能当作项目余量。完整环境的最终大小、语言资源和 GPU 依赖预算在安装前核查。预检本身不安装、不下载、不训练，申请 A6000 一张卡仅读取 GPU 信息，最多 10 分钟。

本地小包：`results/octo_preparation/octo_minimal_preflight_v1.zip`（约 3KB）及同名 `.sha256`。包含本仓库三份文件：`scripts/cluster/octo_preflight.py`、`scripts/cluster/octo_preflight.sh`、`configs/octo_minimal_reproduction.json`。Python AST、JSON 解析与 Bash 语法检查通过；未执行集群作业。

在本地 PowerShell 上传：

```powershell
scp "D:/ntu_related/dissertation/Zero_shot/results/octo_preparation/octo_minimal_preflight_v1.zip" "D:/ntu_related/dissertation/Zero_shot/results/octo_preparation/octo_minimal_preflight_v1.sha256" zixiao005@10.97.216.128:/projects/Zeroshot/
```

在 VS Code 集群终端校验、解压并提交；解压不覆盖已有同名文件：

```bash
cd /projects/Zeroshot
mkdir -p logs baseline_setup
sha256sum -c octo_minimal_preflight_v1.sha256 && unzip -n octo_minimal_preflight_v1.zip && sbatch octo_minimal_preflight/octo_preflight.sh
```

随后用返回的数字 JOBID 查状态和日志：

```bash
sacct -j JOBID --format=JobID,State,ExitCode
cat /projects/Zeroshot/logs/octo-preflight-JOBID.out
cat /projects/Zeroshot/baseline_setup/octo-preflight-JOBID.json
```

这次只需贴回 JSON 小报告；未运行推理前，不依据纸面配置宣称复现通过。官方预训练包含 Bridge 等目标来源，在本地 Bridge 数据上预测好也不能证明独立新数据泛化。后续受控训练比较须统一划分、动作目标与评价协议，完整机器人论文成功率仍需对应硬件或正式基准。官方来源：[代码](https://github.com/octo-models/octo)、[权重](https://huggingface.co/rail-berkeley/octo-small-1.5)、[论文项目](https://octo-models.github.io/)。

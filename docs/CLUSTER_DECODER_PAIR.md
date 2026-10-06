# 下一次降噪器对照：上传、预检、训练

准备日期：2026-10-06。历史操作指南：setup 187755、GPU 预检 187756 和训练 187797 均已完成。当前组拟合训练集但开发失败，作者组后期退化且最终拟合不足。结果见 [PROGRESS.md](PROGRESS.md)；不要再次提交下面的原始作业。下一轮有限方案见 [NEXT_EXPERIMENT.md](NEXT_EXPERIMENT.md)，尚未执行。

## 实验边界

两组顺序占用同一张 A6000：当前 160 万参数降噪器与作者配置的 2.77 亿参数降噪器。相同冻结图文上下文、夹爪头、316/55 窗口、16 步动作、x0 目标、100 步余弦表、裁剪范围和训练噪声/窗口清单。

每组 9875 更新，每更新有效 batch=64，通过 4 个 microbatch=16 累积梯度；每个训练窗口抽样恰好 2000 次。Adam lr=3e-4、无 weight decay、梯度裁剪 1。只用固定最终权重，全量训练/开发指标及逐演示结果；最终测试目标不使用。

独立训练初始化只做一次（seed=42），采样种子为 1101/1102/1103，不能称作三个训练种子。不同网络参数不能相同；只核对相同训练输入和冻结夹爪权重。此比较同时改变网络容量和条件融合，属于整体降噪器对照，不是单因素消融。

新两组明确使用共同 CPU 噪声流、梯度累积和 eval batch=16，不能把旧的 186047 结果当作完全匹配的控制组。本轮不联合训练 Adapter，不复现作者完整视觉策略或论文任务分数。

## 本地文件与容量

文件已经准备在 `D:\ntu_related\dissertation\GPU cluster`：

- `decoder_pair_v1.zip`：约 7.62 MiB，含现有冻结特征、初始小动作头和必要代码；不含原始数据或 CLIP。
- `decoder_pair_v1.sha256`：传输校验。
- `decoder_pair_setup.sh`：CPU 作业，校验、安装 einops、解包。
- `decoder_pair_preflight.sh`：15 分钟 GPU 资源预检。
- `decoder_pair_train.sh`：训练入口，必须提供通过的预检 JSON。

预检会对各网络做 4 次临时更新测量 Adam 状态和梯度显存，再测真实 100 步采样耗时；权重随后丢弃，不参与正式训练。正式训练重新初始化，不继承预检权重。预检通过要求显存保留量 <90% 总显存、估计两组耗时 <6 小时时限的75%。估计不是保证；不通过时不提交训练。

集群仅保存两份最终动作头（合计约 1.1 GiB）、小型预测数组及报告；不存多份优化器检查点，任务中断需要重跑。大权重留在集群。本地后续只接收小报告和图例，不自动下载完整结果。

## 第一步：本地 Windows PowerShell 上传

确认 NTU VPN/网络及之前可用的 SSH 连接。以下命令在本地 PowerShell，不能在集群终端执行：

```powershell
Set-Location "D:\ntu_related\dissertation\GPU cluster"
scp decoder_pair_v1.zip decoder_pair_v1.sha256 decoder_pair_setup.sh decoder_pair_preflight.sh decoder_pair_train.sh zixiao005@10.97.216.128:/projects/Zeroshot/
```

## 第二步：集群终端提交 setup 与 GPU 预检

在 VS Code SSH 集群终端执行：

```bash
cd /projects/Zeroshot
mkdir -p logs
PAIR_SETUP=$(sbatch --parsable /projects/Zeroshot/decoder_pair_setup.sh)
PAIR_CHECK=$(sbatch --parsable --dependency="afterok:$PAIR_SETUP" /projects/Zeroshot/decoder_pair_preflight.sh)
echo "环境作业：$PAIR_SETUP；GPU预检作业：$PAIR_CHECK"
squeue --me
```

GPU 预检在 CPU setup 成功后才运行。setup 使用现有 `/projects/Zeroshot/envs/bridge-diffusion`，只补 `einops==0.8.2`，不会重复安装 PyTorch/JAX。若该环境不存在，setup 会退出；不要改成下载整套大环境。

查看状态及输出（同一个 SSH 终端保留变量）：

```bash
sacct -j "$PAIR_SETUP,$PAIR_CHECK" --format=JobID,State,ExitCode
tail -n 80 "/projects/Zeroshot/logs/decoder-pair-setup-$PAIR_SETUP.out"
tail -n 80 "/projects/Zeroshot/logs/decoder-pair-preflight-$PAIR_CHECK.out"
```

`PENDING` 时日志可能尚未创建，并非报错。换终端后变量可能为空，改用刚才输出的真实编号。先把这两份输出和真实作业编号贴回来，不先启动训练。

## 第三步：GPU 预检通过后再训练

需要核对 `preflight.json` 的 passed、峰值显存、总耗时估计，以及 GPU 类型。根据实测估计可缩短申请时限；不要仅凭本地 CPU 运行时间推断集群训练时长。

下面是通过后的命令模板，**不是当前就执行的步骤**：

```bash
export PAIR_PREFLIGHT="/projects/Zeroshot/runs/decoder-pair-preflight-$PAIR_CHECK/preflight.json"
PAIR_TRAIN=$(sbatch --parsable --export=ALL,PAIR_PREFLIGHT="$PAIR_PREFLIGHT" /projects/Zeroshot/decoder_pair_train.sh)
echo "训练作业：$PAIR_TRAIN"
```

训练代码会再次核对 passed、协议指纹与 GPU 型号；不同 GPU 或失败预检不能直接复用。若首个作业排队，正常查看 squeue，不反复重提交制造重复任务。

## 本地已完成的检查

ZIP 逐文件 SHA256 核对、Python 语法、解包后导入、原小模型初始参数精确一致、两次独立生成的训练输入清单一致、9875 次抽样覆盖每个训练窗口2000次。**本地没有优化器更新，没有执行 GPU 预检或学习实验。** 记录见 `reports/decoder_pair_local_checks.json`。

源码入口是打包后的 `run_decoder_pair.py`，依赖同包的缓存模型和评估辅助文件；不要直接把项目 `diagnostics/` 中单个文件拷到集群运行。

集群要求参考：[Slurm 作业与 GPU 资源](https://github.com/NTUEEECluster/docs/blob/main/slurm-guide.md)、[SSD 与缓存](https://github.com/NTUEEECluster/docs/blob/main/storage-guide.md)。安装通过 CPU 作业进行，GPU 工作通过 sbatch 申请；沿用现有 SSD 缓存设置，GPU 作业不额外设置 cpus-per-task。

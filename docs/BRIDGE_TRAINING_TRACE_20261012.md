# 固定配方的训练过程追踪

已完成：集群预检200038监测开/关差异为0，正式追踪200039重现198871最终预测。三个初始化在1024次均通过训练拟合门槛，seed43随后退化。详见[实际结果与下一项对照](BRIDGE_TRAINING_TRACE_RESULT_200039_20261012.md)。下面保留原始协议和复用命令，已完成作业无需重新提交。

目的：判定198871混合组的训练误差是一直没有降下来，还是降下来后又退化。固定原来的32个训练窗口、16条演示、5种任务，不增加数据或更新次数，也不改生产模型。新包为`cluster_packages/bridge_training_trace_v1`，以前的包和结果保留。

## 配方与观测

三个初始化仍为42/43/44；只重复混合批次组。批量8，完整2048次更新，每窗512份抽样；Adam、恒定学习率1e-4、零权重衰减、梯度裁剪1、位姿x0 MSE。原来的两次当前观测Adapter调用、历史投影、冻结CLIP、多尺度U-Net、噪声表和原生100步DDPM保持相同。夹爪不参与训练。

只增加观测：

- 在更新0、64、256、512、1024、1536、2048处，对全部32个训练窗口执行完整100步DDPM，使用1101/1102/1103三个采样种子。保留逐窗口预测、标签、上下文及完整拟合门槛结果。
- 每次训练更新记录损失、Adapter/U-Net/历史投影裁剪前梯度范数、全局裁剪前范数和裁剪比例。梯度非零不等于学得好；这些记录用于识别断梯度、梯度异常或更新受裁剪限制等现象。
- 只在最终更新评估十条留出演示的全部40个窗口，不按开发误差挑选检查点，不提前停止。
- 正式训练在集群保留三个最终可训练参数检查点，seed43额外保留Adam状态，以便后续按证据决定是否继续。整包PT文件上限4GiB；本地只下载小型结果，不下载这些权重。

阶段检查会切换到eval，但在退出时恢复NumPy、Torch CPU和CUDA随机数状态，以及每个子模块的train/eval模式；检查参数版本号和缓冲区未变，评估全过程无梯度。训练每一步使用与198871相同的随机键和窗口顺序。

## 核验与局限

新包复制了八个冻结源文件，包括原始批次对照脚本，不修改它们；新追踪脚本复用原始批次顺序与指标函数。数据的原始身份及数值容差检查沿用v2，参考文件已经上传，无需重传。正式训练还会核对198871报告SHA256、相同集群的输入身份、初始化参数哈希和首步损失。

本地GPU预检从相同初始参数分别跑8次更新，关闭/开启中途观测，比较参数、损失及原生预测；检查随机数和模式恢复。CUDA注意力反向仍按旧配方采用warn-only确定性，预检允许明确记录的小数值容差。不能因为监测随机数被恢复就宣称跨设备训练必定逐位相同。

正式结果会记录与198871最终训练指标的差异。如果同种子的最终误差没有重现此前量级，应先报告“这次重跑不同”，而不能把新曲线当作旧运行的确切历史。该实验不直接定位唯一坏层：先确定训练阶段，再据结果决定是否冻结Adapter做模块隔离，或进行预先限定的预算对照。

详细本地预检证据：`reports/bridge_trace_local_preflight_20261012/report.json`。固定配方/冻结源码核对：`reports/bridge_training_trace_recipe_verification_20261012.json`。上传包携带预检摘要，ZIP只有源码与清单，不含数据或模型。

本地预检已通过：耗时290.31秒，峰值保留显存3.869GiB。监测开/关的参数、损失、最终预测差异全部为0；与之前本地v2预检相比，初始化和首步损失相同，原来的前4个训练及前4个留出窗口最终预测差异也为0。完整72窗口输入检查通过，CLIP及其他冻结参数未变化。八步预检尚未拟合属于预期，不能当作正式结果。详细独立核对位于同目录`verification.json`。

## 提交

在本地PowerShell上传：

```powershell
scp "D:/ntu_related/dissertation/Zero_shot/cluster_packages/bridge_training_trace_v1.zip" "D:/ntu_related/dissertation/Zero_shot/cluster_packages/bridge_training_trace_v1.sha256" zixiao005@10.97.216.128:/projects/Zeroshot/
```

在集群Bash执行一次：

```bash
cd /projects/Zeroshot
mkdir -p logs
sha256sum -c bridge_training_trace_v1.sha256 && unzip -n bridge_training_trace_v1.zip
TRACE_CHECK=$(sbatch --parsable /projects/Zeroshot/bridge_training_trace_v1/preflight.sh)
TRACE_FIT=$(sbatch --parsable --dependency="afterok:$TRACE_CHECK" --export="ALL,TRACE_PREFLIGHT_JOB=$TRACE_CHECK" /projects/Zeroshot/bridge_training_trace_v1/fit.sh)
printf '追踪预检：%s\n正式追踪：%s\n' "$TRACE_CHECK" "$TRACE_FIT"
squeue --start -j "$TRACE_CHECK,$TRACE_FIT"
```

正式追踪只会在预检成功后启动，且脚本再次检查预检报告是否合格。默认A40；预检时限15分钟、正式2小时，时限不是预计运行时间。沿用已有环境，不执行pip/conda安装，不下载新模型或数据。

查看：

```bash
sacct -j "$TRACE_CHECK,$TRACE_FIT" --format=JobID,State,ExitCode,Elapsed,NodeList
tail -n 12 "/projects/Zeroshot/logs/bridge-trace-preflight-$TRACE_CHECK.out"
tail -n 15 "/projects/Zeroshot/logs/bridge-trace-fit-$TRACE_FIT.out"
```

若终端断连导致变量丢失，直接把变量替换成实际作业号。排队时没有日志属于尚未执行，不要据此判断脚本失败。

正式结果目录为`/projects/Zeroshot/runs/bridge-trace-fit-实际编号`。先读取`report.json`中每种子的`history`与`gradient_summary`，再看逐窗口NPZ；`execution_checks_passed=true`只代表脚本检查通过。

## 存储与下载

正式脚本按200GiB规划限额检查当前项目占用，并为本次4GiB预算及15GiB余量预留空间；这是保守规划，不是实时配额查询。预检不保存权重。最终检查点只包含可训练参数，seed43额外包含优化器；重建时必须使用记录的源码、配置、初始化及冻结CLIP，不能当作独立完整模型随意加载。

只下载小文件，在本地PowerShell（把数字替换成正式作业号）：

```powershell
$TraceJobId = '实际正式编号'
$TraceDestination = "D:/ntu_related/dissertation/Zero_shot/reports/bridge-trace-fit-$TraceJobId"
New-Item -ItemType Directory -Force -Path $TraceDestination | Out-Null
scp "zixiao005@10.97.216.128:/projects/Zeroshot/runs/bridge-trace-fit-$TraceJobId/*.json" "zixiao005@10.97.216.128:/projects/Zeroshot/runs/bridge-trace-fit-$TraceJobId/*.jsonl" "zixiao005@10.97.216.128:/projects/Zeroshot/runs/bridge-trace-fit-$TraceJobId/*.npz" "zixiao005@10.97.216.128:/projects/Zeroshot/runs/bridge-trace-fit-$TraceJobId/runner_snapshot.py" "$TraceDestination/"
```

这个下载命令不包含`.pt`，不要直接`scp -r`整个正式输出目录。

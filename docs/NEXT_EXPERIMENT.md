# 下一轮有限实验方案

状态：2026-10-06 已在原入口实现，并准备上传包；尚未提交集群预检或正式训练。该方案补齐作业 187797 作者分支训练拟合不足的缺口，不是老师要求的正式零样本模型实验。

## 要回答的问题

作业 187797 的作者 U-Net 分支在相同训练输入下，后期记录损失明显升高，最终训练位姿误差 4.685 cm / 12.839°。因此不能用该分支排除动作头影响。只检查一个可证伪的假设：较小的学习率能否在相同初始化及更新预算下获得更好的最终训练拟合，并避免已观察到的后期退化？

不认定学习率已经是根因；不以单个训练批次损失判断动作预测质量。

## 唯一训练变量

新作者分支学习率从 3e-4 改为 3e-5。这是预先指定的一个诊断值，不声称为作者推荐值或最优值。对照为已完成的 187797 作者分支，不重新运行已能拟合的小型项目动作头。

其他训练设置保持相同：作者原始 ConditionalUnet1D、相同初始化 seed=42、相同冻结上下文及夹爪参数、相同 316 个训练窗口、相同 CPU 噪声种子 901、相同 9,875 更新和有效 batch=64/microbatch=16、Adam 及梯度裁剪、x0 目标、100 步余弦 DDPM 和输出裁剪。核对初始化、冻结参数及训练输入 SHA256，不能仅凭种子相同宣称输入一致。

不加入演示 ID、时间索引、未来观测或真实目标位姿作为预测输入。不改 Adapter、动作表示、数据、图像预处理或夹爪监督。

## 中途监测与固定结果

在更新 0、2,500、5,000、7,500、9,000、9,500、9,875 时记录完整训练位姿误差，使用固定采样种子 1101；监测只用训练分区。记录训练梯度范数和最近批次损失，帮助区分局部批次波动与整体动作误差退化。评估必须恢复训练模式，并避免改变共享训练噪声流。

最终第 9,875 次更新使用三个固定采样种子 1101/1102/1103评估全部训练和开发窗口、分演示误差、静止基线及夹爪指标。中途不按开发结果选权重或调整设置。三个采样种子仍不等于三个独立训练初始化。

只保存一个最终模型和小型 JSON/CSV，不保存 Adam 状态或多份大权重。不会通过保存多个中途大模型补救既往未保存的结果。

## 判定与停止条件

1. 如果最终训练位置不超过 0.5 cm、旋转不超过 5°，且不存在非有限参数/梯度，认为本轮训练拟合达到既定诊断门槛；仍需检查逐窗口极端误差与后期监测结果，不宣称所有动作精确拟合。
2. 如果仍达不到训练门槛，停止这条作者分支诊断，不继续扫描学习率、增加轮数或混数据。作者对照仍不合格，不能据此排除动作头。
3. 如果训练达到门槛而开发仍差于静止基线，则增加一条“该作者结构在此冻结上下文协议下也未泛化”的证据，停止继续更换降噪器。不能升级为“所有动作头都没问题”或“CLIP 是唯一根因”。
4. 开发位姿通过要求三个采样种子下总体及三条演示的位置、旋转均严格优于各自静止基线。夹爪平衡准确率与两方向切换单独报告；冻结夹爪的失败不能因位姿改善而隐藏。
5. 这是一次补充运行；即使较低学习率改善，也只支持该设置下的优化解释，不构成多训练种子统计结论。

## 资源限制与执行前提

沿用现有集群环境与一张 A6000。现有预检证明原结构能运行，但加入训练监测后的耗时需重新实测，再决定 Slurm 时限。使用同一个原始缓存，不上传原始数据、不下载 CLIP 或作者权重、不安装一套新环境。

新增集群输出目标不超过 2 GiB；提交前检查真实存储余量，不将 200 GB 配额当作剩余空间。本地只接收不超过 100 MiB 的报告，模型留在集群。

原 `run_decoder_pair.py` 已增加作者单分支、学习率参数和训练监测，没有另建训练系统。原始 187797 输出与旧上传包保持不变。完成集群输入一致性与资源预检后，才可提交这一次正式更新。

本地旧、新入口在相同环境下的作者初始化和完整输入流一致，评估恢复 Python/NumPy/Torch 随机数及各模块模式的检查通过。但本地哈希与集群 187797 不同，不能宣称已核对跨环境一致性；具体差异来源未确定。集群入口会在任何更新前严格核对原作业的初始化、冻结参数及完整输入流；失败就停止。本地没有优化器更新。记录见 [本地检查](../reports/decoder_author_lr_local_checks.json)。

## 当前操作：只提交 setup 与预检

本地 PowerShell：

```powershell
Set-Location "D:\ntu_related\dissertation\GPU cluster"
scp decoder_author_lr_v1.zip decoder_author_lr_v1.sha256 decoder_author_lr_setup.sh decoder_author_lr_preflight.sh decoder_author_lr_train.sh zixiao005@10.97.216.128:/projects/Zeroshot/
```

VS Code 集群 SSH 终端：

```bash
cd /projects/Zeroshot
mkdir -p logs
AUTHOR_SETUP=$(sbatch --parsable decoder_author_lr_setup.sh)
AUTHOR_CHECK=$(sbatch --parsable --dependency="afterok:$AUTHOR_SETUP" decoder_author_lr_preflight.sh)
echo "环境核验：$AUTHOR_SETUP；GPU预检：$AUTHOR_CHECK"
squeue --me
```

同一终端查看：

```bash
sacct -j "$AUTHOR_SETUP,$AUTHOR_CHECK" --format=JobID,State,ExitCode
tail -n 80 "/projects/Zeroshot/logs/author-lr-setup-$AUTHOR_SETUP.out"
tail -n 80 "/projects/Zeroshot/logs/author-lr-preflight-$AUTHOR_CHECK.out"
```

setup 不安装包，沿用现有环境；会输出项目目录 du 占用及文件系统余量，后者不是项目配额余量。GPU 预检包含四次可丢弃更新及完整训练集监测，用于实测新增监测的资源消耗。上传 ZIP 约 7.62 MiB，不含原始图像或 CLIP 权重。先核对 setup 占用、预检哈希与 passed，再提交正式训练；不要复用 187756 的旧预检。

正式训练的后续模板（当前先不执行）：

```bash
export AUTHOR_LR_PREFLIGHT="/projects/Zeroshot/runs/author-lr-preflight-$AUTHOR_CHECK/preflight.json"
AUTHOR_TRAIN=$(sbatch --parsable --export=ALL,AUTHOR_LR_PREFLIGHT="$AUTHOR_LR_PREFLIGHT" decoder_author_lr_train.sh)
echo "训练作业：$AUTHOR_TRAIN"
```

## 与老师要求的关系

老师要求冻结 CLIP、8 层/512 通道 Adapter 与扩散动作解码器联合训练，以及多任务、语言变化、独立训练种子和闭环任务成功率。当前诊断使用旧 2 层/256 通道 Adapter 的冻结上下文、一个任务指令，不能代表这些要求已完成。后续正式协议必须另行固定任务、动作与控制接口、语言划分、数据来源和基线；保留当前 5 条最终测试演示不参与诊断优化。

本轮最多补齐一次失败的结构对照。已有当前位姿、近邻和 patch 汇聚检查不重复开展。

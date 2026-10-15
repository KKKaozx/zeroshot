# 已拟合状态后的学习率对照

已完成集群预检200149和正式对照200150。降低组最终0.937mm/0.344°、32/32训练窗口通过，原学习率组重现21.856mm/5.242°、0/32，开发指标未改善。[完整独立核验结果](BRIDGE_LATE_LR_RESULT_200150_20261012.md)。下文保留执行前协议和复用命令，已完成作业不需重提。

依据追踪200039：seed43在1024次达到2.68mm/1.43°、31/32窗口达标，在1536次仍通过，最后2048次退化到21.86mm/5.24°、0/32。先检验后期更新幅度是否影响拟合保持，不更换样本或模型。

## 固定协议

- 仅训练初始化43，沿用原32训练窗口、16演示、5任务和40开发窗口，CLIP及夹爪冻结。
- 原始八层Adapter、多尺度x0 U-Net、重复当前观测路径、Adam、batch8、wd0、梯度裁剪1、DDPM100及三采样种子不变；9份原冻结源码逐项复制，不修改生产模块。
- 前1024次以lr=1e-4共同训练；核对初始权重、首步loss、同集群输入身份及1024次生成指标，必须通过原拟合门槛后才分叉。
- 保留共同1024次参数和Adam动量/计数；恒定组直接继续，降低组恢复同一参数及完整Adam状态后，只将lr改为1e-5。后1024次使用同一原批次顺序及相同全局update随机键，每组每窗口256次抽样。
- 两组均完成2048次，不早停，不重置优化器。训练生成检查点为共同0/512/1024和各组1536/1792/1920/2048；开发集仅最终评估，不用开发误差选择分叉点。
- 固定分叉点来自已完成的训练诊断，用于检验拟合稳定性，不作为未接触测试或最佳泛化检查点。

原学习率组必须重现旧的最终失败，才能将两组差异解释为干预效应。报告会记录原最终路径/终点指标差，预先约定位置容差0.1cm、旋转0.5°，且原学习率组完整拟合门槛仍为失败；若未重现则先报告控制失败，不能宣布降低学习率解决问题。

## 预检和资源

本地与集群分别预检：先共同更新4次，原学习率连续4次；恢复共同参数和Adam后再用同学习率4次，检查最终参数哈希、全部loss和生成预测完全一致。随后从共同状态用低学习率4次，首步loss应相同、最终参数必须不同。预检不要求8次更新达到拟合门槛，不保存权重，不评估开发泛化。

本地预检报告：`reports/bridge_late_lr_local_preflight_20261012/report.json`。上传打包步骤要求预检通过；正式集群脚本又要求同包集群预检成功及输入匹配。

本地实际预检已通过：耗时219.515秒，峰值保留显存3.852GiB；恢复组与连续组最终参数哈希相同、损失与预测差异均为0，两组全部逐步日志相同。降低学习率组首步loss和梯度相同、更新后参数不同。5组预测已用CPU float64独立复算，完整72窗口原始身份与数值检查通过，CLIP/其他冻结参数未变，无PT文件。本地验证结果为同目录`verification.json`；不能据8次更新宣布拟合或泛化改善。

10份冻结源文件中，9份来自原追踪包且哈希未变，只新增对照入口。Python语法与两份Bash脚本解析通过；ZIP全部内容与内部校验清单逐项核验。上传ZIP为68,862字节，集群实验尚未执行。

默认使用已验证环境兼容的A6000，CPU4、内存22528MiB，预检20分钟、正式90分钟时限；时限不是预计耗时。复用现有环境、数据和CLIP，无pip/conda安装或数据/模型下载。

正式集群保留共同1024次含Adam的可训练参数检查点和两个2048次最终可训练参数检查点，PT总上限4GiB。开跑前按200GiB项目规划限额，预留本次4GiB和15GiB余量；du占用不是实时配额。所有大权重留在集群，本地预检与下载不存PT。

## 上传与提交

本地PowerShell：

```powershell
scp "D:/ntu_related/dissertation/Zero_shot/cluster_packages/bridge_late_lr_pair_v1.zip" "D:/ntu_related/dissertation/Zero_shot/cluster_packages/bridge_late_lr_pair_v1.sha256" zixiao005@10.97.216.128:/projects/Zeroshot/
```

集群Bash（仅提交一次）：

```bash
cd /projects/Zeroshot
mkdir -p logs
sha256sum -c bridge_late_lr_pair_v1.sha256 && unzip -n bridge_late_lr_pair_v1.zip
LATE_CHECK=$(sbatch --parsable /projects/Zeroshot/bridge_late_lr_pair_v1/preflight.sh)
LATE_FIT=$(sbatch --parsable --dependency="afterok:$LATE_CHECK" --export="ALL,LATE_LR_PREFLIGHT_JOB=$LATE_CHECK" /projects/Zeroshot/bridge_late_lr_pair_v1/fit.sh)
printf '预检：%s；正式对照：%s\n' "$LATE_CHECK" "$LATE_FIT"
squeue --start -j "$LATE_CHECK,$LATE_FIT"
```

查看：

```bash
sacct -j "$LATE_CHECK,$LATE_FIT" --format=JobID,State,ExitCode,Elapsed,NodeList
tail -n 12 "/projects/Zeroshot/logs/bridge-late-lr-preflight-$LATE_CHECK.out"
tail -n 12 "/projects/Zeroshot/logs/bridge-late-lr-fit-$LATE_FIT.out"
```

正式结果为`/projects/Zeroshot/runs/bridge-late-lr-fit-实际作业编号/report.json`。只下载小报告/预测/日志，勿递归下载含PT的目录：

```powershell
$LateJobId = '实际正式作业编号'
$LateDestination = "D:/ntu_related/dissertation/Zero_shot/reports/bridge-late-lr-fit-$LateJobId"
New-Item -ItemType Directory -Force -Path $LateDestination | Out-Null
$LateRemote = "zixiao005@10.97.216.128:/projects/Zeroshot/runs/bridge-late-lr-fit-$LateJobId"
scp "$LateRemote/*.json" "$LateRemote/*.jsonl" "$LateRemote/*.npz" "$LateRemote/runner_snapshot.py" "$LateDestination/"
```

## 判断边界

若原学习率组重现退化而低学习率组保持拟合，支持后期更新幅度影响稳定性；需要其他初始化复核，不能说唯一坏层已找到。若两组均退化，则这项干预不足，下一步才考虑冻结Adapter的模块隔离。若低学习率保住训练拟合但开发指标仍差，则训练稳定性与泛化是两个仍需分开处理的问题。此实验不训练夹爪、不运行仿真，也不衡量机器人任务成功率。

# 后期学习率修正的两个初始化复核

状态更新：200208的42/44完整38份小文件已下载独立核验，降低组四个观测点均32/32达标，最终约1.08mm/0.40°、0.93mm/0.38°；原控制预测逐元素复现，留出指标仍未超过静止基线。见[结果与证据级别](BRIDGE_LATE_LR_REPLICATION_RESULT_200208_20261012.md)。下文本地准备及提交描述保留为实际执行协议；无需重复提交已完成作业。

seed43的单变量对照200150已核验：共同参数和Adam状态相同，原学习率组逐元素重现退化，降学习率组四个观测点均32/32通过，最终0.937mm/0.344°，但开发误差未改善。下一步只在初始化42和44重复这一对照，不换数据，不增加轮数，不修改生产模型。

## 固定设计及复现判据

每个初始化各自从0训练到1024，再从共同参数及完整Adam状态分叉，分别以1e-4和1e-5继续到2048。保持原32窗口、16训练演示、5任务、40开发窗口，冻结CLIP/夹爪、batch8、Adam/零wd、裁剪1、dropout与动作噪声随机键、x0目标及完整DDPM100。原分叉后的抽样为每窗256份，每组1024更新。

初始化42原控制最终4.07mm/2.06°、25/32，初始化44原控制最终3.03mm/1.62°、31/32。控制组与原200039结果比较路径和终点指标，位置容差0.1cm、旋转0.5°，并要求完整门槛判定与原运行相同；不能要求原本通过的seed44控制组失败。

共同1024次必须通过原拟合门槛且与原记录匹配后才继续。两组使用相同观测点1536/1792/1920/2048，仅最终评估开发集，不使用开发集挑选检查点。通过训练门槛只证明训练位姿拟合，不代表夹爪、抓取、闭环执行或泛化。

复核入口从已执行200150的冻结入口生成，仅增加初始化参数、同种子预检核对，并将控制门槛比较改为原运行判定。旧10份源文件哈希保持不变；实际训练、状态恢复及学习率干预代码不变。旧包完整保留。

## 预检及存储

两个初始化分别进行短预检：共同4次→连续4次；恢复同一参数和Adam→同学习率4次，要求最终参数、全部loss和预测完全相同；低学习率4次的首步loss相同、更新后参数不同。8次预检未拟合是正常情况，不按其误差评价学习率优劣。

复用现有集群环境和资源，默认A6000，预检25分钟、正式2小时时限。每个初始化保存3个可训练参数检查点（共同含Adam、两个最终），每个上限4GiB，整项集群预算8GiB；开跑前需规划限额200GiB下保留额外15GiB余量。实际seed43约2.46GiB，两个复核预期约4.93GiB，但按8GiB保守检查。这个空间检查是du规划值，不是实时配额查询。

本地预检不保存PT，上传包不包含数据或模型。只在集群运行完整复核，下载时仅取小报告、NPZ、JSONL。

本地两个初始化的短预检及独立CPU核验已通过。连续与恢复同学习率训练的逐步loss、全部最终预测逐元素相同，最终参数哈希相同；分叉首步loss和梯度相同。seed42耗时220.55秒，seed44耗时222.69秒，峰值保留显存均3.852GiB。两组各5份NPZ已独立复算；位置指标最大差异小于0.000001cm，旋转小于0.00006°。这是8次更新的执行验证，尚未运行完整42/44训练，不能作为拟合改善证据。Windows Torch2.11与集群Torch2.5不同，正式复現比较仍在集群同环境完成。

冻结上传ZIP为75,245字节，约73.5KiB，内部逐文件校验和及ZIP校验均通过。原10份源文件保持相同哈希。[seed42独立核验](../reports/bridge_lr_replication_local_preflight_20261012/seed42/verification.json)、[seed44独立核验](../reports/bridge_lr_replication_local_preflight_20261012/seed44/verification.json)。

## 上传与提交

本地PowerShell：

```powershell
scp "D:/ntu_related/dissertation/Zero_shot/cluster_packages/bridge_late_lr_replication_v1.zip" "D:/ntu_related/dissertation/Zero_shot/cluster_packages/bridge_late_lr_replication_v1.sha256" zixiao005@10.97.216.128:/projects/Zeroshot/
```

集群Bash，仅提交一次。一个作业依次运行两个初始化，不需要提交四个训练作业：

```bash
cd /projects/Zeroshot
mkdir -p logs
sha256sum -c bridge_late_lr_replication_v1.sha256 && unzip -n bridge_late_lr_replication_v1.zip
LR_REP_CHECK=$(sbatch --parsable /projects/Zeroshot/bridge_late_lr_replication_v1/preflight.sh)
LR_REP_FIT=$(sbatch --parsable --dependency="afterok:$LR_REP_CHECK" --export="ALL,LR_REP_PREFLIGHT_JOB=$LR_REP_CHECK" /projects/Zeroshot/bridge_late_lr_replication_v1/fit.sh)
printf '复核预检：%s；正式复核：%s\n' "$LR_REP_CHECK" "$LR_REP_FIT"
squeue --start -j "$LR_REP_CHECK,$LR_REP_FIT"
```

状态和末尾日志：

```bash
sacct -j "$LR_REP_CHECK,$LR_REP_FIT" --format=JobID,State,ExitCode,Elapsed,NodeList
tail -n 15 "/projects/Zeroshot/logs/bridge-lr-rep-preflight-$LR_REP_CHECK.out"
tail -n 15 "/projects/Zeroshot/logs/bridge-lr-rep-fit-$LR_REP_FIT.out"
```

正式报告分别在`/projects/Zeroshot/runs/bridge-lr-rep-fit-实际编号/seed42/report.json`和`seed44/report.json`。两个初始化都结束后，整体日志出现`LR REPLICATION FIT EXECUTION COMPLETE`。

本地只下载小文件：

```powershell
$LrRepJobId = '实际正式作业编号'
foreach ($LrRepSeed in @(42,44)) {
    $LrRepDestination = "D:/ntu_related/dissertation/Zero_shot/reports/bridge-lr-rep-fit-$LrRepJobId/seed$LrRepSeed"
    New-Item -ItemType Directory -Force -Path $LrRepDestination | Out-Null
    $LrRepRemote = "zixiao005@10.97.216.128:/projects/Zeroshot/runs/bridge-lr-rep-fit-$LrRepJobId/seed$LrRepSeed"
    scp "$LrRepRemote/*.json" "$LrRepRemote/*.jsonl" "$LrRepRemote/*.npz" "$LrRepRemote/runner_snapshot.py" "$LrRepDestination/"
}
```

## 判定

分别检查控制组复现、降低组在各观测点的训练门槛、最后的开发指标。若42/43/44降低组均通过且改善保持，才能将其作为这套固定32窗口配方的稳定性修正；仍不能直接推广到全量训练或泛化。若某种子控制组未复现，先报告该限制；若降低组退化或精度变差，保留原始结果，不改变种子/样本来筛选成功运行。

# 64窗口拟合复核：固定初始化42

当前结论：201429全部13份小文件已下载并独立核验，原32和新增32均32/32达标，后期四次64/64，复核与200600只改变训练初始化。训练拟合可重复，留出仍未超过静止基线。见[最终结果与范围](BRIDGE_GROWTH64_REPLICATION_RESULT_201429_20261013.md)。下方为设计和执行历史，请勿重复提交。

目的：检验200600在初始化43下的64窗口拟合能否在另一个预定初始化下重复。使用42，成功或失败均报告，避免根据开发结果挑种子。本地短预检及独立核验通过，上传包已冻结。集群已提交预检201428、正式复核201429；下方提交命令为历史，请勿重复提交。

201429最新日志：4096更新结束，最终训练平均1.4992mm/0.6434°、fit gate=True；3072/3584/3840训练平均位置约1.521/1.539/1.544mm。开发40窗口9.6913cm/18.7132°，仍差于静止8.2123cm/15.0984°。上一轮初始化43训练1.2746mm/0.6009°、开发9.7494cm/18.8939°。两次日志初步支持64窗口训练拟合可重复，泛化仍未通过。集群状态、报告执行标志、原32/新增32分组、逐窗数量及数组独立复算待下载核验，不提前宣称64/64达标。

冻结包20文件，16,823,998字节（约16.04MiB），SHA256 `d73ef24aad30c8e0abb8560ed379aa60f3ccba48b37950607b48dd200ddd993a`。ZIP逐文件读取与本地对应文件完全一致、CRC通过，包含预处理参考但没有原始TFRecord或模型权重。本地预检230.91秒、峰值保留显存3.871GiB；104输入、5份预测、32条短跑更新日志已独立检查，恢复参数/损失/预测全部精确一致。总体及两子组位姿误差和短跑拟合门槛也已CPU复算一致；短跑门槛False只表示尚未拟合。见[本地独立核验](../reports/bridge_growth64_replication_local_preflight_20261013/verification.json)。

## 固定变量与判据

生产代码、64训练窗口/32演示、40开发窗口/10演示、五任务比例、图像和动作处理、CLIP冻结、Adapter/U-Net结构、夹爪冻结、batch8、Adam状态衔接、4096更新、2048更新后1e-4降至1e-5、每窗512次抽样、DDPM100和三个采样种子保持不变。生成入口训练逻辑只将初始化43改为42；反向替换后源码文本与200600入口完全一致。训练/开发/输入参考及模型配置逐项比对。

总体与原32/新增32子组的门槛沿用原配方：每输出整体路径≤5mm/5°、终点≤10mm/10°，至少90%窗口在三个采样输出中路径平均均≤5mm/5°。四次后期观测3072/3584/3840/4096全部记录，开发只在最终评估，不用于选权重。

若四次总体及两子组门槛均通过，则在两个初始化下具备更稳固的64窗口拟合证据；如果仅最终通过，记录中间波动；若失败，先检查同一数据的优化轨迹。执行检查通过不等于拟合通过。无论结果如何都不能直接宣称泛化改善或抓取成功。

## 预检和空间

短预检读取全部104输入，以8次共同更新和各8次连续/恢复同学习率/低学习率更新检查执行；观察16个原窗口加16个新增窗口。预检不保存权重，不要求短跑达到拟合门槛。独立CPU核验每步日志、恢复预测数组及误差，并用真实训练入口argparse检查两个shell的所有参数，避免仅语法检查遗漏无效参数。

复用原有环境、原始Bridge和CLIP，不安装包或下载模型。新包只复制小型预处理参考及代码；正式权重约0.5GiB留在集群。本地新增约34MiB，加少量报告，低于100MiB规划限制。默认A6000，预检25分钟、正式90分钟均为作业时限。配额仍以200GiB作规划，开跑前按du预留15GiB和1GiB输出。

## 上传与提交

本地PowerShell：

```powershell
scp "D:/ntu_related/dissertation/Zero_shot/cluster_packages/bridge_growth64_replication_v1.zip" "D:/ntu_related/dissertation/Zero_shot/cluster_packages/bridge_growth64_replication_v1.sha256" zixiao005@10.97.216.128:/projects/Zeroshot/
```

集群终端校验与解压，成功后提交一次预检和一份依赖其成功的正式拟合：

```bash
cd /projects/Zeroshot
sha256sum -c bridge_growth64_replication_v1.sha256 && unzip -n bridge_growth64_replication_v1.zip
```

```bash
REP64_CHECK=$(sbatch --parsable /projects/Zeroshot/bridge_growth64_replication_v1/preflight.sh)
REP64_FIT=$(sbatch --parsable --kill-on-invalid-dep=yes --dependency="afterok:$REP64_CHECK" --export="ALL,GROWTH_PREFLIGHT_JOB=$REP64_CHECK" /projects/Zeroshot/bridge_growth64_replication_v1/fit.sh)
printf '64窗口复核预检：%s；正式复核：%s\n' "$REP64_CHECK" "$REP64_FIT"
squeue --start -j "$REP64_CHECK,$REP64_FIT"
```

正式作业依赖本次新预检，不依赖已结束较久的200365。若预检失败，正式依赖失效时取消，避免一直等待；不重新提交第二份正式训练。

```bash
sacct -j "$REP64_CHECK,$REP64_FIT" --format=JobID,State,ExitCode,Elapsed,NodeList
tail -n 15 "/projects/Zeroshot/logs/bridge-growth64-rep-preflight-$REP64_CHECK.out"
tail -n 20 "/projects/Zeroshot/logs/bridge-growth64-rep-fit-$REP64_FIT.out"
```

完成后只下载报告、JSONL、NPZ及入口快照，不下载PT：

```powershell
$RepJobId = '实际正式复核编号'
$Dest = "D:/ntu_related/dissertation/Zero_shot/reports/bridge-growth64-rep-fit-$RepJobId"
New-Item -ItemType Directory -Force -Path $Dest | Out-Null
$Remote = "zixiao005@10.97.216.128:/projects/Zeroshot/runs/bridge-growth64-rep-fit-$RepJobId"
scp "$Remote/*.json" "$Remote/*.jsonl" "$Remote/*.npz" "$Remote/runner_snapshot.py" "$Dest/"
```

- [上一轮独立核验](BRIDGE_GROWTH64_RESULT_200600_20261013.md)

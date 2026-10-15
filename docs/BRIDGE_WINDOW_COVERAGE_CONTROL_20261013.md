# 同规模取窗覆盖对照

完整核验更新：202069的13份正式小文件与202068预检报告已齐，8192日志/八份预测独立复算通过。89保留与39替换训练组均拟合通过，开发整体退步，具体替换配方不采用，本项收口。见[最终结果与限制](BRIDGE_WINDOW_COVERAGE_RESULT_202069_20261013.md)。下文“待下载/未复算”均为历史状态。

本项检验201663只读诊断发现的输入时段覆盖差异。仍使用相同64条训练演示、五个已见任务和128个窗口；仅替换39个输入窗口，89个身份保持不变。用户已提交集群预检202068和正式对照202069，正式作业依赖预检成功。尚未收到作业状态或结果，不能预先宣称改善。下文“尚未提交”为本地准备时的历史状态，请勿重复提交。

## 固定什么、改变什么

202069终端结果更新：正式日志已输出8192次更新及报告，最终训练2.103mm/0.762°、总体门槛True；6144/7168/7680位置1.971/2.156/1.895mm。开发9.420cm/17.267°，比201663的8.728cm/15.847°及静止8.212cm/15.098°均差。当前不采用39窗口替换作为修复；该结果不支持“增加内部窗口即可改善”这一具体方案，也不证明取窗完全无影响或定位唯一根因。尚未收到Slurm最终状态和下载的小文件，89保留/39替换逐组门槛、逐演示/时段及指标须独立复算。[终端证据范围](../reports/bridge_window_202069_log_summary.json)。

| 项目 | 201663控制 | 本次处理 |
| --- | --- | --- |
| 训练演示/窗口 | 64/128 | 64/128，同演示 |
| 索引最早/内部/最晚窗口 | 64/12/52 | 38/51/39 |
| 任务窗口数 | 32/24/24/24/24 | 完全一致 |
| 初始化/更新次数/batch | 43/8192/8 | 完全一致 |
| 每窗抽样次数 | 512 | 512，同索引抽样与噪声种子 |
| Adam与学习率 | 前4096次1e-4，后4096次1e-5 | 完全一致，不重置Adam |
| 模型/图像处理/动作尺度 | 冻结CLIP、Adapter、多尺度x0 U-Net | 完全一致 |
| 生成与评价 | 原生DDPM100，三个固定采样种子 | 完全一致 |
| 开发集 | 40窗口/10演示，最终评价 | 身份与标签不变 |
| 夹爪 | 冻结 | 冻结 |

“内部”指索引池内输入起点在最早与最晚之间，不能解释为语义上的抓取/释放阶段。39条演示原来的两个输入均在端点，各替换一个端点为最接近输入索引中点的候选；等距时选较早，另一端点按任务/演示身份排序平衡保留。已有内部窗口的12条演示及只有两个候选的13条演示保留原样。不按预测误差或动作难度选样。

这不是新任务零样本测试。40开发窗口已反复使用，本次是探索性控制；即使改善，也不能凭单初始化确定唯一根因。改变窗口同时改变对应画面和未来动作内容，但不会加入新演示或增加训练规模。

## 核验和资源

新包复用冻结`bridge_growth64_v1`与`bridge_growth128_v1`的输入参考；核验旧128输入后，仅替换39个位置。89个原训练窗口和40个开发窗口须与201663原始身份一致，正式集群运行还要求这129份处理后张量哈希精确一致。39个新窗口带独立原始身份及图像/动作/当前夹爪参考，运行时逐个检查；跨Windows/Linux沿用原1e-6数值容差。预检与正式使用相同清单/源码/168份输入检查。

十份生产源文件字节不变，核心上下文、优化循环、Adam状态拷贝/哈希与checkpoint函数AST不变。预检先16次共同更新，然后连续、恢复同学习率、降低学习率三分支各16次；恢复同学习率的损失、参数和预测须精确重放。本地仅短预检，无完整拟合、无PT文件。独立CPU检查选择身份、39参考、168原始身份、更新日志、五份预测、指标和真实argparse启动参数；Bash语法另检。

新增39参考约19.18MiB，不上传原始数据、CLIP或环境。集群复用既有环境，无安装步骤。正式只保存约0.5GiB最终可训练权重到集群，本地只下载小报告和预测；启动前按200GiB规划上限预留15GiB及1GiB输出，du估计并非实时配额查询。默认A6000，预检30分钟、正式2小时为申请时限，不是预计耗时。

实际本地短预检267.73秒，峰值保留显存3.936GiB；168输入、64条短更新日志及五份预测独立CPU核验通过，同学习率恢复的日志/保存数组精确一致。新旧本地128预检初始化权重哈希一致，CLIP及其他冻结模块未变。检查脚本曾因Windows路径字符串分隔符断言失败，改为统一路径表示后通过；没有修改训练代码或数值容差。短跑拟合门槛False符合执行预检目的，不提供正式拟合或改善结论。

冻结ZIP共21份文件、20,171,931字节（19.24MiB），CRC及逐文件字节校验通过，SHA256为`b03e257b12f8c746ba520573e349031b4d1a5ec41dd31ba163adb612e3f4e172`。本地仅保存约1.47MiB短预检输出，无PT；上传前本地D盘剩余约67.6GiB。集群正式作业尚未提交。

## 判定顺序与停止条件

先核验全部128训练窗口的原生生成拟合；沿用整体每采样输出路径≤5mm/5°、终点≤10mm/10°，至少116/128窗口在三个输出中路径均达标。再从保存预测独立拆分89保留窗口与39替换窗口，至少81/89及36/39逐窗达标，检查是否新增样本导致训练拟合失效。原继承报告中的`retained_train`/`added_train`仅代表前64/后64行，不代表本次89/39组，不能混用。

通过拟合后比较最终固定40窗口的路径位置和旋转，并报告第16目标误差、任务、演示及索引时段。控制201663为8.728cm/15.847°，静止为8.212cm/15.098°。不能只凭loss、某个任务或某种汇总更好宣布通过，不用开发集挑checkpoint、训练种子或追加轮数。新窗口若未拟合，先解释训练失败；若拟合但开发未改善，停止取窗覆盖这一解释，不继续扩256。即使开发改善，仍需要独立复核，并不能证明夹爪或机器人抓取成功。

## 上传与提交

本地PowerShell上传两个文件：

```powershell
scp "D:/ntu_related/dissertation/Zero_shot/cluster_packages/bridge_window_coverage_v1.zip" "D:/ntu_related/dissertation/Zero_shot/cluster_packages/bridge_window_coverage_v1.sha256" zixiao005@10.97.216.128:/projects/Zeroshot/
```

集群终端执行，正式依赖预检成功；勿重复提交：

```bash
cd /projects/Zeroshot
sha256sum -c bridge_window_coverage_v1.sha256 &&
unzip -n bridge_window_coverage_v1.zip &&
WINDOW_CHECK=$(sbatch --parsable /projects/Zeroshot/bridge_window_coverage_v1/preflight.sh) &&
WINDOW_FIT=$(sbatch --parsable --kill-on-invalid-dep=yes --dependency="afterok:$WINDOW_CHECK" --export="ALL,WINDOW_PREFLIGHT_JOB=$WINDOW_CHECK" /projects/Zeroshot/bridge_window_coverage_v1/fit.sh) &&
printf '取窗预检：%s；正式对照：%s\n' "$WINDOW_CHECK" "$WINDOW_FIT" &&
squeue --start -j "$WINDOW_CHECK,$WINDOW_FIT"
```

查看状态和末尾日志（断连后把变量换为记录的作业号）：

```bash
sacct -j "$WINDOW_CHECK,$WINDOW_FIT" --format=JobID,State,ExitCode,Elapsed,NodeList
tail -n 15 "/projects/Zeroshot/logs/bridge-window-preflight-$WINDOW_CHECK.out"
tail -n 20 "/projects/Zeroshot/logs/bridge-window-fit-$WINDOW_FIT.out"
```

完成后本地下载小文件，将变量填写为正式作业号：

```powershell
$WindowJobId = '填写正式作业号'
$Dest = "D:/ntu_related/dissertation/Zero_shot/reports/bridge-window-fit-$WindowJobId"
New-Item -ItemType Directory -Force -Path $Dest | Out-Null
$Remote = "zixiao005@10.97.216.128:/projects/Zeroshot/runs/bridge-window-fit-$WindowJobId"
scp "$Remote/*.json" "$Remote/*.jsonl" "$Remote/*.npz" "$Remote/runner_snapshot.py" "$Dest/"
```

- [201663诊断与限制](BRIDGE_GROWTH128_GENERALIZATION_201663_20261013.md)
- [冻结选择身份计划](../cluster_packages/bridge_window_coverage_v1/selection_plan.json)
- [本地短预检独立核验](../reports/bridge_window_coverage_local_preflight_20261013/verification.json)

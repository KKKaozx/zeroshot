# 训练覆盖泛化：一次独立初始化复核

**207558/207521独立工件验收完成，17文件/7预测齐全。** float64原40开发6.808208cm/15.138891°，第16目标8.221684cm/19.032170°；9/10演示位置优于静止、8/10优于同seed小池，五任务位置都改善。旋转只有2/10演示优于静止，整体稍超原旋转门槛，完整原候选未复现通过。两个初始化均支持位置收益方向，旋转任务差异仍在。源码/输入/初始化/缓存证明/8192更新和恢复已核验，无新NN执行、PT或完整缓存下载。按原计划收口，不追加种子或预算。见[独立结果与两初始化图表](BRIDGE_COVERAGE_REPLICATION_RESULT_207558_20261016.md)。下文待下载/待核验和操作命令为历史，勿重复执行。

**207558正式复核完成，终端证据：位置收益方向重复出现，完整原门槛未复现。** COMPLETED/0:0、1小时5分1秒、A6000，8192更新。原40开发6.808208cm/15.138787°，相对同seed42小池202463/real的8.381655cm/15.993918°两项改善；位置比静止8.212326cm低17.10%，旋转比静止15.098500°高0.040287°，基本接近但不能称更优。原旋转门槛15.118500°被超过0.020287°，不改门槛或追加种子，因此按终端摘要完整候选未通过。训练全1836窗5.109892cm/12.651512°不作为延长预算的理由。

目前只有日志，逐演示/任务、第16目标与完整源/输入/实际保存预测仍待独立验收。不认为独立训练两次已充分证明显著性或泛化全部修复；两个初始化都提供固定8192预算下扩大覆盖降低开发位置误差的方向证据，旋转未稳定低于静止。下一步仅下载小工件核验，不再训练、编码或延长预算。下文“正在运行/未提交”与提交命令为历史，勿重复执行。

集群打包（不含权重或完整特征缓存）：

```bash
tar --exclude='*.pt' --exclude='*.partial' -czf /projects/Zeroshot/bridge_coverage_replication_results_207558.tar.gz -C /projects/Zeroshot/runs bridge-coverage-rep-fit-207558 bridge-coverage-rep-preflight-207521
```

本地PowerShell：

```powershell
scp zixiao005@10.97.216.128:/projects/Zeroshot/bridge_coverage_replication_results_207558.tar.gz "D:/ntu_related/dissertation/Zero_shot/reports/"
```

预期17小文件/7预测快照须按实际冻结运行代码核对；不下载529MB权重或1.88GiB完整缓存。终端记录：[summary](../reports/bridge_coverage_replication_207558_terminal_summary.json)。

**当前唯一正式复核已提交：207558；207521完整验证器已通过。** 用户贴出完整预检验证成功，峰值5.632813GiB、暖更新0.174970秒、估计正式1.498208小时、完整缓存1.876761GiB。该估算不含排队，含训练/评价的资源规划值，不是完成时限。完整报告尚未本地下载独立验收；正式Slurm状态与结果未收到，不能认定留出改善。只跟踪207558，8192更新后还要完成1836训练窗和40开发窗的三采样评价。旧207348/202463及预检不重跑，无新编码或PT下载。下文尚未提交/验证后提交命令为历史，勿重复执行。

```bash
sacct -j 207558 --format=JobID,State,ExitCode,Elapsed,NodeList
tail -n 15 /projects/Zeroshot/logs/bridge-coverage-rep-fit-207558.out
```

**当前完整预检207521已完成（用户终端证据）：COMPLETED/0:0、5分22秒、A6000。** 先前20752为漏位编号，实际以207521为准，不重提预检。常学习率与恢复同学习率首损失0.19130820035934448及训练探针11.689527cm/45.192104°一致。短预检误差不是泛化成绩，不追加拟合。完整报告尚未下载本地；下一步运行原包验证器，核对完整1876输入/415演示、seed42起点、既有缓存与资源，通过才提交唯一正式组，使用COVERAGE_REP_PREFLIGHT_JOB=207521。当前无正式编号或新泛化结果。下文未提交预检和首次提交命令为历史，不重复执行。

包已准备，本地局部预检及独立核验通过；尚未提交集群作业，没有新的泛化成绩。主目标是复核207348的留出收益，不追求训练池毫米拟合，不追加更新。

## 固定对照与复用

| 初始化 | 小训练池128窗/64演示 | 大训练池1836窗/405演示 |
|---|---|---|
| 43 | 已验202278，复用 | 已验207348，复用 |
| 42 | 已验202463的real组，复用 | 本次唯一新正式组，尚未提交 |

四组均8192次更新、每批8窗、前4096步Adam1e-4/后4096步1e-5、权重衰减0、梯度裁剪1、冻结CLIP和夹爪、相同Adapter/多尺度x0-DDPM100/真实当前状态/原128训练统计。新大池逐批索引、扩散噪声种子和Dropout种子与207348完全相同；唯一训练变化是参数初始化43→42。固定原40窗/10条开发演示和三个采样种子1101/1102/1103，不能把采样种子当独立训练复核。

已有202463/real全部工件指纹与先前独立审计匹配，其运行代码相对原状态代码只有初始化43→42两数字差异。数据/状态/源码、8192预算、学习率、任务槽和噪声/Dropout种子一致。对应小池开发8.381655cm/15.993918°，不能把新seed42大池直接与seed43小池相减当配对效果。见包内reuse_control_proof.json。

沿用原候选门槛：原40开发路径位置≤7.391093cm、旋转≤15.118500°，两项优于同seed42小池，至少6/10条演示位置优于同seed小池。额外完整报告静止、各任务/演示和第16目标，失败不更换种子、门槛或加训练。即便复核成功，也只支持当前固定预算下覆盖收益方向；不能拆开更多演示、更多动作阶段与每窗重复次数的贡献。

## 缓存与运行完整性

已有完整缓存/projects/Zeroshot/bridge_training_coverage_v1/full_tokens.pt约1.88GiB必须存在，完整预检和正式均要求复用，缺失则停止，不重新编码。缓存中原protocol指纹代表原始输入和冻结CLIP编码来源，仍核对其完整key、视觉/文本指纹以及原168特征精确一致；新初始化不影响冻结编码器。新helper仅显式使用原protocol作为缓存来源指纹并要求已有缓存，原输入和标签解析不变。

新运行主体与207348仅两处初始化表达式不同，原生产模型/解析器不修改。完整预检仍核对415演示payload和1876输入、seed42参数起点、现有缓存及资源，16次入口更新，不评开发。正式验证完整预检与原207334/207348报告来源及输入/cache key相同，才运行一次8192更新；没有Slurm已结束依赖记录问题。

本地只检查224输入/120条演示，约96.890秒、峰值保留5.658GiB，16次入口更新、五预测快照。seed42参数起点与旧本地seed42记录相同，224输入/采样索引与既有覆盖局部记录相同；恢复参数/损失/预测一致，CLIP等冻结参数不变。五快照float64最大复算差1.63e-6cm/0.0000554°。验证器实际拒绝本地局部报告授权正式；没有开发评分、完整本地缓存或新PT。两Slurm脚本bash -n通过。该局部预检不代替集群完整缓存/输入验证。

资源：复用现有环境与权重，不安装包、不下载数据、不重跑小池。规划新增2GiB、200GiB计划上限保留15GiB；du为存储规划而非实时个人配额。正式一份约529MB非CLIP参数留集群，本地之后只取小报告/预测。新zip194670字节，SHA256 b82a5fb93ae57ff5b8ae2ea614610d6393a1fe43e8d1e33876967fdd640c3c5d。

## 现在仅提交完整预检

本地PowerShell：

```powershell
scp "D:/ntu_related/dissertation/Zero_shot/cluster_packages/bridge_coverage_replication_v1.zip" "D:/ntu_related/dissertation/Zero_shot/cluster_packages/bridge_coverage_replication_v1.sha256" zixiao005@10.97.216.128:/projects/Zeroshot/
```

集群：

```bash
cd /projects/Zeroshot
sha256sum -c bridge_coverage_replication_v1.sha256 && unzip -n bridge_coverage_replication_v1.zip && COVERAGE_REP_CHECK=$(sbatch --parsable bridge_coverage_replication_v1/preflight.sh) && echo "训练覆盖复核预检：$COVERAGE_REP_CHECK"
```

等待预检完成后再验证、提交正式，当前不要执行旧207334或207348脚本。未来正式提交模板仅作记录，不是现在的操作：

```bash
if /projects/Zeroshot/envs/multitask-preflight-v1/bin/python /projects/Zeroshot/bridge_coverage_replication_v1/validate_preflight.py "/projects/Zeroshot/runs/bridge-coverage-rep-preflight-${COVERAGE_REP_CHECK}"; then
  COVERAGE_REP_FIT=$(sbatch --parsable --export=ALL,COVERAGE_REP_PREFLIGHT_JOB="$COVERAGE_REP_CHECK" /projects/Zeroshot/bridge_coverage_replication_v1/fit.sh) && echo "训练覆盖正式复核：$COVERAGE_REP_FIT"
fi
```

证据：[本地独立核验](../reports/bridge_coverage_replication_local_preflight_20261016/independent_audit.json)、[冻结协议](../cluster_packages/bridge_coverage_replication_v1/protocol.json)、[可复用小池证明](../cluster_packages/bridge_coverage_replication_v1/reuse_control_proof.json)。重复开发集、两个训练初始化不构成未触碰测试或充分统计显著性证明；全部离线位姿指标，未运行抓取。

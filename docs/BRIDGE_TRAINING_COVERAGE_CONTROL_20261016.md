# 固定训练预算的数据覆盖泛化对照

**207348独立验收完成，预设候选通过。** 17文件/7预测、源码/输入/初始化/缓存指纹/8192日志核验，float64开发7.079412cm/14.559847°、第16目标8.943367cm/18.480202°；7/10演示位置优于旧控制和静止。旋转仅3/10优于静止，不能声称整体修复。下一步先复核一个独立训练初始化，检查可复用seed42小池202463，尚未准备提交；不重跑207348或下载PT/完整缓存。见[独立结果与下一项边界](BRIDGE_TRAINING_COVERAGE_RESULT_207348_20261016.md)。下文待下载/待核验/提交命令为历史。

**207348已完成，终端主要泛化指标改善，完整候选待核验。** COMPLETED/0:0、54分19秒、8192更新；原40开发位置7.079412cm/旋转14.559741°，两项均低于旧控制和静止，达到事前均值门槛。至少6/10演示优于旧控制、保存预测和实际协议仍待小工件独立核验，尚不判完整通过。不立即追加训练或种子。见[结果与排除PT的下载命令](BRIDGE_TRAINING_COVERAGE_RESULT_207348_20261016.md)。下文排队、待完成或提交命令为历史，勿重复执行。

**当前唯一正式作业：207348，预检207334验证器已通过。** 用户贴出`COMPLETE TRAINING-COVERAGE PREFLIGHT VERIFIED`及新编号，峰值保留5.630859GiB、暖更新0.134334秒、估计正式1.114904小时、缓存1.876761GiB。该耗时是含评价的资源估算，不含排队，也非完成时限承诺。完整预检报告未在本地下载独立复核；正式Slurm状态和结果尚未收到，无泛化改善结论。只跟踪207348，不重提作业或重编码，旧202278控制复用。下文未提交/待验证与提交命令均为历史，勿重复执行。

```bash
sacct -j 207348 --format=JobID,State,ExitCode,Elapsed,NodeList
tail -n 15 /projects/Zeroshot/logs/bridge-coverage-fit-207348.out
```

若PENDING，按需查看`squeue --start -j 207348`；8192训练更新后仍需对1836训练窗＋40开发窗进行三采样评价，日志暂时没有新更新不代表卡住。以完成状态和最终报告为准。

**完整预检207334已完成（用户终端证据）。** COMPLETED/0:0、4分15秒、gpu-a6000-1；常学习率与恢复同学习率首损失和最终训练探针数值一致。短预检的11—12cm训练误差不是开发指标，也不要求据此延长拟合。完整报告尚未下载本地；先执行原验证器核对完整输入、缓存、原控制初始化与资源，通过才提交唯一正式组。无新正式编号或泛化结果，不重复预检/编码。下文未完成状态和预检提交命令为历史。

在集群一次执行，验证失败不会提交；不依赖已结束Slurm作业的依赖记录：

```bash
if /projects/Zeroshot/envs/multitask-preflight-v1/bin/python /projects/Zeroshot/bridge_training_coverage_v1/validate_preflight.py /projects/Zeroshot/runs/bridge-coverage-preflight-207334; then
  COVERAGE_FIT=$(sbatch --parsable --export=ALL,COVERAGE_PREFLIGHT_JOB=207334 /projects/Zeroshot/bridge_training_coverage_v1/fit.sh) && echo "训练覆盖泛化正式对照：$COVERAGE_FIT"
fi
```

正式复用完整token缓存，原202278控制不重跑，按8192更新预算结束，评价原40开发位置与旋转。训练结束后还有1836训练窗/40开发窗的三次采样评价，不能因出现8192日志就认定作业应退出。

**当前预检编号：207334。** 用户已返回提交编号，尚未取得Slurm状态或完整报告，不能认定全输入/资源检查通过；正式尚未提交。只跟踪此预检，不重复上传或提交。下文未提交状态及首次提交命令为历史。

```bash
sacct -j 207334 --format=JobID,State,ExitCode,Elapsed,NodeList
tail -n 15 /projects/Zeroshot/logs/bridge-coverage-preflight-207334.out
```

首次完整预检包括415演示核对和1708个新增观察的冻结CLIP编码，可能比以前的短入口预检久；输入/编码进度会写入日志。仍在PENDING且无日志时只需等待。此阶段不评价开发，不是8192正式训练。

用户同意进入此项。新包已准备，仅做一次更大训练池对照，原202278控制复用。**尚未提交集群预检或正式作业。** 原增强206915已独立核验并结束，不重开。

## 本项回答什么

同一CLIP/8层Adapter/多尺度x0-DDPM100及真实状态端口，训练数据由64条演示/128窗扩大到已有405条/1836窗，原10条独立开发演示/40窗保持；测试分区不使用。全部原128训练窗均包含在新池中，按固定阶段起点保留更多真实轨迹覆盖，不做逐级窗口记忆。

初始化43、batch8、Adam、权重衰减0、裁剪1、8192训练更新、前4096步1e-4/后4096步1e-5、原动作单位与状态统计不变；夹爪冻结，不加入增强/EMA或其他数据集。固定的是**训练更新预算**，不是总墙钟或总编码/评价算力。原控制的只读中间观察不影响训练随机数和参数，已由既有观察开关对照证明；新组只在最后评价全部训练池与开发。

新8192×8个任务槽与原计划逐个一致；在同任务的大池内按固定洗牌流抽取窗口，并防止同一batch重复窗口。关闭微波炉16384次访问，其余四任务各12288次，与旧实验相同。新窗口各访问23—72次，平均35.695次；因任务池大小不同，不是每窗同次数。原128池各512次。任务配比没有随扩容改变。

主要判据仍为原40开发窗的位置≤7.391093cm、旋转≤15.118500°，两项低于旧控制且至少6/10演示位置优于旧控制。保留全部任务、演示和终点结果。训练误差只作诊断，不因未达到毫米拟合追加更新。单一池/初始化/预算，失败结束，不改抽样、学习率或追加轮数。

## 数据证明和本地检查范围

新包只保存1876行身份、8192×8抽样索引、415演示既有payload指纹与源码，不传RGB或完整特征。新解析沿用已冻结的dataset.py（与旧415演示几何审计同一哈希），既有审计已检查全部这些payload和相同起点的目标还原。运行时逐演示核对完整序列化内容指纹；原168观察另外使用原有跨平台数值参考，并检查原128训练统计与真实当前状态特征。新状态仅取起点t，不取未来状态作输入。没有重新执行已完成的415演示几何实验。

本地成功预检仅检查必要224输入、来自120条演示，其中包含原168观察以及前8实际批次需要的新增56图像。16次入口优化（共同4＋常学习率4＋恢复同学习率4＋降低学习率4），只观察四个原训练窗，不评分开发，不进行8192正式训练。本地未用槽显式填零以分配完整缓存所需的代表性显存，context严格禁止读取未检查槽；不能把它当作1876输入全验收。

首次入口尝试因误以为私有解析器也返回第五项mask而被字段检查拦下，未加载模型或产生优化更新。按其实际四项返回值修正包装，并以真实16步边界保证无填充；没有改生产解析器、目标或降低数值容差。原失败报告保留在`reports/bridge_training_coverage_local_preflight_20261016`。成功记录为[本地v2独立核验](../reports/bridge_training_coverage_local_preflight_v2_20261016/independent_audit.json)，约99.359秒、峰值保留5.658GiB。原本地初始化/状态/168输入与已验记录完全相同；恢复参数、损失和保存预测一致，CLIP/其他冻结参数保持不变；五快照float64复算最大位置差1.22e-6cm/旋转0.0000611°。

65536任务槽、405/10演示隔离、原128/40身份和全池覆盖已在CPU独立核验。验证器实际拒绝用本地局部预检授权正式作业。Python语法和两脚本bash -n通过。尚未得到集群全输入/缓存/资源结果，不能认定泛化有改善。

## 资源约束

集群预检必须核验全部415演示/1876窗，并创建一次约2GiB的完整冻结特征缓存；原168特征精确复用旧缓存，只编码1708个新增窗口。正式必须复用这份完整缓存，不重新编码。预检16次入口更新、四个已见训练窗探针，不计算开发指标。

完整预检峰值保留显存须<7.5GiB，按实际暖更新与探针采样耗时估算正式须<9000秒；Slurm申请3小时。最终训练结束后还要进行1836训练窗和40开发窗的三采样评价，8192日志出现后仍运行可能处于评价阶段。

缓存上限3GiB，一份最终非CLIP权重<1GiB留集群，报告/预测为小工件；规划新增总量5GiB并在200GiB计划上限下保留15GiB余量。du只是容量规划检查，不是实时用户配额查询。本地没有完整缓存或新权重、模型下载；已存在的数据/原缓存不重复拷贝。

## 操作：当前只提交预检一次

本地PowerShell：

```powershell
scp "D:/ntu_related/dissertation/Zero_shot/cluster_packages/bridge_training_coverage_v1.zip" "D:/ntu_related/dissertation/Zero_shot/cluster_packages/bridge_training_coverage_v1.sha256" zixiao005@10.97.216.128:/projects/Zeroshot/
```

集群终端：

```bash
cd /projects/Zeroshot
mkdir -p logs
sha256sum -c bridge_training_coverage_v1.sha256 && unzip -n bridge_training_coverage_v1.zip && COVERAGE_CHECK=$(sbatch --parsable bridge_training_coverage_v1/preflight.sh) && echo "训练覆盖泛化预检：$COVERAGE_CHECK"
```

查看：

```bash
sacct -j "$COVERAGE_CHECK" --format=JobID,State,ExitCode,Elapsed,NodeList
tail -n 15 "/projects/Zeroshot/logs/bridge-coverage-preflight-$COVERAGE_CHECK.out"
```

完整集群预检结束后，使用包内validate_preflight.py核对原控制身份/初始化、415演示全输入、缓存和资源，成功才提交唯一正式组；本地预检报告不能替代。当前尚未提交正式作业，不运行旧1836窗multitask_training.sh。

协议：[protocol.json](../cluster_packages/bridge_training_coverage_v1/protocol.json)。本轮结论只覆盖当前固定训练预算和同任务独立演示开发，不能认证未见任务或抓取，也不能把覆盖扩大效果归为纯演示数的单变量因果效应。

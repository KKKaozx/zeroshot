# 128窗口扩展拟合检查

目的：在两个初始化已核验64窗口拟合后，检验同一套网络能否拟合更多训练演示。保留200600的64窗口，再新增64窗口/32演示，总128窗口/64演示。初始化固定43。本地短预检及独立复核已通过，传输包已冻结。集群预检201662和正式201663已COMPLETED/0:0；正式日志的初步结果如下，尚待下载小文件独立核验。

## 201663核验完成

13份文件已齐全，8192条日志和八份预测独立CPU复算通过。总体最终1.829mm/0.579°、128/128，原64与新增64各64/64；后期四次总体与两组门槛均通过。开发仍8.728cm/15.847°，差于静止基线。扩容检查收口，不自动扩256。见[完整结果](BRIDGE_GROWTH128_RESULT_201663_20261013.md)。下文为下载过程和早期日志记录，缺件状态已解决。

补件后的核验更新：所需13份已有11份，运行脚本及8192条更新日志通过，六份训练预测已CPU float64复算。原64标签与200600精确一致，新增标签与冻结参考在1e-6内一致。6144/7168/7680原64均64/64、新64为64/64、64/64、63/64，三次总体与两组门槛均独立通过。仍缺`reduced_u8192_train.npz`和`reduced_u8192_development.npz`，所以最终训练与留出尚未复算，不能给出完整核验通过结论。见[最新核验](../reports/bridge_growth128_analysis_201663_20261013/analysis.json)和[已核验逐窗结果](../reports/bridge_growth128_analysis_201663_20261013/windows.csv)。

正式作业A6000耗时44分18秒，共8192更新。6144/7168/7680/8192训练平均位置分别约1.473/1.705/1.752/1.829mm，旋转约0.641/0.613/0.612/0.577°。最终总体fit gate=True；最后几个观测位置误差略回升，需要结合原64/新增64分组和逐窗门槛核对，不能仅凭总体平均宣布全部128窗口稳定达标。

开发40窗口最终8.728cm/15.847°，相较既定静止基线8.212cm/15.098°仍未通过。128训练拟合具有初步正面证据，泛化仍不足；夹爪冻结。本轮不自动扩到256，先下载报告、8192条更新日志和8份预测独立复算，再决定是否需要一次初始化复核或转入泛化诊断。预检仅32次短更新，门槛False属于预期的执行检查阶段，不构成正式拟合失败。

本地下载检查：目前只有report.json和plan.json，缺运行脚本、两份更新日志及八份NPZ，共11份。两份JSON中的冻结清单、源文件哈希声明、原104输入与200600一致性、168输入数值检查、预算及冻结状态核验通过，尚未核验实际运行脚本/8192日志或独立复算预测。报告记载原64最终1.885mm/0.568°、新增64为1.773mm/0.587°，各64/64；6144/7168/7680/8192总体覆盖128/128/127/128，两组四次门槛均True，7680新增组63/64。开发10演示位置5条优于各自静止基线，但总体仍差。上述指标是报告记载，完整复算须补齐文件。见[缺件清单及有限核验](../reports/bridge_growth128_analysis_201663_20261013/partial_inventory.json)。

## 数据选择与输入检查

新增仍限原五任务：关闭微波炉16窗口，关闭抽屉12、打开抽屉12、取西兰花12、拨杆12。每条演示两个窗口。合计任务窗口数32/24/24/24/24。候选仅来自既有Bridge训练分区，排除原64窗口涉及的32演示、全部开发演示和保留测试演示；按任务比例、分片/record身份排序取最前面的剩余演示，选每演示最早和最晚索引窗口。不按模型误差、动作幅度或“容易程度”挑选。

模型输入顺序为原64+新增64+开发40。数据审计顺序为原32+开发40+上次新增32+本次新增64，共168份输入。原104份核验复用已冻结bridge_growth64_v1的清单、helper及参考张量，运行时检查其哈希；新增64带新原始身份和处理后图像/动作参考，沿用图像/位姿绝对容差1e-6、夹爪精确要求。正式入口同时核对200600持久化报告及原104处理后哈希。本轮不读取保留测试目标。

复用集群已有211分片、CLIP和环境，不传原始数据或权重。新增参考约31.30MiB，本地旧64参考不重复复制。

## 固定训练配方

CLIP、8层Adapter、多尺度U-Net、动作归一化、图像处理、x0位姿MSE、重复当前观察的上下文、夹爪冻结、batch8、Adam、梯度裁剪1、零weight decay、原生DDPM100及三个采样种子均保持不变。十份生产源文件哈希一致，核心context、advance、状态拷贝/哈希和checkpoint函数AST完全一致；新入口只调整数据规模、预算、观测及输入检查，不修改模型或采样算法。

每次完整遍历128窗口需16次更新，512次遍历共8192更新；前256遍历1e-4，后256遍历1e-5，4096后降低学习率，不重置Adam。每窗仍512次抽样。更新总量随数据规模增加，不能将本实验解释为固定计算量的训练数据增量泛化对照。

训练观测0/2048/4096/6144/7168/7680/8192，全部128窗口；原64和新增64分别汇总。开发保持原40窗口/10演示，只在最终评估，不用于挑权重。三个采样输出不是三次训练。

拟合门槛沿用：每采样输出整体路径平均≤5mm/5°、终点平均≤10mm/10°；至少90%窗口在三输出中路径平均均达标，128需116个。两子组64窗口各需58个，独立报告门槛。4096未通过也记录并继续既定低学习率阶段，执行检查与拟合门槛分开。

若总体与两子组后期四次均通过，支持此128窗口配方和初始化的拟合能力，再考虑复核；若仅最终通过，记录中间波动；若新增窗口或两组失败，先分析固定数据的优化轨迹。训练拟合不能代表泛化、夹爪学习或机器人抓取成功。

## 资源与预检

短预检16次共同更新后，各16次连续/恢复同学习率/低学习率更新，每分支完整遍历128一次；所有168输入校验，观测16个原窗口+16个新增窗口，不保存PT。独立CPU核验日志、5份保存预测、恢复重放、总体/子组指标和真实argparse参数；shell语法另行检查。

实际本地预检258.78秒，峰值保留显存3.936GiB；168输入、64次记录更新、5份预测复算通过，恢复后的损失和保存数组精确一致，CLIP及其他冻结参数未改变。短跑拟合门槛未通过，仅核验执行流程，不能据此判断正式8192更新的拟合能力。见[预检独立核验](../reports/bridge_growth128_local_preflight_20261013/verification.json)。

冻结ZIP为32,854,589字节（31.33MiB），20份文件，SHA256为`bf09ec383d3ba490c026d16e17ffd8385cd10643176721f06c11ec79faaa64e1`；CRC及压缩包每份文件内容与本地一致。打包后本地剩余约67.61GiB。无需另下载环境、原始数据或CLIP；集群已有的`bridge_growth64_v1`目录和200600报告须保留。

默认A6000。预检30分钟、正式2小时是申请时限，不是运行耗时保证。正式仍只保存约0.5GiB最终可训练权重到集群，上限1GiB；本地只取小报告和预测。开跑前按200GiB规划限额，通过du检查预留15GiB及1GiB输出，未直接查询实时配额。

准备过程遇到一次文本替换范围歧义，生成器在冻结前停止；随后缩小到明确语句并继续，复用未冻结的参考张量前已验证选样及全部张量哈希。没有正式训练、没有修改既有冻结包或绕过校验。

## 上传与提交（本地已核验，可执行）

本地PowerShell：

```powershell
scp "D:/ntu_related/dissertation/Zero_shot/cluster_packages/bridge_growth128_v1.zip" "D:/ntu_related/dissertation/Zero_shot/cluster_packages/bridge_growth128_v1.sha256" zixiao005@10.97.216.128:/projects/Zeroshot/
```

集群终端连续执行，校验或提交失败会停止后续命令：

```bash
cd /projects/Zeroshot
sha256sum -c bridge_growth128_v1.sha256 &&
unzip -n bridge_growth128_v1.zip &&
GROWTH128_CHECK=$(sbatch --parsable /projects/Zeroshot/bridge_growth128_v1/preflight.sh) &&
GROWTH128_FIT=$(sbatch --parsable --kill-on-invalid-dep=yes --dependency="afterok:$GROWTH128_CHECK" --export="ALL,GROWTH_PREFLIGHT_JOB=$GROWTH128_CHECK" /projects/Zeroshot/bridge_growth128_v1/fit.sh) &&
printf '128窗口预检：%s；正式拟合：%s\n' "$GROWTH128_CHECK" "$GROWTH128_FIT" &&
squeue --start -j "$GROWTH128_CHECK,$GROWTH128_FIT"
```

```bash
sacct -j "$GROWTH128_CHECK,$GROWTH128_FIT" --format=JobID,State,ExitCode,Elapsed,NodeList
tail -n 15 "/projects/Zeroshot/logs/bridge-growth128-preflight-$GROWTH128_CHECK.out"
tail -n 20 "/projects/Zeroshot/logs/bridge-growth128-fit-$GROWTH128_FIT.out"
```

最终小报告下载：

```powershell
$GrowthJobId = '201663'
$Dest = "D:/ntu_related/dissertation/Zero_shot/reports/bridge-growth128-fit-$GrowthJobId"
New-Item -ItemType Directory -Force -Path $Dest | Out-Null
$Remote = "zixiao005@10.97.216.128:/projects/Zeroshot/runs/bridge-growth128-fit-$GrowthJobId"
scp "$Remote/*.json" "$Remote/*.jsonl" "$Remote/*.npz" "$Remote/runner_snapshot.py" "$Dest/"
```

- [64窗口两个初始化的核验结果](BRIDGE_GROWTH64_REPLICATION_RESULT_201429_20261013.md)

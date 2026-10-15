# 当前末端状态：一次初始化复核

202278首轮已独立核验：两组128/128训练窗拟合，真实状态开发路径位置/旋转均小幅改善，7/10演示位置改善，但仍差于静止基线。依据首轮运行前固定的计划，本项只做一次完整配对复核。

## 固定设置和判断规则

共享模型初始化43改为42；零/真实状态两组均从42开始。选择42在看到这次复核结果前固定，不根据开发误差选种子。图像、语言、当前状态与统计量、128训练/40开发窗口、批次顺序、训练噪声种子、八层Adapter、多尺度U-Net、x0目标、原生DDPM100、采样种子、batch8以及8192更新均保持原样，4096更新后学习率1e-4降为1e-5。CLIP和夹爪仍冻结，不增加训练数据。

原始输入/source清单继续复用`bridge_growth128_v1/manifest.json`，其中43是原实验历史元数据；本轮运行初始化由冻结的复核协议及两个训练入口明确固定为42，运行报告也须为42。原包不修改。将两个新训练入口的`init = 42`反向恢复为43后，与首轮入口逐字一致；配对协调器仅新增协议/实际初始化一致性断言，不改变训练或评价。

判定顺序：

1. 预检验证零接口精确重放同初始化的旧模型、真实状态有梯度/贡献、观测输入和训练统计一致、冻结模块/恢复分支正确。失败则停止正式作业。
2. 正式配对先核验两组训练拟合。未拟合时不能据此否定状态输入的泛化价值，也不自动加轮数。
3. 两组均拟合后，检查真实组是否再次整体路径位置和旋转均优于零组，并且至少6/10开发演示位置改善。报告全部任务和演示，不挑好的结果。
4. 若未重复，记录为不稳定候选并停止此方案，不追加种子、接口层数或状态权重扫描。若重复成立，可确认两次初始化上的部分收益；只有两项均优于静止才称本开发协议通过。开发集被多次使用，仍不能替代封存测试或闭环抓取。

## 当前准备状态

2026-10-14完整核验已完成：27份正式必要文件、16份快照、每组8192更新和集群预检独立复算通过。零/真实组训练127/128及117/128达标，均满足原≥90%总体门槛，不能说全部窗口达标；真实组开发路径位置/旋转均改善，6/10演示位置改善，候选条件重复。真实组8.382cm/15.994°仍差于静止8.212cm/15.099°。本轮配对收口，无更多状态种子或参数扫描；下一阶段准备同Bridge作者实现参照，尚未启动。见[完整复核结果](BRIDGE_STATE_REPLICATION_RESULT_202463_20261014.md)。以下准备、提交和待下载为历史状态。

2026-10-14最新状态：用户提供Slurm结果，202462预检和202463正式复核均COMPLETED/0:0，分别18分55秒、1小时22分22秒，同在gpu-a6000-1。所贴末尾日志为预检real的32次短更新；不能用其gate False判断正式拟合。正式报告/预测尚未本地下载，尚不能判断候选收益是否重复。下一项仅下载并独立验收，不追加训练。下段是提交时历史状态。

用户现已提交集群预检202462、正式复核202463；尚未收到状态或结果。下段“未提交”是本地准备时状态，正式实验尚不能判定完成或通过。勿重复提交。固定作业号检查命令：

```bash
squeue --start -j 202462,202463
sacct -j 202462,202463 --format=JobID,State,ExitCode,Elapsed,NodeList
if [ -f /projects/Zeroshot/logs/bridge-state-rep-preflight-202462.out ]; then
  tail -n 12 /projects/Zeroshot/logs/bridge-state-rep-preflight-202462.out
fi
```

复核协议已在新结果前固定，本地短预检和独立CPU验收均通过；未运行完整8192次拟合，未提交集群。三组同初始化的原模型初始哈希一致；零组全部原参数、五份预测/上下文/目标快照及64次损失与旧模型精确相同，最大差异0。真实状态首步梯度范数0.114714，最终上下文贡献最大绝对值0.008913；参数实际改变，零接口保持零，冻结模块不变，Adam恢复精确重放。

168输入、训练统计量和旋转特征独立验收通过；15份预测的真实单位指标独立复算通过，最大指标差7.22e-5。真实启动参数和两个Bash脚本语法通过。短跑仅验收计算链；其fit gate False不表示正式拟合失败。验收用时845.66秒，其中三组GPU短跑约262.70/272.02/278.61秒；峰值保留显存约3.936GiB。小文件合计4,625,739字节，无PT、没有本地完整训练或开发生成评价。

包复用已有原始数据、环境和CLIP，没有下载资产。ZIP已冻结，29,288字节（约28.6KiB）、11文件，SHA256为`f33543e13f3832c341226706e360730a8bf06e59c1e1a56d8da96a47ea2c8914`；归档CRC及逐文件内容核验通过。集群仍需在其环境下重做预检，不能跳过门控。跨Windows/Linux的原模型初始化哈希可能因库版本不同而不同，精确重放在同一运行环境内比较。

- [冻结上传ZIP](../cluster_packages/bridge_state_replication_v1.zip)
- [上传校验文件](../cluster_packages/bridge_state_replication_v1.sha256)
- [本地预检报告](../reports/bridge_state_replication_local_preflight_20261013/report.json)
- [独立CPU验收](../reports/bridge_state_replication_local_preflight_20261013/verification.json)

## 上传和提交

以下命令仅在本地验收通过、ZIP已冻结后使用。Windows PowerShell上传：

```powershell
scp "D:/ntu_related/dissertation/Zero_shot/cluster_packages/bridge_state_replication_v1.zip" "D:/ntu_related/dissertation/Zero_shot/cluster_packages/bridge_state_replication_v1.sha256" zixiao005@10.97.216.128:/projects/Zeroshot/
```

集群终端（Bash）：

```bash
cd /projects/Zeroshot
mkdir -p logs
sha256sum -c bridge_state_replication_v1.sha256 && unzip -n bridge_state_replication_v1.zip
```

校验和解压均成功后提交。保留原`bridge_growth64_v1`、`bridge_growth128_v1`及引用的历史报告：

```bash
STATE_REP_PREFLIGHT=$(sbatch --parsable bridge_state_replication_v1/preflight.sh) &&
STATE_REP_FIT=$(sbatch --parsable --dependency=afterok:"$STATE_REP_PREFLIGHT" --export=ALL,STATE_REP_PREFLIGHT_JOB="$STATE_REP_PREFLIGHT" bridge_state_replication_v1/fit.sh)
echo "状态复核预检：$STATE_REP_PREFLIGHT；正式复核：$STATE_REP_FIT"
squeue --start -j "$STATE_REP_PREFLIGHT,$STATE_REP_FIT"
```

```bash
sacct -j "$STATE_REP_PREFLIGHT,$STATE_REP_FIT" --format=JobID,State,ExitCode,Elapsed,NodeList
tail -n 12 "/projects/Zeroshot/logs/bridge-state-rep-preflight-$STATE_REP_PREFLIGHT.out"
tail -n 12 "/projects/Zeroshot/logs/bridge-state-rep-fit-$STATE_REP_FIT.out"
```

默认申请A6000，预检45分钟、正式3小时是作业上限，不是排队或完成时间承诺。每组仅保存最终可训练权重，两份合计约1GiB留在集群；本地最终只下载报告、更新日志和预测。启动脚本按200GiB规划限额检查新增2GiB和15GiB余量，`du`不是实时配额查询。

## 下载小结果

本次已知作业号，集群终端执行：

```bash
cd /projects/Zeroshot/runs
tar --exclude='*.pt' --exclude='__pycache__' -czf /projects/Zeroshot/bridge_state_rep_results_202463.tar.gz bridge-state-rep-preflight-202462 bridge-state-rep-fit-202463
du -h /projects/Zeroshot/bridge_state_rep_results_202463.tar.gz
```

本地Windows PowerShell执行：

```powershell
scp zixiao005@10.97.216.128:/projects/Zeroshot/bridge_state_rep_results_202463.tar.gz "D:/ntu_related/dissertation/Zero_shot/reports/"
tar -xzf "D:/ntu_related/dissertation/Zero_shot/reports/bridge_state_rep_results_202463.tar.gz" -C "D:/ntu_related/dissertation/Zero_shot/reports/"
```

只下载JSON、预测NPZ、日志和协议小文件，不含集群PT；完成后复算全部训练/开发预测及10条开发演示，执行先前固定的复核规则。

两份作业成功结束后，集群终端打包（不含权重）：

```bash
cd /projects/Zeroshot/runs
tar --exclude='*.pt' --exclude='__pycache__' -czf "/projects/Zeroshot/bridge_state_rep_results_${STATE_REP_FIT}.tar.gz" "bridge-state-rep-preflight-${STATE_REP_PREFLIGHT}" "bridge-state-rep-fit-${STATE_REP_FIT}"
```

本地SCP时将文件名里的作业号换为实际正式作业号，下载到本项目`reports`后解压。无需下载PT。

- [首轮完整结果](BRIDGE_STATE_PAIR_RESULT_202278_20261013.md)
- [预设排查计划](BRIDGE_DIAGNOSTIC_PLAN_20261013.md)
- [复核包生成器](../scripts/build_bridge_state_replication_v1.py)

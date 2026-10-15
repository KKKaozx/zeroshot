# 下一步：扩大训练演示后的位姿拟合检查

200600完整64窗口训练及13份小文件独立核验已完成：原32、新增32各32/32达标，整体1.275mm/0.602°；留出仍9.749cm/18.894°，泛化未通过。见[最终结果与证据范围](BRIDGE_GROWTH64_RESULT_200600_20261013.md)。下方为固定设计和执行历史，请勿重复提交。

最新状态（2026-10-13）：修复启动后正式作业200600日志已输出4096更新及报告。训练64窗口平均1.2746mm/0.6009°，日志fit gate=True；开发40窗口9.7494cm/18.8939°，仍差于静止基线。集群状态COMPLETED、报告执行标志、原32/新增32分组及逐窗数组尚待下载核验；不能把日志门槛通过解释为64/64窗口全部达标。本轮仅一个初始化，不等于全部1836训练窗口拟合或泛化成功。

历史启动修复：集群预检200365产生了三分支日志，正式200366在Python参数解析时失败（多余参数`+`），尚未训练。已定位为生成器和fit.sh的启动参数错误。原冻结包、清单、源文件及预检保持不变；另生成修正版启动脚本，复用200365预检。集群预检执行标志仍由正式入口读取report.json验证。下方提交命令均为历史，200600已提交运行，请勿重复提交。

在本地PowerShell上传小型修复工具：

```powershell
scp "D:/ntu_related/dissertation/Zero_shot/scripts/fix_bridge_growth64_launcher.py" zixiao005@10.97.216.128:/projects/Zeroshot/
```

在集群终端运行：

```bash
/projects/Zeroshot/envs/multitask-preflight-v1/bin/python /projects/Zeroshot/fix_bridge_growth64_launcher.py /projects/Zeroshot
if /projects/Zeroshot/envs/multitask-preflight-v1/bin/python -c "import json; r=json.load(open('/projects/Zeroshot/runs/bridge-growth64-preflight-200365/report.json')); assert r['stage']=='preflight' and r['execution_checks_passed'] and r['branch_restore_passed'] and r['input_contract_passed']; print('预检报告检查通过')" && bash -n /projects/Zeroshot/bridge_growth64_fit_fixed.sh; then
  if GROWTH_FIT=$(sbatch --parsable --export=ALL,GROWTH_PREFLIGHT_JOB=200365 /projects/Zeroshot/bridge_growth64_fit_fixed.sh); then
    echo "新正式拟合作业：$GROWTH_FIT"
    squeue --start -j "$GROWTH_FIT"
  fi
fi
```

工具先严格检查原脚本SHA256，再只删除一个多余`+`，核验修正后哈希，写入包外新文件。正式脚本仍校验整个原冻结包，训练入口仍比对预检输入/源文件/清单；没有跳过校验。本地从真实训练入口抽取argparse定义，验证修正后命令参数全部可解析；生成器同步修正。核验记录见[启动参数核验](../cluster_packages/bridge_growth64_launcher_fix_20261013/verification.json)。失败不提供模型效果证据，不需重复短预检。

修正启动脚本后用户提交返回`Job dependency problem`，未分配新作业ID。旧作业依赖可能超过Slurm的MinJobAge保留期（未读取集群配置，不能确认根因）。改为提交前读取200365持久化预检报告并检查通过标志，移除旧afterok依赖；正式入口仍验证报告、清单、源文件和输入一致。参见[Slurm依赖说明](https://slurm.schedmd.com/sbatch.html#OPT_dependency)。

原准备证据：ZIP为16,823,540字节（约16.04MiB），没有原始TFRecord或模型权重。104份输入校验通过；恢复训练的参数、loss和预测完全相同；旧/新分组指标已独立CPU复算。本地预检227.53秒、峰值保留显存3.871GiB，不保存PT；预检未达到训练拟合门槛是16次更新的正常执行结果，不能据此判断完整实验效果。见[预检独立核验](../reports/bridge_growth64_local_preflight_20261012/verification.json)。

## 数据和固定设置

保留原16条演示、32窗口，在同一个已经上传的Bridge训练分区增加16条演示、32窗口，合计32条训练演示、64窗口。开发仍为原10条演示、40窗口，保留测试演示不读取目标。新增演示与原训练、开发、保留测试演示均不重叠。

按原五任务比例新增：关闭微波炉4条，关闭抽屉3条，打开抽屉3条，取西兰花3条，拨杆3条。每条两窗口。只按固定清单中的任务、演示身份和窗口起始索引选择：各任务剩余训练演示按分片/record排序，取最前面有至少两窗口的演示，选其首尾窗口；不按误差、动作幅度或成功结果筛选。任务种类未增加，独立训练演示覆盖增加。

旧10份源文件保持哈希不变，CLIP、8层Adapter、多尺度U-Net、x0监督、完整DDPM100、归一化、图像处理、重复当前图像上下文、Adam、batch8、梯度裁剪1、零weight decay、冻结夹爪均保持原设置。生成的执行入口只调整数据规模、预算、观测和输入校验；没有修改生产模型、dataset或采样公式。

使用预先固定的初始化43，从头训练。选择43是因为它在此前原学习率下退化最明显，不是在新结果出来后挑种子。第一次64窗口检查只有一个初始化，结果成功后仍需另行复核，不能据此替代三个初始化的证据。

## 训练预算和判据

32窗口原配方为2048次更新，每窗512份抽样；前256次完整遍历使用1e-4，后256次遍历使用1e-5。64窗口每次遍历需要8个batch，因此总4096次更新，在2048次更新后降至1e-5，不重置Adam。每个窗口仍512份抽样。总计算量随样本数增加，不能把这次解释为固定总计算量的数据规模单变量比较。

不按开发结果挑选权重。训练在0/1024/2048/3072/3584/3840/4096评估全部64窗口，原32和新增32分别汇总；开发40窗口只在最终4096评估。固定三个采样种子1101/1102/1103，三个采样输出不是三次训练。

总体训练门槛保持原相对标准：每个采样输出路径平均≤5mm/5°、窗口终点平均≤10mm/10°，至少90%窗口在三个输出下分别的路径平均均≤5mm/5°，64窗口需要至少58个。另报原32和新增32的达标数量及门槛，避免总体均值掩盖某一部分失败。

2048时即使未通过拟合门槛，也继续完成既定低学习率阶段，并记录失败；不把提前未拟合当作执行错误。完整执行通过和训练拟合通过分别报告。

若原窗口保持拟合、新增窗口失败，查看新增任务/演示及覆盖；若两部分均失败，查看整体优化轨迹与预算；若两部分均拟合而开发仍差，训练容量与泛化可以更清楚地区分。这些是下一次结果的判定方向，不是提前确定根因。本轮不加入其他数据集、不扫网络结构或噪声设置、不进行PyBullet执行。

## 输入预检、资源与证据

104窗口全部读取并验证。原72份输入沿用已经上传的原始身份及数值参考；新增32份输入带新的原始身份、处理后图像/动作数值参考，容差与原实验相同（图像/位姿1e-6，夹爪及当前测量精确）。额外参考张量约16MiB，属于小型数据处理核验资产，不下载新的原始数据集或CLIP权重。

短预检8次共同训练后，比较连续8次与恢复同一参数/完整Adam后8次训练，要求参数、loss、预测完全一致；低学习率组首步loss一致而参数更新不同。训练8个batch遍历全部64窗口，观测只取16个原窗口与16个新增窗口，共32个，正式观测则覆盖全部64个。短预检只确认执行，不评价拟合。

集群复用现有环境、Bridge和CLIP。默认A6000，预检时限25分钟，正式90分钟；时限不是运行时间估计。正式只保留最终可训练权重，约0.5GiB，上限1GiB；不保存共同阶段Adam检查点。本地预检不保存PT。开跑前按200GiB规划限额预留15GiB及1GiB输出，du检查不是实时配额查询。下载仅小报告、NPZ、JSONL。

清理说明字段时中断了最初的本地预检，初稿和中断输出保留于draft/interrupted目录，未打包或上传，不作为成功或拟合证据。最终清单重新冻结后再预检。

## 上传与提交

先在本地PowerShell上传最终冻结包：

```powershell
scp "D:/ntu_related/dissertation/Zero_shot/cluster_packages/bridge_growth64_v1.zip" "D:/ntu_related/dissertation/Zero_shot/cluster_packages/bridge_growth64_v1.sha256" zixiao005@10.97.216.128:/projects/Zeroshot/
```

集群终端校验并解压：

```bash
cd /projects/Zeroshot
mkdir -p logs
sha256sum -c bridge_growth64_v1.sha256 && unzip -n bridge_growth64_v1.zip
```

校验OK且解压完成后，提交一次：

```bash
GROWTH_CHECK=$(sbatch --parsable /projects/Zeroshot/bridge_growth64_v1/preflight.sh)
GROWTH_FIT=$(sbatch --parsable --dependency="afterok:$GROWTH_CHECK" --export="ALL,GROWTH_PREFLIGHT_JOB=$GROWTH_CHECK" /projects/Zeroshot/bridge_growth64_v1/fit.sh)
printf '64窗口预检：%s；正式拟合：%s\n' "$GROWTH_CHECK" "$GROWTH_FIT"
squeue --start -j "$GROWTH_CHECK,$GROWTH_FIT"
```

正式拟合只运行一组64窗口训练，32窗口基线使用已核验200150，不重复训练原学习率组。两个作业分别是短预检和完整训练，不是两个正式训练种子。

```bash
sacct -j "$GROWTH_CHECK,$GROWTH_FIT" --format=JobID,State,ExitCode,Elapsed,NodeList
tail -n 15 "/projects/Zeroshot/logs/bridge-growth64-preflight-$GROWTH_CHECK.out"
tail -n 15 "/projects/Zeroshot/logs/bridge-growth64-fit-$GROWTH_FIT.out"
```

完成后本地只取小文件：

```powershell
$GrowthJobId = '实际正式编号'
$GrowthDest = "D:/ntu_related/dissertation/Zero_shot/reports/bridge-growth64-fit-$GrowthJobId"
New-Item -ItemType Directory -Force -Path $GrowthDest | Out-Null
$GrowthRemote = "zixiao005@10.97.216.128:/projects/Zeroshot/runs/bridge-growth64-fit-$GrowthJobId"
scp "$GrowthRemote/*.json" "$GrowthRemote/*.jsonl" "$GrowthRemote/*.npz" "$GrowthRemote/runner_snapshot.py" "$GrowthDest/"
```

- [固定数据及配方核验](../reports/bridge_growth64_recipe_verification_20261012.json)
- [上一步三个初始化结果](BRIDGE_LATE_LR_REPLICATION_RESULT_200208_20261012.md)

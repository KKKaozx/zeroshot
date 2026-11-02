# 固定状态规律＋扩散修正：205648已独立复核，候选收口

**最终更新：** 27个结果文件、13快照及8192正式更新独立核验完成。开发7.185cm/16.697°，位置8/10演示优于旧扩散、7/10优于静止；旋转4/10优于旧扩散，完整候选因旋转要求失败。状态基础6.280cm仍更好。标签、基础预测、残差还原均核验通过，最终XYZ没有还原裁剪。按原预算结束，不启动新训练或比例/种子扫描。见[完整结果与限制](BRIDGE_STATE_PRIOR_RESULT_205648_20261015.md)。下文“待下载/复核”保留为终端阶段历史。

2026-10-15。用户贴回正式候选 **205648**完成：COMPLETED/0:0，gpu-a6000-1，37分钟，8192更新。日志训练1.419mm/0.632°、位姿门槛True；开发7.185cm/16.697°。完整报告和预测尚未本地取得独立复算，以下为终端初步结果，不是最终工件验收。

## 当前初步结论：位置优于静止，完整候选未通过

| 原40开发窗口 | 路径位置 | 路径旋转 |
|---|---:|---:|
| 静止基线 | 8.212cm | 15.099° |
| 原扩散202278真实状态组 | 8.349cm | 15.256° |
| 新状态基础＋扩散修正205648（日志） | **7.185cm** | **16.697°** |
| 状态基础预测单独使用 | 6.280cm | 未评价 |

按日志，新方案位置比静止约好12.5%、比原扩散约好13.9%；位置泛化确有部分改善，不能再说本次位置仍差于不动。但旋转变差，超过预定不劣于原模型15.276°的候选门槛；位置也没有超过固定状态基础预测器。即使后续逐演示条件满足，报告均值若复核成立，完整候选仍未通过，不采用为已解决泛化的配方。

按原预算收口，只取回结果复核，不追加训练、改权重比例或种子。训练门槛True不替代上述留出判据。Slurm的Invalid job id表示作业已不在活动队列；会计记录COMPLETED/0:0及最终报告路径显示它正常结束，不因该提示重复提交。

### 只打包小结果并下载

集群Bash（排除所有PT权重；编码缓存本来也不在这两个结果目录）：

```bash
tar -czf /projects/Zeroshot/bridge_prior_results_205648.tar.gz --exclude='*.pt' -C /projects/Zeroshot/runs bridge-prior-fit-205648 bridge-prior-preflight-205618
```

本地PowerShell：

```powershell
scp zixiao005@10.97.216.128:/projects/Zeroshot/bridge_prior_results_205648.tar.gz "D:/ntu_related/dissertation/Zero_shot/reports/"
tar -xzf "D:/ntu_related/dissertation/Zero_shot/reports/bridge_prior_results_205648.tar.gz" -C "D:/ntu_related/dissertation/Zero_shot/reports"
```

复核原始单位预测＝残差＋基础轨迹（包含已声明裁剪）、标签/源码/原128/40身份、8192日志、冻结条件及全部10演示位置/旋转。文件未取得之前不报告逐演示改善数量、终点误差或最终采用结论。终端证据：[记录](../reports/bridge_prior_205648_terminal_summary.json)。

## 查看205648（历史步骤，现已完成）

```bash
squeue --start -j 205648
sacct -j 205648 --format=JobID,State,ExitCode,Elapsed,NodeList
if [ -f /projects/Zeroshot/logs/bridge-prior-fit-205648.out ]; then
  tail -n 20 /projects/Zeroshot/logs/bridge-prior-fit-205648.out
fi
```

不重复提交预检或正式候选，不改固定8192更新预算、128/40划分或评价指标。排队时可能没有日志；完成后需要报告和保存的原单位预测，不能只看训练损失或Slurm退出码判定泛化。

## 205618完成：先校验报告，再提交唯一正式候选

日志中的gate False属于4+4短更新预检，不是正式拟合/泛化判决。CuBLAS/attention确定性警告没有导致本次作业失败；仍以实际报告里的恢复检查和冻结检查为准。

以下集群Python命令先断言关键报告字段及协议/入口/辅助源码身份，再提交一次正式作业；记录提交号，重复运行同命令只读取已记录编号。预检已结束且报告先核验，不依赖可能被Slurm清除的历史afterok记录。正式入口仍会再次校验预检及源文件，原源码或协议不匹配即停止。

```bash
/projects/Zeroshot/envs/multitask-preflight-v1/bin/python - <<'PY'
import hashlib, json, subprocess
from pathlib import Path
root = Path('/projects/Zeroshot')
p = root / 'bridge_state_prior_v1'
r = json.loads((root / 'runs/bridge-prior-preflight-205618/report.json').read_text())
assert all(r[k] for k in ('execution_checks_passed', 'input_contract_passed', 'branch_restore_passed', 'clip_unchanged', 'other_frozen_unchanged'))
assert r['stage'] == 'preflight' and r['state_arm'] == 'real'
assert r['zero_prior_first_loss_difference'] == 0 and r['final_update'] == 8
for key, name in [('residual_protocol_sha256', 'protocol.json'), ('residual_entry_sha256', 'run_prior.py'), ('residual_helper_sha256', 'prior_contract.py')]:
    assert hashlib.sha256((p / name).read_bytes()).hexdigest() == r[key]
assert (p / 'full_tokens.pt').is_file()
note = root / 'runs/bridge-prior-submission-205618.txt'
if note.exists():
    print('已提交，不重复运行；正式作业编号：', note.read_text().strip())
else:
    job = subprocess.check_output(['sbatch', '--parsable', '--export=ALL,PRIOR_PREFLIGHT_JOB=205618', str(p / 'fit.sh')], text=True).strip()
    note.write_text(job + '\n')
    print('预检报告校验通过；正式泛化作业编号：', job)
PY
```

用户已执行上述报告校验及提交步骤，返回正式编号205648。下方提交命令保留供追溯，不再执行；使用原40开发窗口及固定停止条件，不重新做预检或训练旧对照。

## 唯一候选与保留条件

上一项在原Bridge128训练/40开发划分上，任务＋当前末端状态的固定读出开发位置6.280cm，优于静止8.212cm和原扩散8.349cm。读取系数现在只用128训练标签导出，固定正则1.0，没有根据开发结果重新拟合或挑参数。

新候选将原XYZ动作表示为：

`原位置目标 = 固定状态基础轨迹 + 扩散修正量`

训练时从原位置标签减去基础轨迹，原DDPM学习剩余量；生成仍是原100步扩散采样，之后加回同一个基础轨迹，最后按原完整动作目标评分。基础轨迹只接任务指令类别和当前观测XYZ/RPY，不接未来状态、目标物体真值、样本编号或动作真值。训练使用拟合于整个训练分区的基础预测；开发使用同一组固定系数预测，不使用开发标签求解。

保留原冻结CLIP、8层Adapter、扩散网络、真实当前状态投影、图像处理、初始化43、128训练/40开发窗口、8192更新、batch8、Adam、4096更新后的1e-4→1e-5及三采样种子。四元数标签不做残差化；夹爪仍被冻结，不能声称完整抓取策略有效。动作表达改变也改变了网络所见的含噪动作原点，不能把结果解释为某一模块bug已被证明。

残差仍使用原±3位置采样边界，加回基础位置后再应用原±3完整位置边界，并保存裁剪计数；这种裁剪参考系变化属于本候选定义。训练残差最大幅度约1.7704单位，小于3，没有训练标签因该边界被排除。单位仍是1单位=0.1米，评分乘回厘米；不额外改变动作缩放。

## 不重复执行与源码复用

复用202278真实状态组作为原对照，不重训旧组。新入口逐项断言后包装原冻结训练入口，原核心文件不改，并保存实际执行的生成入口及全部指纹。168窗原字段和当前状态合同继续核验。预检与正式作业共享一个约172MiB完整CLIP token缓存，绑定输入、源码、模型、tokenizer和软件身份；不能将身份不符的缓存直接用于新输入。

本地原先只保留压缩视觉描述符，没有这批Bridge完整token，因此本次补算一次并保存。这个缓存不会装入上传ZIP；集群在自己的环境预检时算一次，正式运行复用集群缓存。新增包没有图像、数据集或模型权重，无安装或下载。

## 本地已通过的检查及限制

- 原168输入、目标和当前状态合同通过；固定系数导出与已有预测最大差9.55e-8项目单位。
- 减去基础轨迹再加回全部目标，最大差1.20e-7单位（约1.20e-6cm）；旋转及夹爪标签逐元素不变。
- 零基础轨迹时，原始前向输出逐元素一致；没有重训原对照。
- 实际16次操作更新包含4次共同更新、4次常学习率、4次恢复同学习率重放和4次降学习率；恢复后的参数、损失和预测精确一致。
- Adapter、实际扩散头、状态投影均有非零梯度，CLIP和夹爪保持冻结；5份保存预测的残差＋基础轨迹重建均精确吻合。
- 使用RTX4060 Laptop，约103秒、峰值保留显存3.934GiB，没有保存本地模型权重或评价开发目标。少量更新的拟合门槛未过不构成候选失败，预检只检查入口。

初稿在优化前主动停止，修正缓存元数据中的Torch版本序列化类型后完成有效预检，旧未完成目录保留。Bash在沙箱内受系统对象权限限制；随后只读语法检查成功，没有运行或提交集群作业。当前包装器、数值和预检指纹记录见[独立验收](../reports/bridge_state_prior_preparation_20261015/preflight_audit.json)。

## 固定验收与停止

只启动一个新候选，不重训原控制。集群预检必须成功，正式仍为一次8192更新的最终权重，无中途开发选模或事后增加预算。主要评价原40开发窗口的完整位置/旋转，并列出10演示；原训练位姿门槛保留为质量约束。

候选筛选要求：开发位置同时优于静止8.212cm和原扩散8.349cm，至少6/10演示位置优于原扩散，旋转不劣于原模型（含0.02°数值容差），训练原位姿门槛通过。同时必须报告与状态基础预测6.280cm的差距；只超过旧扩散仍可能是部分改善，不能宣称扩散残差比基础预测更有效。

无论结果如何，到固定预算结束，不扫描基础权重、残差比例、正则、种子或增加轮数。不使用预留测试、不做物理回合；尚未唯一定位根因，也不预告一定改善。现有开发集多次用于诊断，新结果属于探索性候选，不能称作新任务零样本成功。

## 已提交205618：查看状态（历史步骤）

在集群终端运行：

```bash
squeue --start -j 205618
sacct -j 205618 --format=JobID,State,ExitCode,Elapsed,NodeList
if [ -f /projects/Zeroshot/logs/bridge-prior-preflight-205618.out ]; then
  tail -n 20 /projects/Zeroshot/logs/bridge-prior-preflight-205618.out
fi
```

排队时日志可能尚未创建；已结束作业可能不在squeue中，以sacct和实际报告判断。205618现已结束；下文上传和原提交命令仅供追溯，不再执行。正式提交使用上方带报告校验和已提交编号记录的命令。

## 上传并只提交预检（历史步骤）

本地PowerShell：

```powershell
scp "D:/ntu_related/dissertation/Zero_shot/cluster_packages/bridge_state_prior_v1.zip" "D:/ntu_related/dissertation/Zero_shot/cluster_packages/bridge_state_prior_v1.sha256" zixiao005@10.97.216.128:/projects/Zeroshot/
```

集群Bash：

```bash
cd /projects/Zeroshot
mkdir -p logs
sha256sum -c bridge_state_prior_v1.sha256 && unzip -n bridge_state_prior_v1.zip && bash -n bridge_state_prior_v1/preflight.sh && bash -n bridge_state_prior_v1/fit.sh
PRIOR_CHECK=$(sbatch --parsable bridge_state_prior_v1/preflight.sh)
echo "状态基础轨迹预检：$PRIOR_CHECK"
```

查看预检（沿用同一个终端变量，或填实际编号）：

```bash
sacct -j "$PRIOR_CHECK" --format=JobID,State,ExitCode,Elapsed,NodeList
tail -n 15 "/projects/Zeroshot/logs/bridge-prior-preflight-$PRIOR_CHECK.out"
```

本包默认A6000，当前是否有空闲资源未知。若手动改为已兼容的A40，可在提交时加`--gres=gpu:a40:1`，不要同时重复提交相同预检。

预检独立核验成功后，正式入口已准备，下面命令保留供下一阶段使用，目前不执行：

```bash
PRIOR_TRAIN=$(sbatch --parsable --dependency="afterok:$PRIOR_CHECK" --export="ALL,PRIOR_PREFLIGHT_JOB=$PRIOR_CHECK" bridge_state_prior_v1/fit.sh)
echo "状态基础轨迹正式泛化对照：$PRIOR_TRAIN"
```

Slurm已经清除过早作业的依赖时，不能直接跳过报告核验再去掉依赖。本次先只提交预检，避免重复和混淆作业号。

证据：[固定协议](../cluster_packages/bridge_state_prior_v1/protocol.json)、[训练集导出的基础系数](../cluster_packages/bridge_state_prior_v1/prior.json)、[显式入口](../cluster_packages/bridge_state_prior_v1/run_prior.py)、[残差与缓存合同](../cluster_packages/bridge_state_prior_v1/prior_contract.py)、[最初数值验收](../reports/bridge_state_prior_preparation_20261015/numeric_contract.json)、[有效预检独立验收](../reports/bridge_state_prior_preparation_20261015/preflight_audit.json)。

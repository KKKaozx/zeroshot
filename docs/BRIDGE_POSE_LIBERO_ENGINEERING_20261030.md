# 234100新接口：单次排除任务仿真

**已完成并收口：用户取回完整234100后，本机真实执行50次策略调用、200个物理步，任务未完成。** 耗时100.922秒，CUDA最大reserved2.152GiB，无异常、0个原生动作饱和步，0训练更新。初态及全部策略调用结束处均未满足完整目标；末态关抽屉、碗在柜顶两个原生谓词均false。动画首尾图与保存轨迹已查看，50次预测/200tick计数、全调用port+双向mapping、初始未知夹爪[0,0]及1101到1150采样seed一致。没有追加回合/重试/调参。

完整检查点SHA256 `03682dc1760973bcc55b2baee1de2762561129d97c86fe495ad5f30734b1c554`，实际加载通过既有训练报告的base和pose-port身份、冻结编码器/Tokenizer及配方核对。新入口 `diagnostics/run_bridge_pose_libero_engineering_20261030.py` 的真实恢复与物理执行已通过调用层验收；此前四项CPU合成测试与本次真实运行证据分开。

**解除的主实验阻塞：B2中“完整234100当前位姿策略未接入真实LIBERO”的软件子项。** B1源端联合能力、B2空间标定/5Hz接触执行均未解除。这个未标定工程任务失败不能归因某一项，也不能当13模型正式成功率。下一步不再围绕本回合扫映射/控制/采样；需取得缺失物理事实，或另行准备可审阅的源端联合能力修复方案/代码/预算，新增训练仍不得自动提交。

原始报告：`reports/bridge_pose_libero_engineering_20261030/run1/report.json`；固定协议、完整chunks、console.log、execution.gif与首尾PNG均在同目录。下方命令保留用于审阅，**run1已完成，不能再次执行或删除输出绕过拦截**。代码ZIP `cluster_packages/bridge_pose_libero_local_engineering_20261030.zip` 是执行前冻结快照，其中README的“尚未运行”是打包时状态，本页及原始report为执行后的当前结论。

本次直接验证B2软件接口是否能在真实策略和LIBERO中贯通：恢复base及已训练18维当前位姿端口，当前目标EEF→源状态编码，采样后源轨迹→目标执行。不得把本次解释为B1源端能力通过或B2物理标定完成。

固定预算：一个既有预排除LIBERO90任务、demo_0既有初态、环境seed20261026、采样起始seed1101；最多200个20Hz物理步、50次策略调用，包含加载在内600秒子进程硬上限，0更新、0正式评分、0自动重试。保留原5Hz预测/每次4tick执行，不改变控制增益、不追加插值、不重复oracle。遇成功、done、非法输出/异常或预算结束即停。初态恢复后没有状态纠正；不读专家动作或未来观测。

空间约定独立存于 `configs/libero_bridge_pose_mapping_uncalibrated_engineering_20261030.json`：s=1、A=C=I、b和两个TCP偏置为0。这是沿用旧执行候选、经用户同意用于有限工程试运行的假设，绝不是实测事实。绝对位姿输入也受该假设影响，可能落在源端训练分布以外。原REQUIRED空值模板保留，标定事实仍缺。

用户手动下载已有权重（本机PowerShell；无需新集群作业）：

```powershell
scp "zixiao005@10.97.216.128:/projects/Zeroshot/runs/bridge-full-capability-234100/final_policy.pt" "D:\ntu_related\dissertation\Zero_shot\reports\bridge_full_capability_234100\cluster\final_policy.pt"
```

本机检查及唯一正式执行入口：

```powershell
Set-Location "D:\ntu_related\dissertation\Zero_shot"
$py = "D:\ntu_related\dissertation\Zero_shot\training_cache\libero_native_env_v1\Scripts\python.exe"
& $py -B diagnostics/run_bridge_pose_libero_engineering_20261030.py
& $py -B diagnostics/run_bridge_pose_libero_engineering_20261030.py --run
```

若文件在别处，追加 `--checkpoint "实际完整路径"`。实际加载会核对已归档报告的两组参数身份、冻结编码器/Tokenizer身份及源配方，不作远程下载或部分恢复。本次真实加载已执行并通过这些核对。

输出固定为 `reports/bridge_pose_libero_engineering_20261030/run1`，已存在即拒绝再次执行，不能删除该目录绕过停止条件。主进程在超时后终止自己的子进程；失败不自动换参数重试。进度日志命令：

```powershell
Get-Content "D:\ntu_related\dissertation\Zero_shot\reports\bridge_pose_libero_engineering_20261030\run1\console.log" -Tail 40 -Wait
```

正常结束保存report、每次预测和原生动作chunks、动画和首尾图；异常保存错误，硬超时以已落盘记录为准。局部任务是否完成仅作为该工程回合现象记录，不进入40任务正式成功率。不提供上传/Slurm提交命令，因为本次只使用本机既有环境和已完成权重，没有新增远端任务。

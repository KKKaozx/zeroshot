# 完整当前Bridge子池能力实验：代码与操作

2026-10-29用户选择执行此前第二步。本次交付将该选择落实为独立代码包，仍由用户手动上传和提交；助手没有SSH或远端操作。本实验不恢复已关闭分支，不改论文主矩阵，不放行LIBERO。

## 本次固定内容

- 冻结DINOv2 ViT-L/14视觉、CLIP文本；8层Adapter/CLS读出、多尺度U-Net、x0、100步原生DDPM。
- 当前有限池全部10352个Bridge训练窗、1151条演示；不是Bridge公开全量。源坐标相对位姿、名义5Hz/H16、尾部mask和原夹爪联合损失保持。
- 所有可训练参数seed42重新初始化；18→1024位姿端口从零开始；三组AdamW为空。程序不读取任何旧策略/Adam检查点。仅继承冻结编码器、目标仓、评价行和归档后验常数身份。
- batch8，65536更新；固定有放回计划524288次呈现，平均50.646次，实际每窗26–78次、全部见到。Bridge训练统计单独拟合，验证不参与；训练文本固定77token。
- LR2e-4、warmup1024、完整65536步余弦。最终模型固定在65536，不按开发指标选端点；每4096步原子保存完整Adam/调度/计划/RNG。
- 评价128个已有Bridge训练组+64个已有开发组，初始/最终模型各3次采样。不是测全训练池，也不是3个训练seed。各角色四项位姿均值优于不动且轨迹位置配对组区间上界<0才通过本次基本能力标准；不修改旧8窗严格gate。

该实验只能检验当前表示下的单源基本能力。相比历史四源训练，Bridge曝光、从头使用x0/状态、归一化统计等不同，结果不能单独证明混训干扰。

## 预算与保护

预计6.5–7.5小时占卡，单A40 Slurm硬上限8小时，训练循环累计7小时上限。估计基于233818约0.332秒/步；完整Bridge I/O可能不同，不保证上限内完成。预终止信号请求在下一优化边界存断点；初始化或评价阶段仍可能被中断。没有自动续作业/重排队/加seed，job_id.txt不得删除来重提。

至少预留8GiB项目空间。登录节点准入会用项目du与文件系统可用空间作保守检查，但不是权威用户quota；实际quota不足时不要提交。程序不清理既有文件，不建立大特征缓存。原数据、环境、编码器、关键历史模型和新完整状态均保留。

## 下载包与本机上传

小包：`D:\ntu_related\dissertation\Zero_shot\cluster_packages\bridge_full_capability_v1.zip`。不含模型或原图片，包含独立冻结源码、身份/配置、65536批次计划、Bridge统计及192窗数值参考。

本机PowerShell：

```powershell
scp "D:\ntu_related\dissertation\Zero_shot\cluster_packages\bridge_full_capability_v1.zip" zixiao005@10.97.216.128:/projects/Zeroshot/
```

## 集群解压、准入与唯一提交

在你已经登录的集群终端执行：

```bash
cd /projects/Zeroshot
unzip -n bridge_full_capability_v1.zip
cd /projects/Zeroshot/bridge_full_capability_v1
sha256sum -c SHA256SUMS
bash submit_once.sh
```

提交脚本先做CPU准入，再调用一次sbatch。CPU检查包含真实目标仓身份、全部Bridge索引、所有65536批次的来源、状态统计、192评价行/目标/mask及空间。输出`BRIDGE FULL CAPABILITY: 编号`才表示获得作业编号；尚不表示训练已经开始。

若检查失败，保留完整错误，不删除锁或改guard。若job_id.txt中为SUBMITTING，表示sbatch结果可能不明确，先查队列/记账，不能再次提交。

## 查看日志

```bash
JOB=$(cat /projects/Zeroshot/bridge_full_capability_v1/job_id.txt)
squeue -j "$JOB"
tail -n 60 -F "/projects/Zeroshot/logs/bridge-full-capability-${JOB}.out"
```

结束后可查：

```bash
sacct -j "$JOB" --format=JobID,State,ExitCode,Elapsed,NodeList
```

完成应有`BRIDGE_FULL_CAPABILITY_RESULT`、`BRIDGE_SMALL_RESULTS`和`BRIDGE_FULL_CAPABILITY_COMPLETE`。性能gate为false仍然是有效完成结果，不应据此再提交。中断或超时保留断点，但本次脚本没有自动恢复提交入口；需要单独审阅剩余预算。

## 结果保留与取回

输出目录：`/projects/Zeroshot/runs/bridge-full-capability-作业编号`。程序先验证最终策略/完整Adam/实际65536行批次与LR和指标，再生成`small_results.zip`，只装6个小结果文件。最终策略和完整状态留在集群，不加入小结果包。

本机PowerShell，用实际编号替换：

```powershell
$jobId = "替换为实际作业编号"
$resultDir = "D:\ntu_related\dissertation\Zero_shot\reports\bridge_full_capability_$jobId"
New-Item -ItemType Directory -Force -Path $resultDir
scp "zixiao005@10.97.216.128:/projects/Zeroshot/runs/bridge-full-capability-$jobId/small_results.zip" $resultDir
Expand-Archive -LiteralPath "$resultDir\small_results.zip" -DestinationPath "$resultDir\cluster"
& "D:\ntu_related\dissertation\Zero_shot\training_cache\libero_native_env_v1\Scripts\python.exe" -B "D:\ntu_related\dissertation\Zero_shot\deployment\bridge_full_capability_v1\bcap_verify.py" "$resultDir\cluster" --small-only
```

小结果复核会对照包内已绑定的192目标/mask并重算指标，不声称检查了未下载的完整模型/Adam。下载后仍应记录压缩包SHA与集群打印值。

## 本地验证范围

本地新测试使用真实数字标签、原生x0损失/100步采样和CPU替身网络。验证内容及包SHA见`reports/bridge_full_capability_validation_20261029.json`和`reports/bridge_full_capability_build_20261029.json`。未运行实际DINO/CUDA训练、仿真、SSH或集群作业；Shell只做静态LF/shebang检查，不声称Windows已执行bash。

本机复核命令：

```powershell
& "D:\ntu_related\dissertation\Zero_shot\training_cache\libero_native_env_v1\Scripts\python.exe" -B "D:\ntu_related\dissertation\Zero_shot\tests\test_bridge_full_capability.py"
```

以上验证不替代集群真实编码器、设备、身份和资源检查；真实运行结果尚未取得。

# 234100低噪声训练式输入响应：单t0、六次前向、0更新

2026-10-30。现有数组的训练式噪声匹配分析已完成，见`BRIDGE_ROTATION_INPUT_GAP_235002_20261030.md`。本包补齐尚缺的一个网络响应，已准备并通过CPU测试，未提交GPU作业。

## 本次只问一个问题

**同一234100去噪器、同一图像/语言/当前位姿，在t0收到“真实示教＋声明的小噪声”时，是否能改进旋转？与已经保存的原生生成末端响应相比怎样？**

现有235002只含原生生成输入的响应，不能从其统计分布推断另一种输入上的模型输出。本次复用235002的原始t0输出，**不重新生成100步轨迹**。

## 固定设计

- 源：234100完成65536步的`final_policy.pt`，同一参数SHA；CPU准入复用已通过的冻结源检查，并绑定235002三文件SHA。
- 人群：与235002相同的训练前8窗、开发前8窗，不按结果重选。
- 条件：图像、文本、18维当前位姿及Bridge统计完全保留；图文条件只编码16张图一次，保持eval模式。
- 输入：全部7维位姿按原训练前向公式在**t=0**加噪；尾部重复标签与mask按原版本保留，mask仍不输入去噪器。
- 噪声：重建原3采样seed的初始7维高斯，四元数部分须与235002保存的t99初始latent逐值一致。全部六组噪声通过后才运行去噪前向。
- 只新增2个batch8×3噪声的**6次直接去噪器前向**。不调用完整sample，不计算梯度或执行optimizer.step，不计算夹爪头。无新策略/权重端点。
- 配对是同条件、同原始随机来源；该高斯不是原生t0前的最后一次后验随机项，不能称两种输入的噪声逐项相同。
- 这是eval模式下的训练式输入响应，不重演训练dropout，也不是训练loss曲线或新的正式评分。

不同于旧2026-10-27 epsilon/条件置换五t检查：当前为完整Bridge+x0+状态端点，固定t0、条件不换、原生末端输出直接复用；没有新的t扫描。

## 预定读取方式

同时报告训练式带噪输入、其去噪输出、235002已有原生raw t0输出、identity参照的角距/四元数分量误差/范数；按原8组分别计算配对区间。还报告原生mask/批次分母下的eval位姿MSE及位置/旋转数值贡献，不将它们冒称梯度占比。

| 结果模式 | 支持的有限解释 |
|---|---|
| 新输出改善带噪示教，且明显优于已有原生响应 | 该模型在这些低噪声、接近专家的输入上有局部去噪能力；原生路径中已积累的方向偏差未被充分纠正。仍不能锁定条件利用、随机过程或其他因素的唯一因果贡献 |
| 新输出比带噪示教更差 | 正确附近的局部纠偏也不足，应检查监督/拟合/输入条件；不能只说生成输入偏离造成失败 |
| 比原生响应好，但不改善带噪示教 | 两种问题可能同时存在，不能强行归入一个根因 |
| 身份、高斯或保存复算失败 | 工程证据不成立，无新性能结论；停止，不放宽guard或自动重交 |

所有比较仍只有一个训练seed与复用的16组，不放宽234100原联合gate，不允许据结果自动安排旋转权重、采样器或训练变体。

## 资源与验证边界

预计含加载1–2分钟，单A40硬上限5分钟；直接前向循环上限120秒。预留256MiB输出余量，不复制模型/Adam/数据，不清理资产。

CPU三项测试通过：原生diffusion_loss在固定t0/固定噪声下的实际输入与输出逐值等价；按原生100步采样代码生成的CPU随机流验证原初始高斯重建与错误拒绝；真实源标签上的保存数组复核、指标及篡改拒绝。使用CPU替身网络，不是DINO/CUDA性能验证。

原高斯、原native输出、目标/mask、参数和源码身份保持逐值/SHA检查。新前向输入在下载后按CPU公式重建允许1e-6浮点运算差；它不是扩大原生重放容差或修改性能门槛。Shell只做AST以外的文本LF/shebang检查，未执行Bash或SSH。

## ZIP与提交

包`D:\ntu_related\dissertation\Zero_shot\cluster_packages\bridge_low_noise_pair_v1.zip`，8成员10387字节，SHA256为`c594db0725647077ca310d46df012b9a78872b05de8ba69e63a326e0c4910a86`。

本机PowerShell：

```powershell
scp "D:\ntu_related\dissertation\Zero_shot\cluster_packages\bridge_low_noise_pair_v1.zip" zixiao005@10.97.216.128:/projects/Zeroshot/
```

集群已登录终端：

```bash
cd /projects/Zeroshot
unzip -n bridge_low_noise_pair_v1.zip
cd bridge_low_noise_pair_v1
bash submit_once.sh
```

脚本先CPU检查再唯一sbatch。回执为`BRIDGE LOW NOISE PAIR: 编号`；job_id.txt/.submit.lock不能删除绕过保护，SUBMITTING需先核对队列。

```bash
JOB=$(cat /projects/Zeroshot/bridge_low_noise_pair_v1/job_id.txt)
tail -n 80 -F "/projects/Zeroshot/logs/bridge-low-pair-${JOB}.out"
```

正常应有`LOW_NOISE_ORIGINAL_GAUSSIANS_EXACT 6`、六条`LOW_NOISE_T0_FORWARD`、RESULT的0更新/6前向/0完整采样及COMPLETE。没有自动续作业。

## 下载与复核

输出`/projects/Zeroshot/runs/bridge-low-pair-编号/small_results.zip`仅含report.json与paired_inputs_outputs.npz；按日志打印SHA核对。

本机PowerShell：

```powershell
$jobId = "替换为实际作业编号"
$resultDir = "D:\ntu_related\dissertation\Zero_shot\reports\bridge_low_noise_pair_$jobId"
New-Item -ItemType Directory -Force -Path $resultDir
scp "zixiao005@10.97.216.128:/projects/Zeroshot/runs/bridge-low-pair-$jobId/small_results.zip" "$resultDir\small_results.zip"
Get-FileHash -Algorithm SHA256 -LiteralPath "$resultDir\small_results.zip"
Expand-Archive -LiteralPath "$resultDir\small_results.zip" -DestinationPath "$resultDir\cluster"
& "D:\ntu_related\dissertation\Zero_shot\training_cache\libero_native_env_v1\Scripts\python.exe" -B "D:\ntu_related\dissertation\Zero_shot\deployment\bridge_low_noise_pair_v1\verify_low_pair.py" "$resultDir\cluster" --source-trace "D:\ntu_related\dissertation\Zero_shot\reports\bridge_rotation_trace_235002\cluster"
```

完整权重未本地下载，下载复核只验证数组、输入公式和指标；源参数不变身份仍以集群严格验收为支持。

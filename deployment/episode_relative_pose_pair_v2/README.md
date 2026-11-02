# 相对位姿输入v2：移除已缺失的token缓存依赖

2026-10-31。v1文件校验通过，但CPU准入在缺少`runs/bridge-dino-only-cache-v1/full_tokens.pt`处报错。按脚本顺序尚未走到sbatch，没有本次训练或GPU编号。v1原样保留；**v2是独立修订包，未提交。**

## 修复方式及明确的配方调整

原229116全部1876窗的评估上下文已在既有结果中保存并用于235928。本包提供约7.16MB的`adapter_contexts.npz`，原1836train/40dev顺序和数值不变；无需重建数GB视觉token缓存，不再读取原图像。

令`c_old`为原上下文，`f_abs`为原9D绝对状态，`W0`为229116原状态投影，`W`为候选可训练状态投影：

```text
c = c_old + (W @ f_candidate - W0 @ f_abs)
absolute: f_candidate = f_abs
relative: f_candidate = 初态相对9D状态
```

原上下文含冻结Adapter及原状态贡献；评估时history同帧差为零。差分形式在数学上移去旧状态贡献、加入新状态。控制组初始化`W=W0`时先相减得到严格零，再加原上下文，避免拆开上下文的抵消舍入误差。后续新状态的实数等价不承诺不同设备训练/在线浮点逐位相同。

**相对v1缩小训练范围：两组均冻结Adapter和history，仅训练扩散后端与状态端口；两组都不再使用Adapter训练态dropout。** 两组之间仍只改变状态表示。保留已有前端能力，不能把本轮称为Adapter也联合微调的v1结果。原完整源文件、235928头只读保护；编码器/Adapter/history/夹爪更新均为0。候选仍保存全部四模块，并验证冻结两模块与源参数逐元素相同。

相对输入为`R0.T @ (p_t-p0) / old_xyz_std`和`R0.T @ R_t`前两列。训练初态取演示index0，当前取原start；在线须先`policy.begin_episode(observation)`，参考固定到回合结束。原工具相对动作、标签、名义时距和100步DDPM不变。旧权重不能改个输入名称冒充已训练相对模型。在线加载器要求v2格式，用实时图像/语言计算冻结前端，不用训练上下文代替在线观测。

## 固定预算与停止条件

- 同229116第8192步权重，空Adam、同批次/RNG，各2048×8，总4096；常LR1e-5、wd0、clip1，仅更新`diffusion_decoder.*`及`state_projection.*`。不是原Adam精确续训。
- 单A40预计15–30分钟、Slurm硬60分钟；CPU准入后3300秒alarm覆盖配对主体与验证打包。预计2–4GiB显存、24GiB主机内存；仍保留8GiB输出余量，不自动清理。估计未经本版真实GPU运行验证。
- 每512保存完整模块/Adam/RNG；原文件前后SHA一致，冻结Adapter/history与源逐元素一致。超时/非有限/身份或恢复守卫失败就结束，不续训、改阈值/容差、换seed、重试或自动仿真。
- 原1836train/405演示训练，固定128训练锚点和40dev/10演示各3draw评价；原基线预测直接复用。初始2窗/1draw只验证上下文和原生采样恢复；容差仍为上下文1e-5、轨迹分量1e-4，失败不放宽。
- relative的train/dev按演示平均的位置和旋转误差均不超过原基线1.05倍，开发两项也不超过同期absolute的1.05倍，才供审阅一次有限工程回合；不自动运行、不算正式准入。失败保留结果结束本候选。旧夹爪门槛不改，不追加夹爪训练。

本轮只验证冻结有效前端后的相对状态是否保住源位姿能力。世界坐标不变性不解决工具轴/TCP、相机、接触时序或任务泛化；新增因果初态参考并非证明唯一根因。旧夹爪与新上下文兼容性仍未知。13模型及大评估未放行。

## 本地验证边界

6项CPU检查通过，含旧token缓存缺失时的准入、控制初始化上下文逐位保持、两组各2次合成更新/Adam回读、冻结模块梯度/权重保护、相对状态刚性不变性及旧格式拒绝。不是策略成绩。原生核心及真实四模块CPU恢复证据沿用v1，没有重复旧能力实验。本轮0真实训练/完整模型采样、0图像解码/编码、0新仿真、0SSH或提交。实际A40/Torch2.5.1+cu118更新仍待该唯一作业。

训练只依赖两个已有权重、本包及原环境：

```text
/projects/Zeroshot/runs/bridge-backbone-pair-a40-229116/dino/final_trainable.pt
/projects/Zeroshot/runs/dino-gripper-sidecar-235928/gripper.pt
/projects/Zeroshot/envs/multitask-preflight-v1
```

不依赖旧`full_tokens.pt`，不需再上传原图像/编码器/源模型。后续在线加载才复用原`dino_gripper_sidecar_v1`和视觉/文本资产，不自动执行在线回合。

## 审阅后手动执行

本机PowerShell：

```powershell
scp "D:\ntu_related\dissertation\Zero_shot\cluster_packages\episode_relative_pose_pair_v2_20261031.zip" "zixiao005@10.97.216.128:/projects/Zeroshot/"
```

集群使用新目录，不改旧v1：

```bash
cd /projects/Zeroshot
test ! -e episode_relative_pose_pair_v2
unzip -n episode_relative_pose_pair_v2_20261031.zip
cd episode_relative_pose_pair_v2
sha256sum -c SHA256SUMS
bash submit_once.sh
```

```bash
JOB=$(cat /projects/Zeroshot/episode_relative_pose_pair_v2/job_id.txt)
tail -n 60 -F /projects/Zeroshot/logs/episode-relative-pair-v2-${JOB}.out
```

只做CPU依赖核对、不申请GPU：

```bash
/projects/Zeroshot/envs/multitask-preflight-v1/bin/python -B /projects/Zeroshot/episode_relative_pose_pair_v2/train_pair.py --root /projects/Zeroshot --check-only
```

不删除锁/job_id/结果来绕过重复保护。完成后只取小报告与原预测，JOB替换真实编号：

```powershell
scp "zixiao005@10.97.216.128:/projects/Zeroshot/runs/episode-relative-pair-v2-JOB/small_results.zip" "D:\ntu_related\dissertation\Zero_shot\reports\episode_relative_pair_results_20261031\small_results_v2.zip"
```

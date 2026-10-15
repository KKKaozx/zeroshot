# 真实张量的官方 DDPM 更新对照

不重训、不安装软件、不下载权重。复用 193761 final.pt、原有 40 条训练演示每条第一个窗口、seed=0。

1. multitask-preflight-v1 环境在 GPU 调用原生 model.sample，记录全部 100 个时间步的真实 x_t、decoder x0 输出和实际高斯噪声。记录时原样返回所有噪声；逐批无记录重放必须输出和 CUDA RNG 逐元素一致。
2. 进程退出释放模型后，official-pusht-v1 环境在 CPU 调用已经安装的 diffusers==0.11.1 DDPMScheduler.step，用同一 x_t、x0 和噪声比较更新。

官方源代码：https://github.com/huggingface/diffusers/blob/v0.11.1/src/diffusers/schedulers/scheduling_ddpm.py

官方 scheduler 使用 prediction_type=sample、fixed_small 和相同 beta，clip_sample=false；在调用前按 Bridge 规则裁剪 x0 的 xyz 到 ±3、四元数分量到 ±1。这样控制项目已有动作范围，而非复制 Push-T 的 epsilon 和全部分量 ±1。还单独比较官方独立计算的 cosine beta。官方 step 方法未修改，仅注入已记录的随机噪声。

对照每个 t 的真实下一步 latent（t>0），t0 按已有最终位置裁剪/四元数单位化后比较 final pose。报告还比较每步均值、方差、x0→epsilon→x0 的浮点差异，记录 scheduler 源文件哈希与版本。

判读：如实际更新在 atol=rtol=1e-5 内一致，降低这一条链上采样实现错误的优先级；如不一致，先检查首次差异及数量级，区分 CUDA/CPU 浮点误差与实质差异。数值容差不是机器人成功阈值。报告执行通过与实现一致是两个独立字段。

限制：每步重置为项目真实输入，未进行官方 scheduler 的自由运行闭环，不宣称两个完整策略相同；一个采样 seed、40 个训练窗口；未评估开发或预留测试动作。不能证明模型条件能力、训练拟合或泛化正确。

资源：新增约 7 MB 的未压缩数组（保存为压缩 npz）和小 JSON；不创建 checkpoint。作业时间上限 15 分钟，不含排队，也不是运行时间承诺。无需下载张量到本地，只下载最后 report.json。

本地仅执行过噪声捕获/重放/异常清理测试、核心代码哈希检查与 Python/Bash 语法检查。本地未安装 diffusers；官方 scheduler 的导入和执行会在集群已有环境先检查，失败就停止，不自动安装依赖。

## 操作

本地 PowerShell：
```powershell
scp "D:/ntu_related/dissertation/GPU cluster/bridge_scheduler_compare_v1.zip" "D:/ntu_related/dissertation/GPU cluster/bridge_scheduler_compare_v1.sha256" zixiao005@10.97.216.128:/projects/Zeroshot/
```

集群：
```bash
cd /projects/Zeroshot
mkdir -p logs
sha256sum -c bridge_scheduler_compare_v1.sha256 && unzip -n bridge_scheduler_compare_v1.zip
SCHED_JOB=$(sbatch --parsable /projects/Zeroshot/bridge_scheduler_compare_v1/scheduler_compare.sh)
echo "采样器对照：$SCHED_JOB"
sacct -j "$SCHED_JOB" --format=JobID,State,ExitCode,Elapsed,NodeList
tail -n 8 "/projects/Zeroshot/logs/bridge-scheduler-$SCHED_JOB.out"
```

报告：`/projects/Zeroshot/runs/bridge-scheduler-JOBID/report.json`。重连后将变量替换为实际编号。

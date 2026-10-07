# 固定latest权重的条件依赖检查

目的：区分带噪真实目标的一步还原与从纯噪声生成动作，并观察图像/语言条件是否影响预测。
复用190189的latest、其归档生产模型/加载器及原数据，不创建优化器、不反传、不更新参数。
这不是训练、机器人任务成功测试，也不能单独定位Adapter或DDPM实现错误。

按源分片、记录、窗口起点顺序，每任务前8条训练演示各取第一个完整窗口，共40训练窗口；
全部10条开发演示的40窗口保留。只检查位姿；不重复整套夹爪评价。
选择不参考误差，保留测试目标不读取。训练子集结果不能代表全部1836训练窗口。

三组条件为正确图文、同任务另一演示的图像替换、另一任务的语言替换；
替换来源保持在同一训练/开发分区，当前夹爪输入和真实目标不变。
替换映射明确记录，开发窗口数不均衡，因此图像替换不是严格的一一排列。
检查图像像素差、语言是否确实改变，以及上下文/输出变化，防止无效对照。

每组使用配对的0/1/2噪声种子，执行：

- t=0/24/49/74/99：真实7维位姿按生产alpha_bar加噪后，一次前向直接预测x0。
- pure_noise_t99：同一噪声，完全移除真实目标信号，只做t99的一次x0预测。
- native_sample：直接调用归档生产model.sample，执行原100步完整生成。

报告每个条件/种子/任务的物理位置与旋转误差、原始7维分量MSE、
相对于正确条件的配对输出变化及自身静止基线。一步还原位置保持原始输出，
完整采样沿用生产输出裁剪与四元数归一化；原始MSE与角度误差含义不同。
参数摘要在推理前后必须一致，且所有参数无梯度，否则作业失败。

解释边界：低噪声还原好不代表从图文独立预测好，输入中保留了目标信息。
条件替换使结果改变只证明敏感性；只有正确条件更准确才支持条件有用。
高噪声表现差不能单独证明语言/图像被忽略；替换本身也可能超出训练分布。
一步高噪声较好、完整生成较差时再核对采样链路，仍不能直接认定采样公式有bug。

本地已完成真实80窗口复读、训练/开发隔离、替换来源检查、Python/Bash语法检查。
本地没有加载CLIP或checkpoint，GPU推理尚未执行。
只导出probe脚本、Slurm脚本、准备清单与哈希，不复制数据、环境或权重。

在本地PowerShell上传：

```powershell
scp "D:/ntu_related/dissertation/Zero_shot/training_cache/exports/bridge_conditioning_probe_v1.zip" "D:/ntu_related/dissertation/Zero_shot/training_cache/exports/bridge_conditioning_probe_v1.sha256" zixiao005@10.97.216.128:/projects/Zeroshot/
```

在集群校验、解压并提交：

```bash
cd /projects/Zeroshot
sha256sum -c bridge_conditioning_probe_v1.sha256 && unzip -n bridge_conditioning_probe_v1.zip
```

上一步成功后：

```bash
PROBE_JOB=$(sbatch --parsable bridge_conditioning_probe_v1/bridge_conditioning_probe.sh)
echo "检查作业：$PROBE_JOB"
squeue --me
```

日志：/projects/Zeroshot/logs/bridge-condition-作业编号.out；
结果：/projects/Zeroshot/runs/bridge-conditioning-作业编号.json。
最终只取回JSON即可，不下载权重或全部预测数组。

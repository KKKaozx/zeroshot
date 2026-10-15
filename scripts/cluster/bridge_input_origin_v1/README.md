# 冻结 DDPM 输入来源对照

目的：在同一模型、同一时间步、同一观察条件下，直接比较真实动作前向加噪输入与模型反向采样输入的 x0 估计误差。这是训练配方内的故障隔离，不是新的论文复现，也不改变官方/项目模型进行训练。

复用旧作业 193761 final.pt，避免将 195649 的重放差异混入之前的 194978/195008 噪声探针。所有核心代码、权重、窗口参考均核对原始哈希。选择原有 40 条训练演示（每任务 8 条，各取第一个窗口），采样种子 0/1/2，时间步 99/74/49/24/9/0。未读取开发或预留测试图像与目标。

链输入直接从原生 model.sample 的 decoder hook 取得，未重写反向公式。每批无 hook 重放并核对逐元素输出和 CUDA RNG 状态。真实动作仅用于随后独立生成前向加噪输入与评分，从不进入原生生成。前向噪声使用独立 generator，避免改变原生采样随机流。

输出：整体、逐任务、逐窗口的预测干净动作物理误差；原始 xyz/quaternion MSE；输入 xyz 均值/标准差、四元数范数；输出裁剪比例和真实目标超界比例；参数与 checkpoint 不变检查。物理误差只评分预测 x0 按已有规则裁剪和单位化后的结果，不把含噪 latent 当可执行位姿。

## 判读

- 若低噪声的真实加噪输入也还原不好，去噪器对训练式输入的还原仍有不足；不能只归因于反向采样输入。
- 若前向加噪输入较好，但同 t 的链输入明显更差，则提高输入分布偏移/链误差累积的排查优先级；尚不能分辨具体由损失、动作尺度、条件信息或采样设计造成。
- 若两种输入的物理误差相近，输入来源假说解释力有限，应停止重复同类扫描。
- 若目标超界，计算其覆盖范围与影响后再考虑改裁剪/归一化；不能只凭存在超界就认定根因。

注意：前向输入含真实未来动作，是还原诊断，不是部署策略；两类输入没有相同 latent，因此只控制时间步和观察，不能宣称所有因素相同。三个采样种子非独立训练重复，40 窗口不是完整训练集。匹配演示误差不能判断替代有效路线或机器人任务成功。任何输出 PASSED 只表示执行检查通过。

## 运行

无需安装、下载数据或权重，也不保存新 checkpoint。结果预计数百 KB，作业预算 20 分钟（不是排队或运行时间承诺）。

本地 PowerShell：
```powershell
scp "D:/ntu_related/dissertation/GPU cluster/bridge_input_origin_v1.zip" "D:/ntu_related/dissertation/GPU cluster/bridge_input_origin_v1.sha256" zixiao005@10.97.216.128:/projects/Zeroshot/
```

集群 Bash：
```bash
cd /projects/Zeroshot
mkdir -p logs
sha256sum -c bridge_input_origin_v1.sha256 && unzip -n bridge_input_origin_v1.zip
ORIGIN_JOB=$(sbatch --parsable /projects/Zeroshot/bridge_input_origin_v1/input_origin.sh)
echo "输入对照：$ORIGIN_JOB"
sacct -j "$ORIGIN_JOB" --format=JobID,State,ExitCode,Elapsed,NodeList
tail -n 10 "/projects/Zeroshot/logs/bridge-input-origin-$ORIGIN_JOB.out"
```

完成后只下载 `/projects/Zeroshot/runs/bridge-input-origin-JOBID.json`；重连后需要将变量替换成实际编号。

# 项目202278：冻结融合条件→扩散动作头诊断

状态：204017完成且两个小文件已独立复算通过。原40窗三次预测重放差0；固定9对留出条件首次估计10.056cm，末次10.197cm，9/9窗首次错误已达最终90%以上；训练条件各阶段窗口平均位置误差最大4.49mm。没有新的训练或泛化修复。[逐窗/演示核验、图与解释边界](BRIDGE_PROJECT_CONDITION_RESULT_204017.md)。下文未运行为准备时状态。

冻结ZIP为13,298字节、7个文件，SHA256：`d669642696be164dbb5e8964190a4596c0bb7f7fccf621021f1d76889a74dce7`。包内及ZIP逐文件复读校验通过。现有约505MiB检查点保留在集群，本地不增加权重副本。

## 这次回答什么

作者203876已确认：在固定的一对有效观察上，专家轨迹差16.49cm，作者输出仅差1.00cm。这是作者模型的局部结果，不能直接解释项目模型。项目202278对应的两份融合条件并不相同，因此这次检查**不同条件进入自己的原生扩散动作头后，轨迹估计有什么变化，错误是否在后续生成链中扩大**。

使用已有202278真实状态组的最终权重及保存的1024维融合条件。条件已包含Adapter和真实当前状态贡献，不再次运行CLIP、Adapter或状态投影；重建模型时仅从集群现有CLIP资产恢复原冻结参数，以验收初始化及完整原预测。诊断检查融合条件边界，不能分别评价图像、语言、状态端口，更不能直接证明某一层坏了。

## 固定协议与停止条件

- 权重：`runs/bridge-state-fit-202278/real/reduced_u8192_trainable.pt`，SHA256为`3cc532296cd1991665d411b3e885ed1f4f4b9eff8b1706b46ac4d84645928e36`；初始化43，8192次既有更新，本次更新数0。
- 先按原顺序、原batch=2、原采样种子1101/1102/1103重放全部40个开发窗口，包括原来的8维输出。最大原始分量差必须≤1e-4，否则停止。XYZ该阈值对应每轴0.001cm。
- 通过全部重放后，固定9对西兰花观察，沿用已完成作者观察审计的图像近邻配对。索引：138→111、139→111、140→54、141→111、142→113、143→106、144→29、145→115、146→114。开发观察和训练观察各自使用自己的目标，禁止把训练目标注入模型。
- 条件替换仅替换选中行的完整融合条件及原测量夹爪输入，不拼接不一致的图像/状态。夹爪输入是为了完整重放；本诊断只评价7维位姿，冻结夹爪头的输出不作为抓取能力证据。
- 两次采样恢复同一CPU/CUDA随机数状态，验证初始噪声完全相同及结束随机数状态完全相同；原生100步随机DDPM不变。
- 在真实decoder前向调用的99、74、49、24、0步记录原始x0读出与原生浮点转换、裁剪后的干净轨迹估计。记录钩子不替换输出，不消耗随机数；带/不带钩子的最终输出要求逐元素完全相同。
- 复核未选中batch行、所有参数与缓冲区不变；源码、检查点、缓存均验收哈希。150次原生整链调用，没有优化器、新权重文件、依赖安装、数据或模型下载。

每步的x0是“当前对完整16目标干净轨迹的估计”，不是第几次真实机器人运动。姿态以单位四元数计算；近零四元数使用原生产输出的单位姿态回退并记录数量。位置仍沿用原标签0.1m一单位。终点是第16目标，没有把这些到达位姿当增量积分。

## 如何判断结果

记录9对的原条件误差、训练条件误差、配对输出变化及专家轨迹差，同时保留逐窗、逐种子的NPZ供独立复算。

如果最初99步就对训练条件较准、对留出条件较差，后续维持该差距，证据偏向新观察条件下的动作读出不足。若初期估计较好而反向链后续明显恶化，则证据偏向生成链输入逐步偏离及误差累积。两者可以同时出现。没有任何一种结果能单独确定“唯一U-Net错误”或“Adapter正确”。

换到训练观察且换到其训练真值后误差变小，属于条件响应对照，不是修复留出泛化。9对是事后固定诊断样本，40开发窗口也已多次使用；两者都不是未见测试集。与一条专家轨迹的误差不能代替实际机器人任务成功率。

## 本地已检查的范围

CPU检查调用原冻结源码的`RobotAdapterModel.sample`及实际多尺度decoder的缩小随机参数实例（通道16/32），验收钩子与随机数不变、配对初始噪声、未替换行不变、参数/缓冲区不变、XYZ单位与四元数符号、168个原测量夹爪缓存及固定索引。Bash启动语法检查通过。

这些检查验证诊断工具本身。**本地没有加载真实检查点或CLIP，没有执行集群GPU重放。**真实检查点需在下述作业完成后再核验。

## 上传与提交

本地PowerShell运行（文件放在项目目录，避免在兄弟目录增加副本）：

```powershell
scp "D:/ntu_related/dissertation/Zero_shot/cluster_packages/bridge_project_condition_probe_v1.zip" "D:/ntu_related/dissertation/Zero_shot/cluster_packages/bridge_project_condition_probe_v1.sha256" zixiao005@10.97.216.128:/projects/Zeroshot/
```

集群Bash运行：

```bash
cd /projects/Zeroshot
mkdir -p logs
sha256sum -c bridge_project_condition_probe_v1.sha256 && unzip -n bridge_project_condition_probe_v1.zip && PROJECT_CONDITION_JOB=$(sbatch --parsable bridge_project_condition_probe_v1/probe.sh)
echo "项目条件诊断作业：$PROJECT_CONDITION_JOB"
squeue --start -j "$PROJECT_CONDITION_JOB"
```

默认一张已有兼容验证的A6000，上限15分钟；这不是排队时间预测。复用`envs/multitask-preflight-v1`，小包小结果，不下载检查点到本地。若排队，只查当前资源；不同时提交多份正式诊断。

查看状态与日志：

```bash
sacct -j "$PROJECT_CONDITION_JOB" --format=JobID,State,ExitCode,Elapsed,NodeList
tail -n 15 "/projects/Zeroshot/logs/bridge-project-condition-$PROJECT_CONDITION_JOB.out"
```

断连后将变量重新设为实际编号，不能把`JOBID`照抄成文字。失败时先看日志，不重新训练或跳过重放断言。

完成后仅下载两个小结果：

```bash
cd /projects/Zeroshot/runs
tar -czf "/projects/Zeroshot/bridge_project_condition_results_$PROJECT_CONDITION_JOB.tar.gz" "bridge-project-condition-$PROJECT_CONDITION_JOB/report.json" "bridge-project-condition-$PROJECT_CONDITION_JOB/predictions.npz"
```

本地PowerShell把下面`实际编号`替换为收到的作业编号：

```powershell
scp zixiao005@10.97.216.128:/projects/Zeroshot/bridge_project_condition_results_实际编号.tar.gz "D:/ntu_related/dissertation/Zero_shot/reports/"
tar -xzf "D:/ntu_related/dissertation/Zero_shot/reports/bridge_project_condition_results_实际编号.tar.gz" -C "D:/ntu_related/dissertation/Zero_shot/reports/"
```

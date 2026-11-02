# Bridge控制增量8维：单一源端训练候选

日期2026-10-31。状态：完整代码及材料已准备，尚未提交或训练，也未执行真实模型推理/物理仿真。目标仍是OXE训练→LIBERO零样本评估。只使用Bridge源数据和234100源权重；不使用LIBERO示教、236840监督权重或目标域拟合统计。本候选不计入老师五条件/13模型成果。

## 本次具体改什么

旧8维目标是未来实际到达位姿。本候选直接学习记录中的控制增量：`[dx/.1,dy/.1,dz/.1,delta_qx,delta_qy,delta_qz,delta_qw,binary_open]`，外部仍8维，明确使用新格式`bridge_controller_increment_pose8_candidate_v1`。原始Bridge动作7维中的Euler三维转换为四元数；模型预测8维，再经几何反馈转换为LIBERO原生7维。

公开采集器族的更新规则为`D=Rz(dyaw) Ry(dpitch) Rx(droll)`、`p_new=c+d+D(p_previous_goal-c)`、`R_new=D R_previous_goal`。`c`是当前实测EEF支点，previous_goal是前一期望目标。在线从当前实测初始化目标一次，以后保留目标历史；每条名义5Hz命令只更新一次目标，之后4个20Hz原生tick反馈跟踪同一目标，不把同一增量叠加4次。`online_policy.py`已经接好新端点加载、源状态特征、持续目标更新及原生反馈；旧未来位姿端点不能直接装作新控制增量执行。

直接监督增量不需要恢复源端缺失的desired_state。本实现没有伪造源端绝对期望目标或规定任务动作轨迹；未来物理回合的动作必须由新训练策略预测。

## 必须随结果保留的假设

公开采集/转换族支持这个控制定义，但本地缓存0.0.1与公开RLDS 1.0.0的精确导出/对齐/控制器配置对应仍未恢复。此候选采用明确工程假设：obs[t]对应action[t]，XYZ按米理解，名义5Hz；先对原运动命令逐分量裁剪XYZ±.05、Euler±.25。没有新测量到时间、物理尺度或延迟，不额外平移图像/命令行。

源目标工作空间裁剪配置未知；本候选目标域不加工作空间裁剪，原生反馈有幅度边界。源/目标坐标轴、单位、控制点对应仍需显式传入映射，零偏移同点约定未经物理标定；保留原18D绝对当前状态及源训练统计也仍可能跨域失配。缺历史commit不再成为无限前置，但不能把假设写成已证事实。假设依据与固定单记录数学对照见项目`docs/BRIDGE_COMMAND8_DEFINITION_20261031.md`及`reports/bridge_command_target_definition_20261031/`。

公开来源：

- [采集器](https://github.com/rail-berkeley/bridge_data_robot/blob/main/widowx_envs/widowx_envs/base/robot_base_env.py)
- [旋转/变换](https://github.com/rail-berkeley/bridge_data_robot/blob/main/widowx_envs/widowx_envs/utils/transformation_utils.py)
- [raw→numpy转换](https://github.com/rail-berkeley/bridge_data_v2/blob/main/data_processing/bridgedata_raw_to_numpy.py)
- [RLDS builder](https://github.com/kpertsch/bridge_rlds_builder/blob/main/bridge_dataset/bridge_dataset_dataset_builder.py)

## 数据与初始化

沿用234100的10352个Bridge训练窗/1151训练演示；开发64个不同演示各1窗，与训练组不交叉。新H16标签从当前行开始，保守排除每记录最后一行，尾部mask，不使用未来实测位姿标签。源端二值夹爪仅原始exact0/1转-1/+1；中间连续值masked，不能当Panda速度。本次读取1215条已有记录的数值字段，0图像解码/编码器调用；裁剪计数训练846、开发25个运动分量，中间夹爪训练879、开发40行（全记录计数，非窗口累计次数）。仅训练覆盖的39572条唯一命令拟合均值基线，重复窗口不重复计数。

234100完整权重只作warm start：`runs/bridge-full-capability-234100/final_policy.pt`，SHA256 `03682dc1760973bcc55b2baee1de2762561129d97c86fe495ad5f30734b1c554`。恢复全部base和18D pose_port，原文件只读；新空三组AdamW，绝非恢复旧Adam继续旧目标。冻结DINOv2-L/14及CLIP文本，8层Adapter、多尺度x0扩散及夹爪/状态端口沿用。只改变动作标签/执行合同及相应平移采样边界（归一化±.5）；无新Min-SNR或夹爪支线。旧constructor配置仅作架构/冻结资产恢复模板，不能把它当新标签定义。

本包独立图文reader直接读绑定的Bridge原记录、当前image_0/指令；当前命令历史和18D状态在包内绑定。不需要旧full_tokens.pt、旧未来位姿目标仓或下载/重编码回退。运行依赖仅已有Bridge原文件、234100完整权重、原冻结编码器资产、原项目环境。本包的vendor是原bridge_full_capability_v1字节副本，不执行其历史提交/训练入口。

## 固定预算、评估与停止

一个模型、seed42，8192次更新×batch8、恒定LR1e-5、weight decay .0001，原uniform-t masked x0损失+0.25夹爪BCE。预先保存全部采样计划。单A40、4CPU、40GB RAM；预计50–65分钟，Slurm硬90分钟，CPU准入后进程5100秒（含构造、守卫、训练、生成、导出及验证打包），训练3900秒。要求8GiB输出/原子落盘余量；du仅项目占用估计，不代表权威用户quota，不自动清理。

固定原128训练组+64开发组，每组1窗、每窗3个采样draw（1101/1102/1103；不是3训练seed）。**主门槛**：两角色的第一条命令位置/旋转均值均优于零命令及训练均值命令，同时相对零命令的两项配对组bootstrap95%区间上界均<0。H16全段/终点命令误差、相对均值的区间及夹爪命令准确率为次指标。只选最终8192端点，不按开发集选最好权重。指标是控制增量误差，不能与旧未来到达位姿误差作改善百分比，也不是任务成功率；本次多项合同改变，不是单变量因果对照。

每1024步原子保存latest_state.pt，含完整模型/Adam/日程/计划/RNG；最终另存policy.pt，读回全状态并复算保存预测的标签/指标/门槛。错误/到预算立即停止，保留最后完整原子状态，不自动恢复、重提、加步或扫loss/种子/池/控制器。恢复原语已有CPU恢复检查，但生产入口没有自动续训选项。通过仅表示值得审阅下一次工程回合，不自动运行、放行正式矩阵或宣称零样本迁移成功。失败本候选结束。

如将来单独授权，已备`evaluate_local.py`：默认仅readiness，无--run不启动；仅1个预排除复合任务demo0初态、200tick/50调用/600秒，初态恢复一次，0专家动作/中途重置，产GIF/首末帧/命令目标及原生trace。未过源门槛须另行明确未准入工程授权，原gate仍false。本轮没有执行它。

## 本地检查范围

7项CPU检查通过：编码/裁剪/夹爪mask/尾部、完整绑定材料与训练均值、原生损失收到新目标及三组Adam/RNG恢复、严格新端点加载与旧格式拒绝、合成环境中目标历史持续且不重复累加、保存指标/首命令门槛、合成TFRecord中当前图文/命令行相位且无旧标签仓。使用TinyModel及合成环境，未进行真实DINO策略训练或物理仿真。首次CPU替身测试停滞后已中断，限制线程并使用项目内临时目录后最终7项通过；不当作集群策略故障或能力改善。

本地依赖导入检查无CUDA初始化/模型构造；本机Torch2.11+cu128、TF2.21与远端目标Torch2.5.1+cu118、TF2.15不同，远端完整材料/环境/空间准入须由下述手动入口实际执行，尚未声称通过。Shell仅静态LF/shebang检查，未在Windows声称执行bash -n。

## 用户审阅预算后手动操作

本机PowerShell上传小代码/数值材料包，不上传权重或原数据副本：

```powershell
scp "D:\ntu_related\dissertation\Zero_shot\cluster_packages\bridge_command8_source_v1_20261031.zip" zixiao005@10.97.216.128:/projects/Zeroshot/
```

集群终端（目录若已存在，先保留并检查；不要覆盖旧提交目录）：

```bash
set -e
cd /projects/Zeroshot
test ! -e bridge_command8_source_v1
unzip -n bridge_command8_source_v1_20261031.zip
cd /projects/Zeroshot/bridge_command8_source_v1
bash submit_once.sh
```

submit_once先完整CPU准入，再唯一一次sbatch；提交前写标记，返回不明确时保留标记查队列，不直接重提。它打印`BRIDGE COMMAND8 SOURCE: 编号`。查同一日志（填实际编号）：

```bash
JOB=实际编号
tail -n 80 -F /projects/Zeroshot/logs/bridge-command8-source-${JOB}.out
```

完成标记`BRIDGE_COMMAND8_SOURCE_COMPLETE`在最终验证/ZIP之后；先取原small_results.zip，不自动取大状态或启动仿真。本机PowerShell：

```powershell
$job = "实际编号"
$dest = "D:\ntu_related\dissertation\Zero_shot\reports\bridge_command8_source_$job\cluster"
New-Item -ItemType Directory -Force -Path $dest | Out-Null
scp "zixiao005@10.97.216.128:/projects/Zeroshot/runs/bridge-command8-source-$job/small_results.zip" $dest
```

保留集群policy.pt及latest_state.pt；后续按原结果审阅，当前不清理、不自动提交任何后续任务。

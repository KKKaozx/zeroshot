# Bridge新初始化与运动尺度候选

2026-10-31。完整本地候选，未提交、未训练、未启动真实模型推理或物理仿真。OXE→LIBERO零样本目标保持。237424已关闭，不恢复其训练或改写其结果；本候选不计老师五条件/13模型进度。

## 具体修复及依据

237424把234100旧未来到达位姿的全部学习权重迁入新控制增量任务，再以1e-5微调8192步；训练/开发的六项运动误差均值都未优于零命令及训练均值基线。它没有建立源端基本能力。236840独立监督5/5参照采用新初始化与直接原生命令，但数据、状态和任务复杂度也不同，不能据此归因某一个因素。本候选不使用236840权重、LIBERO示教或目标域统计。

这次一起处理新任务的初始化与训练坐标：

1. 冻结预训练DINOv2-L/14及CLIP文本；8层Adapter、扩散运动后端、夹爪头全部fresh seed42，18D当前状态端口为零初始化。不加载234100、237424或任何旧策略学习权重，运行不要求这些checkpoint存在。
2. 仅用Bridge训练覆盖的39572条唯一命令拟合7个运动分量均值/标准差；窗口重复不重复计数，开发完全不参与。内部`z=(command7-mean)/scale`，std_floor=.001，本次七维原std均大于floor。损失仍为uniform-t masked x0 MSE+0.25二值夹爪BCE，不重开Min-SNR或夹爪事件实验。
3. DDPM完整100步均在标准化坐标内运行；需要clip clean时先逆变换到原命令坐标裁剪，再标准化回去。最终逆变换、单位化四元数，夹爪头的轨迹条件也使用训练时相同标准化坐标。不能直接把旧XYZ±.5/四元数±1边界套在标准化值上。
4. 新训练固定LR1e-4、三组新空AdamW/weight decay .0001，batch8、8192步。初始化、尺度和学习率共同改变，明确是整体修复候选，不是单变量因果对照，不能据结果宣布某个因素是唯一根因。

已有训练数值表明七个原运动分量std约为[.10412,.14396,.12646,.01504,.01591,.02584,.001179]，范围约122倍。这是不同坐标尺度的事实，不是已证明的性能根因。`motion_normalizer.json`绑定精确数值、训练唯一命令键摘要及原输入SHA；只新增一个源训练坐标变换，不生成新示教或伪造desired_state。

## 不变的外部动作定义

模型外部仍输出H16×8：`[dx/.1,dy/.1,dz/.1,delta_qx,delta_qy,delta_qz,delta_qw,binary_open]`。7维只是内部连续运动部分，另有夹爪1维；未变成原生7维模型。外部格式仍`bridge_controller_increment_pose8_candidate_v1`，新端点格式`bridge-command8-fresh-scaled-policy-v1`。不按新格式解释旧策略端点。

沿用237424同一10352训练窗/1151演示、64开发组、训练计划、18D当前状态及源训练状态统计、当前图文与历史命令相位、源标签/尾部mask。原包的inputs.npz/rows.json/batches.npy字节不变；无新增raw记录、图像或编码器调用。训练时保留`command_actions`原始命令供评估，只有loss的运动坐标标准化；报告仍按原控制命令误差评估，不能用标准化后loss替代策略能力。

在线执行仍每条名义5Hz命令只更新一次期望目标，4个20Hz原生tick反馈跟踪同一目标，保留目标历史。`p_new=c+d+D(p_previous_goal-c)`、`R_new=D R_previous_goal`，源命令裁剪XYZ±.05/Euler±.25；raw0/1夹爪映射-1/+1，中间值masked。`online_policy.py`已经接入新端点的严格加载、同一normalizer、完整模型两组恢复和相同执行器；`evaluate_local.py`仍默认readiness。

## 假设与证据边界

公开采集族与本地缓存0.0.1的精确导出/对齐尚未绑定；名义5Hz、raw XYZ按米、源裁剪及跨机器人控制点/轴/单位映射仍是明示工程假设，并非新物理标定。原18D绝对状态的跨域适用性未被证明。初始化与尺度修复不解决这些事实缺口，也不授权LIBERO监督适配。公开源证据与数值原型保留在项目docs/BRIDGE_COMMAND8_DEFINITION_20261031.md及reports/bridge_command_target_definition_20261031/，不重复历史查证。

11项CPU替身/合成检查包括：绑定源数据/相位/mask、标准化坐标进入真实原生loss及Adam/RNG恢复、外部8维往返、原坐标裁剪、原100步DDPM常数下的理想预测器完整采样往返、安装采样器不改模型状态/RNG、严格normalizer加载与旧端点拒绝、持续期望目标执行接线。它们证明实现口径一致，不证明真实DINO策略已学会动作或机器人任务成功。没有加载真实策略/编码器权重进行本轮推理。

本机完整训练依赖导入检查仍仅CPU，无模型/CUDA；本机Torch2.11+cu128、TF2.21与目标Torch2.5.1+cu118、TF2.15不同。远端材料/环境/空间准入尚未执行。Shell只做LF/shebang静态检查，不声称已执行bash -n。原vendor及237424部署保持字节不变。数值准备首次因NPZ在循环内重复解压而中断，已改一次读入再计算并完成；没有触发训练或额外原数据读取。

## 固定预算与停止条件

单A40、4CPU/40G RAM，8192×8、seed42、LR1e-4，不增种子或训练臂。预计50–65分钟，以237424相同网络/数据的45分钟训练计时为估计依据，新增仿射采样开销尚未实测；Slurm硬90分钟，CPU准入后全程序5100秒，训练3900秒，8GiB输出/原子落盘余量。du仅项目占用估计，非权威quota，不自动清理。

每1024步原子保存完整模型、三组Adam、计划/日程/RNG。只选最终8192端点；错误、预算或未通过时保留原子状态并结束，无自动恢复、重提、加步、池/损失/控制器扫描。固定预算不保证fresh模型收敛，也不以未收敛为自动续训理由。

沿用原128训练组+64开发组、每窗1101/1102/1103三采样draw（非三训练seed）。主门槛完全不变：两角色第一命令位置/旋转均值均优零命令和训练均值，且相对零命令的两项配对组bootstrap95%区间上界均<0。全H16/末条命令和夹爪命令准确率为次指标。只报告逆变换后的原命令误差；位置是归一化记录单位，不擅自换算cm。237424原结果保留作参照，不重采样旧策略。

通过仅支持审阅已准备的一次排除正式矩阵的工程回合，不自动仿真或放行13模型。本候选失败则停止这条局部修补训练链，带现有证据进行整体技术方案审阅；不自动换一组初始化/尺度/LR，也不宣告零样本不可行或自行转为LIBERO监督路线。

## 用户审阅后手动操作

只上传小代码/数值包，现有原Bridge分片、冻结编码器资产和原环境继续使用；不需要任何旧学习策略权重、旧token bank或未来到达位姿目标仓。

本机PowerShell：

```powershell
scp "D:\ntu_related\dissertation\Zero_shot\cluster_packages\bridge_command8_fresh_scaled_v1_20261031.zip" zixiao005@10.97.216.128:/projects/Zeroshot/
```

集群终端：

```bash
set -e
cd /projects/Zeroshot
test ! -e bridge_command8_fresh_scaled_v1
unzip -n bridge_command8_fresh_scaled_v1_20261031.zip
cd bridge_command8_fresh_scaled_v1
bash submit_once.sh
```

提交入口先检查完整环境、manifest、normalizer和原图文材料及空间，才唯一一次sbatch；标记不明确时保留标记查队列，不重提。输出`BRIDGE COMMAND8 FRESH SCALED: 编号`。

```bash
JOB=实际编号
tail -n 80 -F /projects/Zeroshot/logs/bridge-command8-fresh-scaled-${JOB}.out
```

完成标记`BRIDGE_COMMAND8_FRESH_SCALED_COMPLETE`在最终验证和ZIP之后。只先取原小包；本机PowerShell：

```powershell
$job = "实际编号"
$dest = "D:\ntu_related\dissertation\Zero_shot\reports\bridge_command8_fresh_scaled_$job\cluster"
New-Item -ItemType Directory -Force -Path $dest | Out-Null
scp "zixiao005@10.97.216.128:/projects/Zeroshot/runs/bridge-command8-fresh-scaled-$job/small_results.zip" $dest
```

保留集群policy.pt及latest_state.pt。助手本轮没有SSH、sbatch或任何训练/仿真操作。

# 236840：固定五回合关抽屉5/5成功，域内正对照完成

2026-10-31。原ZIP和完整policy已下载，指纹、保存预测独立复算及完整CPU模型恢复通过；随后按原预算完成一次五回合验收，未追加训练或回合。

| 初态示教ID | 环境判定 | 20Hz控制步 | 策略调用 | 模拟时间 |
|---|---|---:|---:|---:|
| 25 | 成功 | 84 | 21 | 4.2秒 |
| 26 | 成功 | 72 | 18 | 3.6秒 |
| 27 | 成功 | 72 | 18 | 3.6秒 |
| 28 | 成功 | 72 | 18 | 3.6秒 |
| 29 | 成功 | 64 | 16 | 3.2秒 |

五次均从预定演示states[0]恢复一次，无专家动作、执行中状态修正或运行错误。每次观察后预测H16原生7D命令，执行前4条不同时间步命令后重观察；恒定开爪，非夹爪时序成果。总364控制步/91调用，整个child113.141秒（含模型加载与环境/图像输出），不是113秒模拟时间。每回合≤200tick/50调用，总≤600秒均满足。首/末帧已目视核对；trace成功标志、种子、动作范围与GIF帧数已检查，无后续重跑。运行器原部署全部SHA未变。

结果位于`reports/libero_native_supervised_results_20261031/rollout1/`，含原report、五套execution.gif、initial/final.png及trace.json。审计为`reports/libero_native_supervised_236840/download_audit.json`和`rollout_closeout.json`；可用`tools/audit_libero_native_236840_download.py`只读重算下载结果，不会训练/采样/仿真。

## 对论文主线的意义

现在有冻结DINO+Adapter+扩散后端通过原生命令实际关抽屉的正对照，因而不能笼统声称DINO不能学、该网络完全不能控制LIBERO。该参照解除域内学习策略闭环可行性的证据缺口，可保留其接口供主线迁移工作参照。

这是单任务目标域监督、同任务五个固定留出示教初态的5/5，未证明未见任务泛化或零样本迁移；也未证明监督域、原生命令、机器人状态和全新初始化之中哪一项单独导致改善。OXE源端多窗能力及未来实测位姿→原生OSC的动作/状态合同仍是主实验阻塞，老师五条件/13模型未放行。下一项回到该合同和源策略，不延长本域内训练、扩任务或优化成功率。

## 本地验收边界

policy SHA cb79d8333de212aa95810050908dba90e543ccf0f76560e70fc29deb4d824bd8、完整state SHA f7a36dc0c1fa9b767efc226b562ec4f26dffb2c335af30e71b0099f8ae6a047d均匹配。train/dev128窗×3draw全部保存目标/mask/上一命令/索引与原绑定输入逐值一致；独立float64运动MSE=.002821958128/.011661449239，与报告float32舍入一致。原准入确认为Torch2.5.1+cu118，原验证文件已接收，集群报告完整Adam检查通过；Adam仍未本地下载，不称本地独立Adam验收。

## 以下为下载前历史记录（待下载/未仿真描述已失效）

# 236840：域内原生动作训练完成，下一项为固定五回合

2026-10-31。完整日志已接收，原文及SHA保存在`reports/libero_native_supervised_236840/`。4096更新完成，无程序错误。1888张图像冻结编码完成；编码器更新0、没有加载原OXE权重。此为单任务目标域监督对照，zero_shot=false、formal_matrix_admitted=false，尚无实际策略成功率。

## 已有离线结果

统计六个原生运动命令维度，不包括恒定张开夹爪。每角色固定128窗×3采样draw；训练20条演示，开发5条演示，窗口重叠且开发复用，不能称独立128任务或三训练seed。以下数值来自集群报告，原预测尚待下载复算。

| 角色 | 模型运动6D MSE | 训练均值命令基线 | 上一命令保持基线 |
|---|---:|---:|---:|
| 训练 | 0.0028219580 | 0.0524997997 | 0.0377486546 |
| 开发 | 0.0116614488 | 0.0514015043 | 0.0370542434 |

开发MSE相对两基线分别低77.313%和68.529%，训练分别低94.625%和92.524%。这是原生命令空间的混合位置/轴角归一化MSE，不能换算成某个统一厘米误差、每步准确率或关抽屉成功率。

训练可见18个更新点，有限值已核对；未声称逐步独立审计4096更新。报告初末状态指纹不同，完整模型/Adam读回验证按代码顺序在COMPLETE前运行，原验证文件仍需取回才可独立查看。完整policy SHA为`cb79d8333de212aa95810050908dba90e543ccf0f76560e70fc29deb4d824bd8`。

主体计时665.917秒（11.10分钟）、reserved峰值5.666GiB；不含CPU准入及最终CPU验证/ZIP。冻结编码与准备阶段33.649秒包含加载等，不等于纯GPU每帧速度。尚无sacct终态。末尾Ctrl+C在COMPLETE之后，未中断已完成的端点。

## 注意力警告与证据边界

日志出现Memory Efficient attention backward非确定性警告。原协议实际设置为deterministic=True、warn_only=True；此警告未停止训练。不承诺固定种子/批次/保存RNG即可逐位重训复现，不因此追加重训或切换后端。警告中的bridge-diffusion包路径本身不能证明启动了错误环境；实际运行环境尚待原admission文件核对，当前不重开环境排查。

本轮没有识别OXE迁移的唯一根因。监督域、原生命令标签、状态含义及初始化均与旧229116不同；新对照不能当单变量修复或主13模型进展。明确得到的是同域离线命令误差下降，是否具备实际关抽屉能力还要由已登记五回合回答。

## 接下来只取现有端点并完成实际验收

```powershell
scp "zixiao005@10.97.216.128:/projects/Zeroshot/runs/libero-native-supervised-236840/small_results.zip" "D:\ntu_related\dissertation\Zero_shot\reports\libero_native_supervised_results_20261031\small_results.zip"
scp "zixiao005@10.97.216.128:/projects/Zeroshot/runs/libero-native-supervised-236840/policy.pt" "D:\ntu_related\dissertation\Zero_shot\reports\libero_native_supervised_results_20261031\policy.pt"
```

取回后核对端点SHA、原报告与保存预测，再执行已准备的`evaluate_local.py --run`：demo25–29，各200tick/50调用，总600秒，保存五份视频与原生成功结果。当前模型与小ZIP在本机仍缺，尚未加载新模型或开始仿真。没有新增训练授权/提交、续步、种子或自动重试；离线MSE不设成无限前置门槛。

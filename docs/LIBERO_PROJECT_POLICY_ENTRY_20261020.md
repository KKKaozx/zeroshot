# 主模型到LIBERO执行器的推理回调

2026-10-20。新增`code/libero_project_policy.py`，复用主模型`RobotAdapterModel`、`dataset.prepare_image`和`train.tokenise`，供已有`execute_policy_chunk`调用。没有更改训练模型、训练损失或采样器源码，没有修改Push-T214273提交包、Bridge207558或CLIP冻结要求。

## 已实现

- 从可信的本地主训练checkpoint读取配置和完整学习模块权重，严格核对键集合与形状，不对缺失的夹爪头或其他模块进行随机补全。冻结CLIP仍由本地已有目录加载，不下载模型。
- 固定检查8维工具相对动作、尺度、预测长度、扩散类型以及当前夹爪二值命令编码；连续开度输入与此LIBERO命令接口不兼容时拒绝加载。当前入口不自动转换Bridge连续测量，也不把二维Push-T权重扩展成三维。

2026-10-20后续收紧：仅`current_gripper_encoding=binary`还不足以证明输入是命令。现在还要求训练配置明确`current_gripper_semantics=last_command_open_positive_v1`。旧主训练配置只记binary、实际可能取传感器阈值，不能直接用于此入口，也不能通过事后填字段把实测状态改称命令。见[四来源准备时发现的语义缺口](OXE_BASELINE_READINESS_20261020.md)。

随后新增两维命令/可用性模式：编码`last_command_with_availability_v1`、语义`last_command_open_positive_with_availability_v1`，支持未知先前命令`None`。它是新训练模式，不是改名迁移旧实测模型；完整接入及已知/未知评价分母检查见[命令输入链路](OXE_COMMAND_INPUT_CHAIN_20261020.md)。
- 直立RGB沿用训练的letterbox和归一化；文本沿用原tokenizer入口、77长度上限与attention mask。这里不再次翻转图像，方向转换由执行器完成一次。
- 每次推理采用`eval()`和`inference_mode()`，检查两个CLIP编码器均冻结。采样步数、beta表类型、预测目标、裁剪及夹爪解码规则保留检查点配置，不为部署另设推理配方。
- 每次调用用明确采样seed加调用序号，保留调用者CPU及实际使用CUDA设备的随机状态。输出不另行阈值化或裁剪，必须已经是合法四元数与二值夹爪命令；记录输入/输出SHA、采样seed和调用序号。

连接方式是先加载一次，再把回调传给现有执行器。下面是调用关系示意，不是已具备正式输入的可提交任务：

```python
policy = load_project_policy(checkpoint_path, local_clip_path, execution_contract,
                             device=device, sampling_seed=sampling_seed)
result = execute_policy_chunk(env, policy, observation, instruction,
                              current_open_command, execution_contract,
                              native_step_budget=native_step_budget)
```

`execution_contract`仍须明确目标到达时间、执行前缀、尺度和TCP依据。当前主模型不使用回调中的参考世界位姿作为额外神经网络输入；该位姿由执行器用于固定参考系还原，不能说本次给模型添加了当前位姿特征。

## 实际检查

在已有LIBERO隔离环境中构造本地随机小型CLIP和tokenizer，运行实际主Adapter、扩散解码器、夹爪头及保存/加载路径。未下载预训练权重，未用真实任务图像或优化器。检查包括：

- 主调用路径与新回调的输出最大差为0；独立保存/重载后的连续两次推理相同。
- 完整模型state dict推理前后不变，CLIP冻结且处于eval；CPU全局RNG保持。
- 错尺度、错长度、连续夹爪编码、缺失夹爪权重、不合法输入以及解冻CLIP均拒绝。

命令：

```powershell
& 'training_cache/libero_native_env_v1/Scripts/python.exe' scripts/check_libero_project_policy_20261020.py
```

[检查报告](../reports/libero_project_policy_20261020.json)。首次用基础Python时缺少SciPy，随后直接使用已有LIBERO环境通过，没有为此新建环境或安装依赖。CUDA路径未实测。

## 尚不能据此宣布的结果

检查使用随机小网络，不是训练好的ViT-L/14策略；0训练更新、0物理步、0基准成绩。完整权重键存在也不证明夹爪实际获得了有效训练。主checkpoint不包含冻结CLIP权重，当前加载器记录原主干名与本地目录，但未自动证明两者权重身份相同；正式执行必须核对训练主干指纹，不能只看形状相同。

推理入口至此已有可执行代码和一致性证据。下一项转向OXE无增强主基线的真实训练清单、来源合同和固定预算，并绑定完整动作/夹爪训练及部署配置；不再重复小网络接线或原生示范回放。真实模型闭环、跨机器人TCP、源数据时序和四条件正式成绩仍未完成。

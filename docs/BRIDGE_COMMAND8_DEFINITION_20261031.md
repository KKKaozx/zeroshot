# Bridge控制增量8维：候选定义、可运行代码与一条真实记录样例

2026-10-31。本轮没有训练、模型推理、仿真、集群操作，也没有读取LIBERO示教或使用236840权重/统计。OXE训练→LIBERO零样本评估目标保留。本包是本机数值原型，无训练入口、sbatch任务或tail日志命令；不是新的策略成功结果。

## 已经实际落实

新增code/bridge_controller_command_contract.py：原始Bridge动作的平移3维加Euler增量3维转换为“位置增量3＋相对四元数4＋显式二值夹爪1”，仍为8维，但使用独立版本标记，不能混入旧未来实测位姿模型或执行合同。Euler增量组成Rz*Ry*Rx，是世界轴旋转增量，不是给实测Euler角逐项相加。

实现控制目标更新和原生反馈入口：

```
p_new = c + d + D * (p_previous_goal - c)
R_new = D * R_previous_goal
```

c为当前实测旋转支点；previous_goal是上一次控制期望，不能默认为当前实测位姿。给定轴和单位约定后，增量先转换到目标坐标。每条源控制增量只推进目标一次，其间原生反馈继续追踪同一目标；不能把一条名义5Hz动作在4个20Hz控制步中累加4遍。原反馈几何函数直接复用，没有换增益/插值或重做旧5Hz实验。

工作区边界由调用者明确给定，本次数值例显式关闭工作区裁剪；没有假装恢复源端真实限幅。源夹爪原值存在0.206838/0.306715等中间值，不能直接当Panda夹爪速度。本例仅将原始0/1端点映射为二值开闭；2条中间值监督掩码为0，不执行未知夹爪，也不重开夹爪专项。

## 新证据与适用边界

本轮读取的是既有清单第一条固定Bridge训练记录：bridge_data_v2-train.tfrecord-00667-of-01024::0，payload SHA5a00e99d952e42d17e67d570d238115b0e56ca97cc060aa7027e293f98a550f7。45行state/action，0图像解码。没有重新做旧metadata/单位/TCP核查；本次新增读取原动作数值，并沿用既有来源绑定。

公开[采集器](https://raw.githubusercontent.com/rail-berkeley/bridge_data_robot/main/widowx_envs/widowx_envs/base/robot_base_env.py)将增量作用于上一次期望目标，使用当前EEF位置作为旋转支点；控制流程还包含动作/工作区裁剪、可配置的每步目标重置。[变换函数](https://raw.githubusercontent.com/rail-berkeley/bridge_data_robot/main/widowx_envs/widowx_envs/utils/transformation_utils.py)给出了上述刚体更新规则。

公开[raw→numpy转换](https://raw.githubusercontent.com/rail-berkeley/bridge_data_v2/main/data_processing/bridgedata_raw_to_numpy.py)从policy_out读取动作，只保留full_state等字段；[numpy→RLDS实现](https://raw.githubusercontent.com/kpertsch/bridge_rlds_builder/main/bridge_dataset/bridge_dataset_dataset_builder.py)直接复制对应state/action。该公开RLDS实现标为1.0.0，本地缓存为0.0.1，不能据此宣称精确导出版本已绑定。公开采集器有desired_state，缓存没有该字段；动作/观测的确切对齐和历史配置也不能由当前字段自动恢复。没有重新施加旧相机延迟移位。

因此，“观测state＋action直接得到源端真实绝对期望轨迹”没有被证明，不能生成这类伪真值；缺的是期望目标历史或等价初始目标及重置/限幅设置。新的直接控制增量编码不需要恢复这些绝对期望标签，但原动作含义、导出对齐及裁剪仍是需要明确的来源假设。不得把精确历史commit追查变成无限期停工条件；若以公开采集实现族的工程假设继续单一源端候选，须将假设、完整训练代码和固定预算单列给用户审阅，不称已恢复物理真值，不自动训练。本轮尚未生成该训练包。

## 真实数值示例

取固定记录index1：原始平移约[-0.000598,-0.014494,-0.012539]，Euler增量约[0.013551,0.010566,-0.075776]，夹爪1。原动作编码到8维为：

```
[-0.00597699, -0.14494452, -0.12538753,
  0.00697080,  0.00502259, -0.03791325, 0.99924410, 1]
```

这是实际记录的候选增量编码，没有使用未来实测位姿。相同当前观测和动作，给定两种不同的“前一期望目标”可得到相差1厘米的新目标（在既有米尺度假设下）。A场景明确令前一期望等于当前测量；B场景将前一期望X偏移1厘米。两者都是数值假设，不是采集器当时真实目标，也不是LIBERO真实观测。对应原生命令存于numeric_example.json，仅做代数计算，0仿真。

复用既有未校准映射只是为了展示完整数值链，不证明源与Panda已物理校准。该反例不是旧训练失败根因的因果证明。

## 验证与固定范围

5项CPU检查通过：非交换Euler/齐次变换、非单位轴与尺度映射、缺期望历史的非唯一性、目标反馈不重复累加及显式裁剪、旧合同/未知夹爪/缺历史拒绝。另从保存的公开浏览器源码捕获中只选取已检查的纯数学函数，实际执行44条原动作×2个显式历史假设，共88次数值对照；位置最大差1.11e-16、旋转矩阵最大差6.66e-16。这些检查不等于88个任务或任何机器人能力指标。

本地原型固定1条训练记录（1.59MB payload，不复制入包）、0图像/模型/新训练/仿真；不扩样本、扫坐标、改控制频率或继续工程检查。浏览器捕获与其摘要已保留，摘要仅标识捕获文本，不是原始上游文件字节或历史commit。终端下载原始源码遇DNS失败，改用网页工具读取并保存捕获，没有安装网络工具或修改权限。

本机复查：

```powershell
& 'D:\ntu_related\dissertation\Zero_shot\training_cache\libero_native_env_v1\Scripts\python.exe' -B 'D:\ntu_related\dissertation\Zero_shot\tests\test_bridge_controller_command_contract.py'
```

真实样例构建器tools/build_bridge_command_target_example_20261031.py默认拒绝覆盖已有numeric_example.json，保留本轮证据，通常无需再运行。根code新文件与单独候选配置已落地，旧数据、旧权重和所有历史部署快照未改；B1源端联合能力和B2物理执行仍未解除，13模型未放行。

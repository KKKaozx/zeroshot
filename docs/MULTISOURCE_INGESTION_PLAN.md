# 已有数据的接入顺序与缺口

2026-10-07。此次实际读取文件清单、TFDS元数据和VIOLA HDF5字段/形状，没有解码图像、读取数值动作目标、下载或训练。文件存在及分片数量不等于完整性/训练合同验收。盘点证据：[multisource_local_inventory_20261007.json](../reports/multisource_local_inventory_20261007.json)。

| 来源 | 本地版本及约占用 | 已有支持与证据 | 角色及下一项 |
|---|---|---|---|
| Bridge | TFDS bridge_data_v2 0.0.1，123.38GiB，1152分片 | 生产解析器；已完成55演示相对位姿/夹爪时间/RGB实读验收，完整模型pilot运行但未通过整体预测 | 核心训练；继续扩充同五任务独立演示，固定旧开发/测试隔离，不把解析验收说成性能通过 |
| Language Table | TFDS language_table 0.1.0，399.87GiB，1024分片 | xy解析器和只监督xy的mask已存在；元数据描述xArm planar pushing，仍需核对具体real来源 | 核心训练；核对真实来源、xy坐标/时序与监督mask实际传播，其他六维不得当真值；没有夹爪的输入观测也须表达缺失 |
| BC-Z | TFDS bc_z 1.0.0，50.43GiB，576分片 | reached位姿及专用native_commands诊断解析器已存在；二者目标含义不同，native10步不可直接拼成Bridge16步 | 核心训练；确定用于多源的reached合同，核对旋转、位置单位、命令夹爪时间及输入测量编码，保留eval分区 |
| Fractal | TFDS fractal20220817_data 0.1.0，111.07GiB，1024分片 | RT-1 reached pose解析器已存在，夹爪closedness反向到open符号 | 核心训练；实际核对xyzw、工具坐标、单位、夹爪及动作时刻，不把库中有函数当验收 |
| LIBERO | 93.54GiB，130个HDF5 | 已核对一个文件元数据与OSC_POSE控制器；正式rollout未完成 | 最终任务评价；训练演示若作工程适配需单独说明，最终任务划分先隔离，不用于OXE零样本模型选择 |
| CLIPort | 0.43GiB，50个pickle文件，不是50条独立演示 | 原生10训练演示/61 primitive检查及部分专家回放证据；统一loader已排除旧合成动作 | 最终任务评价；原生pick/place与连续轨迹桥接仍未通过，不能将四个合成位姿放入训练 |
| VIOLA | 19.01GiB，3个HDF5文件，不是3条演示 | 实际元数据首demo action为4维，含图像、ee_states和gripper_states；通用4D HDF5解析器存在但默认排除缺旋转来源 | 后续附加训练/鲁棒性对照；先确认4维动作语义、旋转缺失、语言来源与augmented样本和原演示的划分关联 |
| Fanuc | TFDS fanuc_manipulation 1.0.0，8.85GiB，93分片 | features描述action为6维xyz和Euler增量，观测eef为7维位姿、state为13维含夹爪状态；语言字段steps/language_instruction，与现有自然语言eef schema不同 | 跨机器人附加实验；需要明确解析入口、控制帧/尺度/频率及夹爪命令监督缺失，不能用观测夹爪状态冒充命令 |

本地路径在盘点JSON中记录。Bridge与Language Table在E盘，BC-Z/Fractal/VIOLA/Fanuc在本地OpenX目录，LIBERO与CLIPort在对应目录。不重新下载或拷贝整份到系统盘。

## 两项已经发现的实际接入限制

当前 `UnifiedRobotDataset` 的 `bridge_current_gripper=continuous` 明确只允许Bridge；非reached的BC-Z诊断目标也限制为单独BC-Z。因此，当前已经通过的Bridge训练命令不能直接指向四来源目录。多源接入需定义每来源的夹爪观测编码与缺失观测表达、保留明确有效监督掩码，并为来源混合建立独立清单。不能只删除保护条件。

Language Table输出虽为接口占位8维，真实监督只有xy。现有pose loss会接收mask，但多源有效监督、输入夹爪占位以及采样时未监督自由度是否干扰共有预测仍需实读核对。当前Bridge五任务manifest不允许其他来源，不能冒充多源manifest。

## 数据规模与容量

四核心来源约684.75GiB，大于集群200GB十进制SSD配额。第一阶段采用原盘按固定清单导出、分批传输；扩充数据包规划上限5GiB，项目保留至少20GiB余量，不同时保存全量压缩和解压副本。上限是规划约束，不是已完成实时配额查询。最终大规模训练需另定流式/存储方案，不把小批试运行写成老师要求的全量训练完成。

本地TFDS元数据的train演示数：Bridge53,191、Language Table442,226、BC-Z39,350、Fractal87,212。Bridge/Fractal与邮件中的参考数25,460/73,499不同；现有版本标签已记录，还需核对导出来源与计数口径。不得仅凭版本名认定与参考数据完全一致，也不据此判定下载错误。

## 下一阶段操作顺序

1. 继续盘点未扫描的Bridge train分片，补现五任务训练候选；旧10开发/13测试与旧诊断演示隔离。新增候选先按元数据收集，不根据开发预测误差选数据。
2. 对Language Table、BC-Z、Fractal各取预先固定的训练记录，执行本来源单位/帧/时序/有效监督核对。只补当前缺项，复用已有解析器；没有加载最终评估目标。
3. 建立四来源训练/开发清单及源内完整演示隔离、场景/目标去重、采样比例。先固定样本数和更新预算，再提交混合短预检；来源均衡与按规模采样需明确选择，避免399.87GiB的Language Table主导所有更新。
4. 多源预检通过后进行一次预先固定预算的训练并与Bridge扩充参照比较。正式独立训练seed与零样本任务评价仍是后续验收，采样seed不能替代训练seed。

本清单是接入计划，不是已经实现的四源训练配置。此次未新增网络、修改数据处理或启动训练。先使用四核心来源；VIOLA/Fanuc单列实验，LIBERO/CLIPort保留评价角色，不将八个来源一次性混合。

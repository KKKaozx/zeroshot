# 缓存转换链核对：本轮结果与剩余缺口

2026-10-20。接续既定的数据合同核对，没有训练、推理、读取新的样本目标、重新划分演示或改动生产模型代码。

后续更新：[独立实现支持补充](OXE_POSE_SOURCE_SUPPORT_20261020.md)已找到Fractal同名0.1.0版本的明确xyzw转换依据，并补查Bridge底层接口单位。下文是补充之前的审查记录，保留其证据边界。

## 找到了什么

**Bridge 找到了与缓存更接近的 NumPy→RLDS 转换家族**：[kpertsch 的转换器](https://github.com/kpertsch/bridge_rlds_builder/blob/main/bridge_dataset/bridge_dataset_dataset_builder.py)使用与缓存原始路径一致的 `numpy_256/scripted_numpy_256`，按原始文件内序号保存 `episode_id`，直接复制状态与动作数组。它不输出 `time_stamp`。但该公开分支声明1.0.0，本地缓存是0.0.1，尚未恢复实际发布提交，不能称为已证明逐字相同的转换器。

[作者原始数据→NumPy代码](https://github.com/rail-berkeley/bridge_data_v2/blob/main/data_processing/bridgedata_raw_to_numpy.py)保留 `full_state` 和 `time_stamp`，并对2021-07-23以前的采集做相机延迟对齐。因此，若能取得同一原始NumPy文件，可以补查时间；不能在已转换缓存上再盲目执行一次延迟修正。

**Bridge 朝向有默认旋转约定**：[作者状态转换](https://github.com/rail-berkeley/bridge_data_robot/blob/main/widowx_envs/widowx_envs/utils/transformation_utils.py)保存的是扣除默认旋转后的欧拉角。写成矩阵：

```text
R_recorded = R_tool × Dᵀ
R_tool     = R_recorded × D
```

[控制器](https://github.com/rail-berkeley/bridge_data_robot/blob/main/widowx_envs/widowx_controller/src/widowx_controller/widowx_controller.py)中的 `D` 通常不是单位矩阵，还受配置的底座角度影响。本地缓存没有各次采集的该配置；这里是公开代码依据，尚未完成每条历史导出的绑定。

这对当前接口的影响已单独核算：

- 新的源坐标位置差直接用 `p_future − p_current`，不依赖工具朝向；新的左相对旋转中，同一演示内不变的 `D` 会抵消。
- 旧的工具坐标位移若直接使用记录角度，其坐标轴会与物理TCP轴相差 `D`。旧接口不能凭数值往返正确就宣称物理轴已标定。
- 固定正交旋转本身保持欧氏距离，不足以单独解释旧模型离线位置误差或泛化差；没有据此宣布找到根因，也没有修改207558。

64个独立合成矩阵组合验证了上述抵消关系，最大矩阵元素差 `2.22e-16`。这是几何性质核算，不是实际机器人误差或模型性能提升。

## 哪些链接不能当作完整依据

[TFDS的共享RTX代码](https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/robotics/rtx/rtx.py)通过[importer](https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/robotics/dataset_importer_builder.py)读取已转换GCS数据；它不是采集器或原始转换器。BC-Z链接指向0.1.0，而本地1.0.0元数据指向 `google3` 内部构建模块，尚未建立两者等价关系。

[dlimp的另一份Bridge转换器](https://github.com/kvablack/dlimp/blob/main/rlds_converters/bridge_dataset/bridge_dataset_dataset_builder.py)处理480×640原图、深度和原始轨迹路径，与当前256×256 NumPy导出不同，不能用来替代本地版本证明。

[Language Table作者README](https://github.com/google-research/language-table/blob/main/README.md)可对应真实数据的442,226条演示，但示例版本为0.0.1，缓存为0.1.0；其仿真10Hz设定不能证明真实数据的导出间隔。XY监督、缺失夹爪输入的既定处理不变。

[Octo的RT-1转换](https://github.com/octo-models/octo/blob/main/octo/data/oxe/oxe_standardization_transforms.py)直接把Fractal已到达位姿作为本体状态，并没有独立解释四元数的分量顺序。BC-Z作者模型也不是该缓存的TFDS导出实现。不能把下游代码使用了某字段，当作物理约定已核实。

## 尚未完成的具体项

| 项目 | 当前结论 | 所需证据 |
| --- | --- | --- |
| 各来源位置到米的转换 | 未完成准确导出绑定 | 对应发布版本的字段规范、转换代码或原始采集接口说明 |
| Fractal四元数顺序 | 生产解析器使用xyzw，独立来源证据仍缺失 | 对应 `base_pose_tool_reached` 导出的明确分量顺序；不按误差挑顺序 |
| 导出观测的时间间隔 | 只有文献名义频率，缓存没有逐帧时间戳 | 同源原始时间戳或明确的导出抽帧/采集规则；非实时循环也不能保证严格等间隔 |

本轮公开源码核对已完成到上述边界，不能把尚未取得的内部转换代码写成通过。下一步限定为补齐这些来源证据，再确定目标时间与窗口清单；不为缺少来源依据而新增拟合实验、猜尺度或静默重采样。已冻结的训练/验证/OXE源留出演示角色不重新分配。

证据：[结构化报告](../reports/oxe_conversion_lineage_20261020/report.json)、[复核脚本](../scripts/record_oxe_conversion_lineage_20261020.py)。网页引用为本轮查看的公开分支，尚未绑定历史导出提交。生产数据/训练/执行接口的SHA与既有接口报告一致，`training_ready=false`。

# 三核心来源实读与夹爪合同修正

2026-10-07。使用已有文件、生产解码方法和独立旋转矩阵还原。各来源固定第一train分片，按长度选择首条超过16帧的记录，不依据误差筛选。Language Table记录0为27帧/3窗口，BC-Z记录1为91帧/19窗口，Fractal记录0为115帧/25窗口。图像实际解码、动作有限值及相对位姿还原通过，最大位置数值误差<3e-7、旋转矩阵误差<2e-7。没有模型加载或训练，没有评价测试目标。

Language Table仅xy维有效监督，其他六维mask为0，实际生产返回已核对。其夹爪输入仍是占位，混源时的缺失输入表达尚未处理。数值往返检查不能证明物理单位或执行等价，各来源完整性、频率、单位和场景隔离也未完成。

## Fractal：发现并修正相对夹爪命令解释

固定[Octo作者标准化代码](https://github.com/octo-models/octo/blob/241fb3514b7c40957a86d869fecb7c7fc353f540/octo/data/oxe/oxe_standardization_transforms.py)的RT-1转换先调用[rel2abs_gripper_actions](https://github.com/octo-models/octo/blob/241fb3514b7c40957a86d869fecb7c7fc353f540/octo/data/utils/data_utils.py)：正值关闭、负值打开、死区保持前一个状态；初始状态按第一个有效命令的反向推断，无有效命令时默认打开。原解析器把raw值小于0.5一律当打开，丢失了关闭后零命令的保持状态。

新增显式 `rt1_gripper_policy=relative_scan_v2`，先按整个演示重建绝对开关命令，再按目标对应的前一时刻取值。固定Fractal记录的25窗口400个重叠动作目标中，旧阈值与新策略有162项不同；新策略206项打开，旧策略368项打开。窗口重叠，不能称162次独立物理事件。

保持/关闭/重新打开、首次打开前的初态、全零、死区和非有限输入共5项测试通过。新Fractal训练必须显式选择新策略；旧checkpoint缺省保持legacy阈值，评价按保存的policy恢复，不把旧标签悄悄改掉。策略写入checkpoint data_config及数据身份；本次Bridge模型未使用Fractal，此修正不能解释189937的失败。

## BC-Z：未来观测状态与发出的命令不同

现有reached目标的夹爪标签取下一观测的sensed_close，而作者转换取当前future/target_close的第一命令。前者是实测状态监督，后者是命令监督；不能在混源时把两者当同一含义。固定第一train分片记录1/2/3，邻接比较分别0/90、72/200、20/199项不同。记录1恰好全开，单条通过不足以验收切换语义。

新增显式 `bcz_reached_gripper_policy=preceding_command_v2`，保持原到达位姿轨迹，夹爪使用前一观测的首个绝对关闭命令。三个固定记录1792个完整窗口目标与原始对应命令一致，实际生产解码检查通过。旧future_measured_v1默认与旧权重保留，评价和身份记录选择的合同；新混源计划应明确选命令策略。此处不宣称项目的未来位姿与作者原生增量位置动作执行等价。

## Bridge扩充与下一项

新增32–63分片的52条同任务元数据候选，检查原始origin/id未与现45训练、10开发、13保留测试重复；旧开发和测试隔离保持。新增候选尚未进行全数值/RGB验收，继续寻找更多独立演示，不马上把52条当训练就绪。

混源训练仍未就绪：连续Bridge单来源保护未删除；缺失观测、部分自由度监督、采样周期和各源单位仍需补齐。此次实读完成与政策修正不是模型性能结果。

证据：[三来源实读](../reports/core_source_fixed_record_checks.json)、[BC-Z状态/命令对照](../reports/bcz_measured_command_label_check.json)、[Bridge扩充候选摘要](../reports/bridge_expansion_candidate_summary.json)。

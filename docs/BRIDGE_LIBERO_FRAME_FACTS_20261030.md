# Bridge→LIBERO动作定义：本地事实核查收口

2026-10-30。已完成一轮静态事实抽取，读取一个固定已有Bridge训练记录和一个既有排除任务XML，0模型/仿真/训练/远端操作。复用2026-10-20来源谱系结论，没有重做来源比较、oracle、控制参数或策略实验。两个新增CPU测试验证固定变换组合及错误拒绝，不是能力指标。

## 本次具体落实

新增可运行的 `code/libero_eef_frame_definition.py` 与 `tools/extract_bridge_libero_frame_facts_20261030.py`，实际生成 `configs/libero_panda_eef_observation_definition_20261030.json`。配置绑定本机观测/控制器源码SHA和实际工程任务XML SHA，范围只到该Panda/XML，不能自动推广其他机器人。

| 项目 | 精确定义 |
|---|---|
| `robot0_eef_pos` | 世界坐标中的`gripper0_grip_site`位置 |
| `robot0_eef_quat` | `robot0_right_hand` body的世界姿态，xyzw |
| 观测使用的虚拟EEF坐标系 | 原点在grip site，坐标轴平行right_hand；不是hand body原点 |
| OSC反馈 | grip site的位置和姿态 |
| hand body→grip site固定几何 | body坐标中位置`[0,0,0.097]`模型长度单位；姿态`Rz(-90°)` |

因此可用新代码把观测姿态转换为控制器姿态：`R_world_site = R_world_hand * R_hand_site`。**0.097不是要额外加到观测位置上的TCP补偿**，观测位置已经在site。固定右乘因子在世界左相对旋转中抵消：`R_future_site R_current_siteᵀ = R_future_hand R_current_handᵀ`。当前反馈转换没有因这个差异而需要修正；不能再把−90°施加到既有左相对增量上。

## 源端缺的具体事实

实际读取的是当前训练清单第一条Bridge记录 `bridge_data_v2-train.tfrecord-00667-of-01024::0`，payload身份与清单绑定；没有解码图像、动作或未来目标。其原始路径为：

`/nfs/kun2/users/homer/datasets/bridge_data_all/numpy_256/rss/toykitchen2/pnp_sweep/99/train/out.npy`，`episode_id=10`。

该记录的全部字段及metadata已存入 `reports/bridge_libero_frame_facts_20261030/facts.json`。它没有显式单位、基座/TCP或采集配置字段；这是一条真实记录的事实，不冒充所有原始轨迹的全面扫描。公开采集代码和精确缓存版本之间的未绑定沿用旧报告，未重复联网调查。

要把跨机器人映射写成已证实，仍缺：

1. 精确`bridge_data_v2/0.0.1`导出器的版本/提交或发布规范，与原始采集记录的对应。
2. 对应采集轨迹的机器人/EEF/TCP配置，包括绝对姿态所需的`normal_base_angle`/默认工具旋转及几何点定义。只有numpy路径和episode编号不能给出这些参数。
3. 所选择的共同控制/接触点及源端到该点的几何对应。LIBERO内部hand→site变换不能确定它与WidowX哪个物理点对应。
4. **共同坐标迁移定义**：这是需要明确的表示/迁移设计，不能靠两个独立场景的机器人配置推导唯一“真实世界配准”。不能要求一份并不存在的通用Bridge→LIBERO世界标定作为无期限前置条件。

用户已同意的原XYZ按米解释和名义5Hz仍可作为披露的工程假设，不新增恢复全部真实时间戳的要求。原 `libero_bridge_pose_mapping_REQUIRED.json` 保持空缺；本次已证实的目标内部几何单独存放，不能把它代填为source→target注册。

## 对主实验和下一步的实际影响

本轮解除的是**该目标配置的观测原点/轴定义不清**这一合同子项，没有解除B1源端联合能力、跨机器人标定或5Hz接触执行。没有发现可据此改动当前执行器的坐标错误，也没有证据把最近失败归因于−90°或0.097。

旧来源审阅已经证明Bridge固定右侧默认工具旋转在源端左相对旋转中抵消。本轮直接沿用该结论：**准备源端位置/旋转联合能力修复方案不必等待该默认旋转值**；绝对状态端口的跨机器人对应则仍需要它或另一明确的新状态表示。不得把固定目标端几何说成源端旋转失败根因。

本轮事实核查已关闭，不追加记录/任务/映射扫描。下一项转入单一源端联合能力修复候选的方案、代码和固定预算；不自动提交训练，不恢复13模型或语言工具。既有234100位置学习和旋转失败、低噪声/旋转trace及有限仿真证据直接复用。

本机复查代码命令（事实输出目录已存在，抽取器拒绝覆盖，通常无需再运行）：

```powershell
Set-Location "D:\ntu_related\dissertation\Zero_shot"
$py = "D:\ntu_related\dissertation\Zero_shot\training_cache\libero_native_env_v1\Scripts\python.exe"
& $py -B tests/test_libero_eef_frame_definition.py
# 首次静态抽取入口；已执行，不重跑：
# & $py -B tools/extract_bridge_libero_frame_facts_20261030.py
```

代码、配置、测试和事实报告打包为 `cluster_packages/bridge_libero_frame_facts_local_20261030.zip`，不含权重/数据。无需上传、Slurm提交或tail，因为本轮没有集群任务。

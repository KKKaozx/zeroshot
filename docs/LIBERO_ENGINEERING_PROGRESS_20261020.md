# LIBERO正式实验的并行准备

用户授权Push-T训练期间推进其他工作。此次独立推进LIBERO官方执行入口，不改214273的包、作业、预算或环境，不接入207558模型，不重跑Bridge拟合、Push-T资源预检或闭环评分。

## 已实际完成

本地原93.55GiB数据保留原位置，无重复下载。以官方提交8f1084e3132a39270c3a13ebe37270a43ece2a01核对全部130任务的HDF5文件名、BDDL和初始状态资源：Spatial/Object/Goal/10各10任务，90套件90任务，共6500演示；130个文件都记录OSC_POSE、20Hz、OpenGL图像约定。仅读取HDF5属性/形状和官方文本，不解码最终测试数值或初始状态。

新建工作区隔离环境training_cache/libero_native_env_v1，复用基础Python3.10/Torch，仅在新venv安装仿真依赖（robosuite1.4.0、MuJoCo2.3.7、NumPy1.26.4等）。原基础环境与集群Push-T环境没有改动。Windows适配限于官方支持的private macros选择GLFW/关闭硬编码/tmp日志、复制同一包内字节相同的MuJoCo DLL到robosuite期望位置；官方LIBERO受跟踪源码未改。该环境继承了一些不参与仿真的TensorFlow/Proto依赖，pip check有版本冲突，不能称全环境依赖检查通过；当前成功仅覆盖实测仿真入口。没有为此重装基础环境或安装第二份Torch。

对先前已检查元数据的LIBERO90任务`KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet_and_put_the_black_bowl_on_top_of_it`，固定demo_0：

- 官方OffScreenRenderEnv重置、两个128×128相机、8步零控制已实际通过。
- 记录XML仅重定位95个文件路径，逐元素确认其他XML属性不变。开始恢复一次demo初始状态，状态最大差0；之后197个原生动作连续执行，过程中不恢复状态、不改动作、不接模型。
- 原生`check_success`从第186步为真，终点为真。EEF位置回放对记录均值差约3.845mm，最大4.381mm，不是逐元素轨迹复现，也不能称该工程控制等于原论文全部基准。
- 这是官方示范动作的执行正对照，不是用户模型成功、泛化或正式任务成功率。工程任务从最终成绩中排除，最终套件/初始状态计划尚待固定。

报告：[完整任务与资源匹配](../reports/libero_official_assets_20261020/report.json)、[重置](../reports/libero_native_reset_v3_20261020/report.json)、[连续原生回放](../reports/libero_native_replay_20261020/report.json)。失败的早期环境启动日志保留，不冒称所有尝试首次即通过。

## 新动作边界

`code/libero_action_interface.py`实现固定观测工具参考系下8维位姿目标到原生7维OSC命令的显式转换。保持整个动作块同一参考点，不累计相对位移；世界旋转增量按原生控制器左乘规则求解；夹爪开正/关负转为原生相反符号；越出单次控制范围直接拒绝，不静默裁剪；OpenGL图像到策略输入显式垂直翻转。

使用实际robosuite `OperationalSpaceController.set_goal`，128组合成姿态核对最大位置差2.78e-17m、最大旋转差7.30e-9rad；非法四元数/非二值夹爪/非有限值/越界拒绝，连续两目标同参考点测试通过。它只证明目标几何与原生控制器一致，不证明20Hz时序、跨机器人TCP标定、学习到的夹爪行为或模型任务成功。尚未将该模块接入生产模型或数据加载器。

## 不能跳过的正式训练缺口

主模型推理回调也已实现：复用训练图像/tokenizer与原扩散采样器，绑定尺度/长度/二值夹爪编码，严格加载完整学习模块。实际主模型代码配随机小型本地CLIP的CPU保存/加载与连续两次推理一致，直接路径和回调最大差0；不是训练好模型或物理闭环。详见[入口、检查及正式限制](LIBERO_PROJECT_POLICY_ENTRY_20261020.md)。接线检查不再重复，下一项转向真实OXE主基线清单和预算。

后续已修正LIBERO专用读取器的post-step夹爪一行错位和OpenGL图像方向，并添加显式时间的目标执行调度。前者用同一已使用工程演示的5个窗口核对，OXE解析器未变；后者用实际OSC目标计算和合成滞后反馈检查，未新增物理回放、未加载学习策略。时序/TCP物理合同仍未验收，不能称LIBERO模型已跑通或Bridge根因已找到。详见[修复依据、执行规则和剩余条件](LIBERO_TIMING_AND_READER_REPAIR_20261020.md)。

老师第一封明确核心是四来源OXE训练、LIBERO/CLIPort零样本评价；这次按该角色准备，不能静默将LIBERO训练演示加入四来源后称零样本。四组为无增强、仅五条同义改写、仅反事实、组合，独立训练seed42/43/44，至少12个运行；扩展套件会增加实际预算。矩阵已保存于[配置草案](../configs/libero_four_condition_matrix_20261020.json)，并不是可执行训练命令。

目前还缺完整多源manifest和掩码验收、训练专用同义改写/可靠反事实库、20Hz时序/TCP与夹爪执行合同、正式任务/初始状态划分、固定预算及完整恢复验收。四条件可选增强已接入主训练入口并通过合成合同检查，默认关闭；见[接线、库格式与检查边界](LANGUAGE_CONDITIONS_INTEGRATION_20261020.md)。这仍不足以直接启动12组完整研究实验；不提交空壳或用单个LIBERO适配模型冒充主实验。

下一项应在此已可运行的官方环境上接入当前完整模型的一次工程观测与动作调用，并完成时序/夹爪的明确合同，同时落实训练专用增强模块；不再重复本次原生重置/回放来凑进度。最终任务标签不进入这些工程调试。

官方依据：[LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)、[固定版本环境入口](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/8f1084e3132a39270c3a13ebe37270a43ece2a01/libero/libero/envs/env_wrapper.py)。

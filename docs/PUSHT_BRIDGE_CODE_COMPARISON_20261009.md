# 已跑通的 Push-T 与 Bridge：完整流程对照

这次不是重复试参。对照固定作者代码 5ba07ac6661db573af695b419a7947ecb704690f 与当前 code/models.py、code/train.py，结合已有训练报告；不修改模型、标签或提交训练。

| 项目 | 官方代码 / 已运行证据 | 现有 Bridge 193761 流程 | 能说明什么 |
|---|---|---|---|
| 输入 | 运行日志明确 image + agent_pos；代码支持历史观测 | 单幅 CLIP 图像 + 语言，当前末端位姿主要用于造相对标签而非数值条件 | 观测内容与任务不同，不是单独换动作头 |
| 目标 | Push-T 的二维控制动作 | 未来 16 步相对到达位姿，xyz + 四元数，另接夹爪头 | 到达状态误差不能等同于原生控制动作误差 |
| 归一化 | compute_loss 和 predict_action 调用同一个 normalizer，参数写入模型 state_dict；推理反归一化后执行 | xyz 固定除以 0.10m，四元数单位化；采样按坐标裁剪 | 确定存在配方差异，是否导致误差未验证 |
| 训练目标 | 官方同时支持 epsilon 和 sample，目标按 scheduler.prediction_type 选择 | 193761 是 sample/x0；193195 已测试 epsilon 且结果更差 | 不能把换 epsilon 当作尚未尝试的解决方案 |
| 采样 | 原版调用 scheduler.set_timesteps 与 step | 自写 DDPM/DDIM；此前 100 步公式核对未发现确定错误 | 成功参考不等于自写公式已被证明错误 |
| EMA | 作者 workspace 支持每步更新 EMA，eval.py 按 checkpoint 的 use_ema 选择模型 | 193761 未使用 EMA | 是未覆盖因素，不能直接宣布为根因；需先读取实际 checkpoint 配置 |
| 夹爪 | Push-T 没有夹爪 | 训练夹爪输入真实未来位姿，推理输入预测位姿 | 已确认训练/推理输入变化；不解释全部位置误差 |
| 评估 | 环境闭环执行、逐回合最高覆盖 reward | 真实演示上的离线位姿和切换误差 | 各有用途，数值不能直接对比 |

## 为什么先读取已经下载的 checkpoint

实际 JSON test seeds=4300000..4300049，而下载的网页 config.yaml 为 test_start_seed=100000。因此不能默认网页中的模型尺寸、历史步数、EMA、训练轮数均为这次实际运行参数。原版 eval.py 用的是 checkpoint 内 cfg。

read_checkpoint.sh 只在 CPU 上读取已存在的约 4 GB checkpoint，导出 cfg 和 model/ema_model 中保存的 normalizer 参数。无需安装、联网、GPU、训练或重新评估；输出为小型 YAML/JSON。执行前仅做了本地 Bash 和 Python 语法检查，集群读取尚未运行。

这一步的否证标准明确：以保存的 cfg 核对此前关于历史长度、采样器和 EMA 的说法；以权重中 normalizer.scale/offset 核对实际动作尺度。读取后再选受控改变，避免将整个配方同时更换并错误归因。

## 当前结论

### 195568 读取实际 checkpoint 配置

CPU 读取作业 COMPLETED，ExitCode 0:0；依据用户粘贴输出，实际配置确认：EMA=true，训练 seed=42，观测历史=2，数值 agent_pos=二维，动作=二维，轨迹 horizon=16，每次执行动作数=8，U-Net down_dims=[512,1024,2048]，100 步 DDPM，100 个训练时间步，cosine beta，fixed_small variance，clip_sample=true，prediction_type=epsilon。测试起始 seed=4300000，n_test=50，n_train=6，训练 batch_size=64。

training.num_epochs=3050 仅表示训练计划，不能当成这份选定 checkpoint 的实际完成轮数。下载的是文件名含 epoch=0500 的选定权重；精确更新次数仍未读取。不能将“20 epoch vs 3050 epoch”作为已验证解释。

这确认了作者成功策略同时使用历史观测、数值状态与 EMA；与当前 Bridge 单帧、无数值位姿条件、无 EMA 的方案并非只差扩散网络。仍不能认定其中任一项是唯一根因。

checkpoint_config.yaml 与 checkpoint_contract.json 已取回本地，动作尺度对照完成。model 与 ema_model 保存的所有 normalizer 参数完全一致。

作者 x/y 原始控制坐标范围分别为 [12,511]、[25,511]；保存的 scale 分别为 0.0040080161、0.0041152262，offset 为 -1.04809618、-1.10288072。按 `normalized=raw*scale+offset` 计算，两个坐标的训练范围均约为 [-1,1]；归一化后的标准差约 0.4072、0.3952。推理反变换为 `(normalized-offset)/scale`。

Bridge xyz 使用工具坐标系相对米制位移除以 0.10，未从训练样本拟合各轴 scale/offset。这是确定的配方差异，但不是已确定的错误。不能照搬二维像素坐标的数值去缩放三维米制位移，也不能直接把作者 `image*2-1` 的预处理套到冻结 CLIP 上。四元数的几何约束也不等同于二维控制坐标。

证据：reports/official_pusht_195527/normalizer_comparison.json。未修改现有标签、损失、模型或已有 checkpoint，未启动新训练。

## 接下来如何隔离一个因素

若继续训练实验，优先考虑同一次训练中保存普通模型与 EMA 模型，再用相同窗口、相同初始采样噪声分别评价。共享优化轨迹，保持 Bridge 划分、CLIP、Adapter、动作表示、归一化、U-Net、预测类型和预算不变。这样检验 EMA 对当前方案的影响，避免同时换归一化和目标函数后无法归因。

判读标准：若 EMA 相对本次普通权重在物理位姿误差上没有一致改善，就降低 EMA 的排查优先级；若改善，则只能说 EMA 对这份训练配方有帮助，不能说它解释所有失败。一个训练 seed 和多个采样 seed 只能构成初步受控证据，不证明训练重复稳定性。

2026-10-09 已实现并打包单因素 EMA 对照，尚未提交集群作业。复用原训练 runner 增加可选 shadow；普通权重与 EMA 权重共享同一优化轨迹，完整 1836/40 窗口分别评价一步 x0 与 100 步 DDPM，配对 CUDA 噪声流。已通过 EMA 数值更新、随机优化过程隔离测试，以及五个核心文件原版哈希和指标辅助模块导入检查。打包不含数据、CLIP 或 checkpoint；预计新增两份约 0.5 GB 的权重。操作见 scripts/cluster/bridge_ema_pair_v1/README.md，准备记录见 reports/bridge_ema_pair_preparation_20261009.json。未获得 EMA 改善结果。

我们已经验证一个完整扩散策略能在作者任务上取得高得分；仍未确定 Bridge 的唯一根因。现有公式核验、epsilon 对照和多尺度 U-Net 实验均应保留，不因官方案例成功而推翻，也不重复提交相同实验。

来源：作者 policy/diffusion_unet_hybrid_image_policy.py 的 compute_loss、predict_action、conditional_sample；model/common/normalizer.py；workspace/train_diffusion_unet_hybrid_workspace.py；eval.py。均按上述固定 commit 阅读。

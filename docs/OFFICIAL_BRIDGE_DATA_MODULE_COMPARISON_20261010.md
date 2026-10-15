# 官方复现与 Bridge：数据/监督模块逐项对照

官方依据：成功复现 195527 的真实 checkpoint_config.yaml、normalizer_comparison.json，以及固定提交 5ba07ac6661db573af695b419a7947ecb704690f 的源码。已将约 65 KB 源码保存到 baselines/diffusion_policy_data_reference/，逐文件 URL/哈希见 provenance.json。未下载数据或权重，未修改生产数据/模型，未启动训练。

## 结论

没有发现本次已检查的位姿转换往返或合成解析索引错误。确认了原生控制目标与未来到达位姿的监督语义差异、归一化差异、窗口边界差异和观测输入差异。它们不能被统称为代码故障，也不能一起更换后归因。

## 模块对照

| 项目 | 官方成功 Push-T | 当前 Bridge | 判读 |
|---|---|---|---|
| 动作目标 | 直接取记录的 action：二维绝对代理位置控制目标；环境 PD 控制器跟踪该目标 | xyz/姿态来自未来 state；转为当前工具系相对目标，夹爪来自 action 命令 | 你的位姿不是直接训练原生 action 前六维，这是确定的语义差异 |
| 观测时刻 | 16 步序列前两帧作为历史，第二帧为当前；输出执行索引1:9 | 图像时刻t，位姿state[t+1:t+17]，夹爪action[t:t+16] | +1是到达位姿定义，并不能因与官方索引不同就认定错位 |
| 窗口 | 每步采样，horizon16，pad_before1/pad_after7，边界重复 | stride4，选定Bridge协议只用16个真实未来状态，舍弃不完整尾部 | 样本密度和尾部事件权重不同；不是单纯改一行索引 |
| 位置归一化 | 各维 scale/offset，limits方式，存入模型并在推理反变换 | 每个轴统一除以0.10米，无各维统计偏置/尺度 | 可设计单独对照；不能照搬Push-T二维像素常数 |
| 姿态/夹爪 | 没有 | 单位xyzw四元数、独立夹爪头 | 不应把官方二维位置的归一化直接扩到四元数 |
| 输入状态 | 两帧RGB和数值agent_pos | 单帧RGB和语言；当前末端位姿用于构造标签，但不进入位姿context | 这是条件信息差异，留待条件模块；不可在归一化实验同时添加 |
| 图像预处理 | RGB/255再映射[-1,1]，96像素图像及84像素裁剪 | CLIP224预处理及专用mean/std | 编码器不同；保留CLIP就必须保留其输入约定 |

## 时间对齐的具体含义

令官方序列起点为 s，两个观测为 o_s,o_{s+1}，当前时刻 t=s+1。模型训练 action[s:s+16]，推理选择 action 序列的索引1:9，即 a_t 到 a_{t+7}。不要把整段 action_pred 的第一项当作官方当前执行动作。

项目当前观测是 o_t，第一项位姿是实测 p_{t+1} 相对 p_t 的变换，第一项夹爪是命令 g_t。接下来分别是 p_{t+2}/g_{t+1}，直到 p_{t+16}/g_{t+15}。这在“命令推动下一观测”的代码假设下自洽，但本轮合成检查无法验证 Bridge 真实时间戳、执行延迟或命令与到达轨迹的物理对应。

## 实际执行的检查

1. 调用了项目实际 `_get_tfrecord_bridge_state_action`，用20帧带明确编号的合成数据、start=2：图像/语言来自2，位姿来自3..18，夹爪命令来自2..17。当前连续夹爪测量按2*x-1传入，末步无效action按既定规则排除。原始action前六维使用哨兵值，确认不会被误当作位姿目标。
2. 1000组随机参考/目标位姿，实际 relative_pose_action→decode_relative_pose 往返最大位置绝对误差 1.49e-7 米，四元数符号不变的最大L2误差1.91e-7；另用绕z轴90°的已知几何实例检查工具坐标方向。不是仅对照同一公式的输出形状。
3. 执行官方实际 create_indices 函数（仅去掉JIT装饰器），20帧、horizon16、padding1/7得到13个窗口；项目完整未来16步、stride4只保留start=0。验证边界采样策略确实不同，不据此认定哪方错误。

结果：reports/bridge_data_contract_checks_20261010.json 与 reports/official_bridge_window_comparison_20261010.json。未新检查真实原始记录，不代表整个 dataloader 已获全面正确证明。

## 归一化的细节与控制变量

官方 PushTImageDataset.get_normalizer 直接取 replay_buffer['action'] 和 state[:,:2]，没有在此处按 train_mask 过滤；因此准确表述是“按该数据对象的整体 replay_buffer 拟合”，而不是“源码严格仅按训练划分拟合”。当前权重保存的x/y范围为[12,511]和[25,511]，映射至约[-1,1]，反变换使用相同scale/offset。

若在 Bridge 检验该因素，必须固定物理标签、划分、窗口、图像、模型、预测类型、优化预算和噪声种子；只在冻结训练窗口上计算xyz的轴向归一化参数，并把反变换和合法裁剪范围一起写入新checkpoint协议。开发/预留测试不参与拟合。四元数暂不变，旧权重不能直接改尺度推理。

注意：改变xyz尺度会同时改变MSE中的有效分量权重，实验结论应称为“位置归一化配方的影响”，不能声称仅改变数值范围而不影响目标权重。实施前先审计完整训练集各轴范围、标准差、常量维及异常值；没有这些统计时不直接训练。

## 代码位置

- 项目 code/dataset.py:128 相对编码、214反变换、950完整窗口筛选、1541 Bridge解析、1627未来状态、1632夹爪命令索引。
- 项目 code/models.py:241 get_context_vector 只接收图像/文字，未接收末端位姿。
- 官方 dataset/pusht_image_dataset.py:60 get_normalizer、73输入构造。
- 官方 policy/diffusion_unet_hybrid_image_policy.py:267反归一化、270执行窗口。
- 官方 common/sampler.py:create_indices/sample_sequence，model/common/normalizer.py:_fit/_normalize。

## 下一步

数据模块先选位置归一化作为可隔离候选，先统计冻结训练标签，暂不改成原生控制命令，也不添加历史或状态。命令/到达目标差异需要另立动作接口协议，不能塞进同一对照。归一化尚未被证明是根因；本次未准备新训练包。

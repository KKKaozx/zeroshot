# 当前实现与 Diffusion Policy 作者代码对照

日期：2026-10-06。只读对照，无模型修改、优化器更新、重新训练或仿真。

## 结论

当前模型使用的是自行简化的条件时间 U-Net，并非作者的完整网络。主要差异涉及观测信息、动作契约、网络容量、条件融合、夹爪建模和训练配置。它们值得受控比较，但不能仅凭源码差异确定泛化失败的根因。

当前余弦 DDPM 的加噪/反向采样与作者使用的算法家族一致，已有数学核验也支持当前 7 维分离夹爪路径。作者实现支持 epsilon 和 x0 两种目标，所以预测 x0 不等于偏离 DDPM 或改成普通回归。

已确认的 8 维掩码缺陷影响另一种可选配置，不影响最近完成的 7 维位姿加分离夹爪诊断。当前结果仍是训练拟合通过、开发泛化失败。

## 对照范围与版本

作者仓库：[real-stanford/diffusion_policy](https://github.com/real-stanford/diffusion_policy)。固定提交 `5ba07ac6661db573af695b419a7947ecb704690f`，本地参考副本位于被 Git 忽略的 `training_cache/references/diffusion_policy/`。

本次选择作者的 `DiffusionUnetImagePolicy`、`ConditionalUnet1D`、`train_diffusion_unet_image_workspace.yaml` 及其默认 `lift_image_abs` 任务。该任务是 Robomimic Lift，不是 Bridge、LIBERO 或 CLIPort。其他配置可以有不同数值，不能把以下设置推广到作者全部实验。

本地对照 `code/models.py`、`code/diffusion_decoder.py`、`code/dataset.py`、`code/train.py`，以及集群 186047 的冻结上下文源快照。对照起点为本地提交 `6c7514574b2558d7f2a6391aa129836b3beec2db`。相关文件指纹记录在 [source_comparison.json](../reports/source_comparison.json)。

## 逐项对照

| 项目 | 所选作者实现/配置 | 当前项目/最近诊断 | 分类与实际意义 |
|---|---|---|---|
| 观测 | 2 个观测时刻；默认任务有外部/腕部相机、EEF 位置/四元数、夹爪关节位置 | 单帧主相机、固定语言；当前夹爪测量仅进入夹爪分支；位姿头没有这些历史与状态输入 | 待验证：信息量不同，可能影响动作阶段判断；未证明加历史必然解决 |
| 视觉与语言 | 所选配置训练 ResNet18 图像编码器，无本项目 CLIP/语言 Adapter | 冻结 CLIP 图文编码；原上下文来自 2 层 Adapter；最新头部实验进一步冻结上下文 | 合理设计差异：作者策略不是目标语言方法的直接完整替代品 |
| 动作语义 | 所选绝对控制任务 10 维：位置 3 + rotation_6d 6 + 夹爪 1 | 到未来到达观测的工具坐标相对位置 3 + xyzw 四元数 4 + 二值夹爪 1 | 接口差异：不可直接比较误差、复制动作切片或接上同一控制器 |
| 归一化 | Dataset 提供可保存/反变换的 LinearNormalizer；绝对动作位置按数据统计范围缩放，其他维度按契约处理 | 位移除以固定 0.10 m；四元数单位化并约定符号；无对应数据统计范围的动作 normalizer | 合理设计差异：不能断言固定尺度错误；更换尺度意味着新训练/权重契约 |
| U-Net 容量 | 所选配置 down_dims=[512,1024,2048]，3 个下行层级、2 次降采样；每级两残差块，两个中间块 | 当前 hidden_dim=128，128→256 一次降采样，四个条件残差块 | 待验证：明显更小，但已能拟合完整训练，容量不足并非已确定根因 |
| 上采样与 skip | ConvTranspose1d；通道拼接 skip，再经残差块 | 线性插值 + Conv1d；skip 相加 | 合理设计差异：不是仅调用同一个网络的不同名字 |
| 残差卷积顺序 | Conv1d→GroupNorm→Mish | GroupNorm→Mish→Conv1d | 合理结构差异；不能仅凭顺序宣布错误 |
| 全局条件 | 时间嵌入与完整观测特征拼接后，传入各残差块 | 上下文经 MLP 压到 128 维，与时间 MLP 输出相加 | 待验证：当前有更强信息压缩；可以工作，不等于自动证明信息丢失导致失败 |
| FiLM | 所选配置 scale * hidden + bias；条件编码含 Mish | (1+scale) * hidden + shift；条件线性投影 | 合理参数化差异；需训练对照，不能机械判错 |
| 预测目标 | 源码支持 epsilon/sample；所选 YAML 默认 epsilon | 配对诊断已验证 x0/sample 更好，最新完整头部诊断使用 sample | 两者都属于 DDPM；小样本拟合改善不是泛化修复 |
| 加噪与调度 | diffusers 的 DDPMScheduler.add_noise/step；100 步余弦表，fixed_small 方差 | 自行实现同类公式，100 步余弦表及后验方差 | 当前路径未发现新的公式错误；这次是源码核对，未新做跨库逐位数值测试 |
| 去噪裁剪 | 所用 diffusers 0.11.1 的 clip_sample=True 对 x0 各维裁剪到 [-1,1]，然后反归一化 | x0 的 xyz 限制为 ±3（对应每轴 ±0.30 m），四元数分量 ±1，输出后再单位化 | 范围不同有动作尺度原因；不能直接换成库默认裁剪，否则限制成每轴 ±0.10 m |
| 夹爪 | 与其他动作维度一起在同一轨迹中扩散 | 分离分类头；训练时可看真实位姿，推理时看预测位姿；最新实验夹爪冻结 | 待验证：存在输入分布差异；真实位姿诊断也失败，不能只归因于位姿预测误差 |
| 损失 | 对目标轨迹作 MSE；mask 用于条件 inpainting，所选全局条件下动作不被条件固定 | 对有效监督维度作 MSE，并另加夹爪分类；当前 Bridge masks 全有效 | mask 含义不同；当前全有效 7 维均值形式不构成额外掩码解释 |
| 优化与 EMA | 所选配置 AdamW lr=1e-4、betas=(.95,.999)、wd=1e-6；warmup/cosine；使用 EMA | 最新诊断 lr=3e-4、wd=0、固定预算，无 EMA，冻结上下文 | 训练方案差异；不应把作者 8000 epochs 直接用于当前 25 条演示 |
| 在线执行 | horizon=16；从 n_obs_steps-1 取 8 个动作后重新观测规划 | 离线比较未来 16 个相对到达目标；尚未执行正式 benchmark | 当前没有任务成功率，预测时序与控制接口必须另行确定 |

作者源码入口（均固定提交）：

- [U-Net / FiLM / skip](https://github.com/real-stanford/diffusion_policy/blob/5ba07ac6661db573af695b419a7947ecb704690f/diffusion_policy/model/diffusion/conditional_unet1d.py)
- [图像策略、训练目标与采样](https://github.com/real-stanford/diffusion_policy/blob/5ba07ac6661db573af695b419a7947ecb704690f/diffusion_policy/policy/diffusion_unet_image_policy.py)
- [所选训练配置](https://github.com/real-stanford/diffusion_policy/blob/5ba07ac6661db573af695b419a7947ecb704690f/diffusion_policy/config/train_diffusion_unet_image_workspace.yaml)
- [所选任务和动作维度](https://github.com/real-stanford/diffusion_policy/blob/5ba07ac6661db573af695b419a7947ecb704690f/diffusion_policy/config/task/lift_image_abs.yaml)
- [Robomimic 数据转换与 normalizer](https://github.com/real-stanford/diffusion_policy/blob/5ba07ac6661db573af695b419a7947ecb704690f/diffusion_policy/dataset/robomimic_replay_image_dataset.py)
- [diffusers 0.11.1 调度器](https://github.com/huggingface/diffusers/blob/v0.11.1/src/diffusers/schedulers/scheduling_ddpm.py)

## 明确问题、已有证据及未确定项

**已有明确问题：** `code/train.py:385` 将 diffusion 的监督 mask 固定切成前 7 维。非分离 8 维输出会出现 8 对 7 的形状错误，已在旧审计实际触发。当前 separate_gripper_head=True 诊断不受此错误影响。本轮未修复，不用该可选路径启动新训练。

**有界偏差：** ±3 的位置裁剪使训练集 20 个位置目标不能完全达到真值。记录的整体位置误差下限约 0.00473 cm；开发集没有超界目标，因此它不能解释约 10.41 cm 的开发位置误差。平均训练通过也不能称作所有目标无误差。

**已有反证限制：** 当前较小 U-Net 能拟合完整训练窗口；回归头的开发验证此前也失败。因此“全部失败源于 DDPM”与证据不符，但 U-Net 的结构或训练方案仍可能影响泛化。

**不是本轮发现的新错误：** 单帧、较小网络、FiLM 参数化、skip 相加、无 EMA 都是设计/配置差异。没有进行新训练，不能给它们分配因果贡献。

## 下一次实验应如何收敛

先写清楚一个正式基线的数据、观测和动作接口，再做一次可审计的对照。若继续用现有缓存定位降噪器，候选实验为：在同一冻结上下文、同一 316/55 划分、同一 7 维位姿契约及固定夹爪头下，对照当前 denoiser 与作者 ConditionalUnet1D。保持 x0 目标、噪声表、裁剪、抽样清单、优化预算与评价范围相同；给作者模块传入该固定上下文作为 global_cond。

这属于“整套降噪器结构”的对照，会同时改变容量、条件注入和残差形式；不能包装成单因素层数消融，也不能声称复现作者完整策略或论文分数。作者大模型可能需要调整批量/预算，若不能匹配须明确记录，不能偷偷改设置。各结构使用明确的独立初始化种子；相同 seed 不意味着不同结构的参数相同。

若该对照仍能拟合训练但开发失败，应结束动作头排查，将下一阶段转向有新信息的观测/任务协议；不继续依次更换损失、增加轮数。若开发改善，也需新的独立任务/演示及正式训练种子验证。

作者原生多观测、多相机和 state 输入属于另一组实验，不与 denoiser 结构对照同时修改。已有当前位姿线性投影/patch 平均读出检查不是完整多帧观测实验，也不应重复运行。

正式训练必须从训练分区拟合所需统计量。作者所选 Robomimic 数据代码在 get_normalizer 中对完整 replay buffer 计算统计，未按 train_mask 切片；不能因为文件叫训练 dataset 就声称统计一定只来自训练演示，也不应无检查复制到本项目严格留出协议中。

## 本轮完成范围

阅读并固定作者代码版本；核对当前配置与已有报告；保存差异表和源码指纹。没有新增诊断脚本、修改模型、安装作者环境、下载训练集、启动优化器或进入仿真。当前结论是实施差异清楚了，泛化根因尚未锁定。

# 扩散实现审查：2026-10-09

结论：本次没有发现足以解释当前误差的确定性公式错误。发现了比较中的混杂因素、训练/推理输入差异，以及与论文实现的设计差异。不能把这些差异直接认定为根因，也不能宣布整个模块正确。没有修改生产模型、监督、采样或已有权重，没有提交新集群实验。

## 审查对应哪一份代码

`reports/bridge-unet-train-193761.json` 中 models.py、adapter.py、diffusion_decoder.py、dataset.py、train.py 五个 SHA256 均与当前 code/ 文件一致。195008 与193761训练报告、194978参考报告的来源哈希一致。审查不是拿修改后的代码推测旧作业。

## 公式与结构核对

| 项目 | 代码依据 | 核对结果和界限 |
| --- | --- | --- |
| 前向加噪 | models.py 的 diffusion_loss | `sqrt(alpha_bar)*target + sqrt(1-alpha_bar)*noise`；每样本随机t；符合官方策略的前向过程。 |
| 预测类型 | models.py 的 training_target；训练入口设为sample | x0分支监督干净动作，epsilon分支监督噪声；没有看到两种目标颠倒。官方策略也支持这两种分支。 |
| 余弦噪声表 | models.py 的 diffusion_betas | 按官方余弦定义独立计算100个beta，与本地实现最大绝对差0。 |
| 100步DDIM | evaluate_bridge_ddim_steps.py 的 ddim_sample | x0转换epsilon、eta=0更新与官方公式一致。先从未裁剪x0求epsilon，再裁剪x0，有官方 `use_clipped_model_output=False` 的对应写法；不能把未重算epsilon直接报成bug。 |
| DDPM均值与方差 | models.py 的 sample 与 posterior buffers | 静态核对符合相邻步后验系数及fixed-small方差形式。x0分支绕经epsilon再恢复x0，会产生浮点消减；本次随机toy t99误差最大0.000233归一化分量，未证明这是厘米级误差来源。没有执行完整真实模型DDPM对照。 |
| 多尺度U-Net | diffusion_decoder.py 的 MultiscaleConditionalUnet1D | 输入B×16×7转通道优先；三层256/512/1024，两次下采样、两次上采样、skip拼接，FiLM及时间/上下文拼接与官方布局对应。时间嵌入等配置不同，不是完整论文复现。 |
| 四元数 | dataset.py 的 normalise_quaternion/relative_pose_action；评价pose_errors | 目标单位化并固定w非负；旋转误差使用abs(dot)，q/-q等价。没有看到xyzw/wxyz混用或角度公式少乘2。w=0处仍存在半球边界，不能声称该表示没有学习难点。 |
| 位移单位 | POSITION_SCALE_METERS与评价POSITION_CM | 训练除0.10m，评价乘10得到cm；本分支单位一致。 |

本地 CPU 数值核验使用已有PyTorch环境，不加载模型权重。100步sample/epsilon两种toy输出，与独立转写的官方eta=0公式最大差均为0。**这是公式核验，不是运行diffusers包的端到端复现，不证明学习效果。**证据：`reports/diffusion_code_audit_20261009.json`。

少步采样使用本项目linspace取整时间表，不能默认等同于官方默认时间步间隔；一步分支还有明确保留旧诊断输出约束的特例。上述数值核验只覆盖100步，不把结果扩大到所有少步协议。

## 确认的差异及其含义

### 1. 回归对照不是只换动作头

`train.py:policy_loss`：回归为xyz MSE加单位化四元数的 `1-abs(dot)`；扩散sample为原始7分量MSE的均值。旋转几何、xyz/旋转相对权重、整体位姿损失尺度均不同；共同的0.25夹爪损失系数因此也不代表相同任务权重。

所以192651回归更好证明的是该完整训练配方更好，不能单独证明U-Net结构差、扩散不适用或Adapter正确。此前“同一条件下只比动作头”的表达应收窄为共享数据/初始化等条件，但位姿目标函数不同。

### 2. 夹爪存在真实轨迹到预测轨迹的输入变化

`models.py:diffusion_loss` 给夹爪头 `actions[..., :7]`（真实未来轨迹）；`sample`及独立评价把预测位姿送入夹爪头。部署时不能拿到真实未来轨迹，这是需要解释和单独验证的训练/推理差异。它可能影响夹爪，但不构成位姿头误差的直接证明，也未证明其解释全部切换失败。

### 3. 固定位移缩放不是官方的统计归一化

项目按0.10m统一缩放；没有根据训练集为每个维度拟合均值/范围。官方策略通过normalizer转换训练与采样输出，具体方式依任务而定。两种设计可以合法存在，但本项目不能称为照搬官方训练配方。

采样xyz裁剪到±3，即每坐标轴±30cm；训练没有对目标同样裁剪。代码存在可表达范围与目标范围不一致的风险，但当前证据没有计算超界目标比例，不能说真实数据已经违反范围或这就是根因。

### 4. Bridge监督和完整论文复现不是同一任务

`dataset.py:_get_tfrecord_bridge_state_action` 用未来观测state建立相对于输入时刻的到达位姿轨迹，夹爪取对应命令。它不是直接预测Bridge原生6维运动命令。当前末端位姿用于构造标签，没有作为数值条件送入位姿网络；网络主要靠单幅图像和语言解释相对运动。是否足够可观测还没有被证明。

官方U-Net布局对照不意味着已经复现官方数据、动作表示、观测历史、优化器、EMA或任务成功率。193761用batch2、20epoch、无EMA；所查官方示例含EMA和其他训练配置。不能把这些差异全改掉再将结果归于某一个原因。

### 5. CLIP使用的是项目自定义图像预处理

`dataset.py:prepare_image` 是双线性等比例缩放后均值填充；官方CLIP处理器默认bicubic、短边缩放、中心裁剪。两者不同，但当前数据多为方形，不能直接认定差异足以解释失败。当前Adapter以cls_patch_mean读出，patch信息有进入条件的路径；不能沿用旧CLS-only假设宣布patch分支失效。

## 195008能证明什么

固定40训练窗口，每任务8条演示的第一个窗口，三个采样种子。总体第一次x0估计8.355cm/14.405°，100步最终10.678cm/17.210°；五任务平均位置均增加。最后一步相对输入位置仅增加约0.052cm，旋转反而改善约1.018°。它支持整条链上的动作估计退化，不支持“最后一步破坏了全部结果”。

中间x0快照和链更新的约束不完全相同；不是已验证的提前停止控制策略。40窗口不是全部训练集，三个噪声种子不是三个独立训练重复。不存在真实抓取成功率结论。

## 当前决策

暂停新增训练和参数扫描。保留已确认的公式核验结果；修正结果解释，不能为了交付“根因”而随意改变公式。仍未确定根因。

后续若继续，应先选一个具有明确否证条件的问题：例如能否在少量固定训练窗口上从纯噪声稳定恢复动作（区分基本学习能力与数据规模），或真实/预测位姿对夹爪的独立影响。它们只列为待决策，不在本次生成新脚本或提交作业。未经同损失对照，不能将回归优势作为结构定论。

## 核对来源

- [Diffusion Policy训练目标与normalizer调用](https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/policy/diffusion_unet_lowdim_policy.py)
- [官方ConditionalUnet1D](https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/model/diffusion/conditional_unet1d.py)
- [Diffusers DDIM步骤、裁剪与epsilon处理](https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_ddim.py)
- [Diffusers DDPM后验系数和余弦beta](https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_ddpm.py)
- [Diffusion Policy示例配置](https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/config/train_diffusion_unet_lowdim_workspace.yaml)
- [统计归一化实现](https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/common/normalize_util.py)
- [CLIP图像处理器默认配置](https://github.com/huggingface/transformers/blob/main/src/transformers/models/clip/image_processing_clip.py)

来源于2026-10-09读取的main源码，链接内容会变化。本次是源代码审查和CPU公式核验，不是固定官方commit的完整论文复现。

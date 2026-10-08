# Bridge官方式Conditional U-Net资源预检

准备日期：2026-10-08。193390已结束当前紧凑U-Net的epsilon分支。下一项先检查官方Diffusion Policy风格的动作U-Net能否在现有A40/A6000资源内稳定前后向，不直接开始完整训练。

本预检新增可选的`MultiscaleConditionalUnet1D`，现有`ConditionalDiffusionDecoder`不改名、不改参数，也不加载或覆盖任何checkpoint。新结构使用三尺度`256/512/1024`通道、kernel 5、每尺度两个条件残差块、FiLM scale/bias、两次下采样及对称上采样。结构依据：

- [官方ConditionalUnet1D](https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/model/diffusion/conditional_unet1d.py)
- [官方图像DDIM配置](https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/config/train_diffusion_unet_ddim_hybrid_workspace.yaml)

GPU作业分别对现有紧凑解码器和新解码器执行4次batch 2前向、MSE反向、梯度裁剪和AdamW更新，输入形状固定为动作`[2,16,7]`、上下文`[2,1024]`。记录参数量、更新耗时、峰值显存、输出形状、权重变化和上下文梯度。显存门槛暂设32GiB，低于A40的40GiB强制申请值，给后续CLIP、Adapter和优化状态留出空间。

该作业只使用合成张量，不加载CLIP、Adapter、数据、旧权重或保留测试目标；通过只允许下一阶段接入完整模型做真实batch预检，不能说明精度、泛化或仿真成功。若解码器单独已超过32GiB或反向不稳定，则停止该结构，不进行完整训练。

入口：[预检程序](../code/preflight_official_unet.py)、[解码器](../code/diffusion_decoder.py)与[集群脚本](../scripts/cluster/bridge_official_unet_preflight.sh)。

本地CPU结构检查已完成：紧凑解码器6,120,455参数，官方式解码器77,670,023参数，约为12.69倍。新结构的输出形状、动作与上下文梯度、时间/条件依赖及非法序列长度检查共3项通过；既有Bridge相关19项测试全部通过。CPU检查没有运行完整尺寸反向传播，也不能替代GPU优化器显存测量。

## 193548资源结果

作业193548在A40上COMPLETED、0:0，用时24秒。两个解码器均完成4次AdamW更新，输出形状、权重变化和上下文梯度检查通过。紧凑解码器峰值保留显存0.137GiB、暖更新0.0105秒；官方式解码器1.566GiB、0.0281秒。新结构单独增加约1.43GiB保留显存，资源门槛通过。

这仍是合成上下文与动作，不代表完整策略能训练或精度改善。下一步只把新结构作为`diffusion_architecture=multiscale`可选项接入现有模型，保持默认`compact`和旧checkpoint兼容；在相同Bridge划分上运行6次真实batch更新，使用冻结CLIP、8层Adapter、独立夹爪头和x0目标，不进行开发评价或保存临时权重。只有完整链路的梯度、模块变化和显存均通过，才准备正式配对训练协议。

证据：[193548完整报告](../reports/bridge-unet-check-193548.json)。真实batch预检入口：[驱动](../code/run_bridge_official_unet_real_preflight.py)与[集群脚本](../scripts/cluster/bridge_official_unet_real_preflight.sh)。

## 193711真实batch结果与正式训练门槛

作业193711在A40上COMPLETED、0:0，用时31秒。完整模型557,761,928参数，其中可训练131,521,672、解码器77,670,023；峰值保留显存4.145GiB，暖更新0.1116秒。首步CLIP无梯度且六步后哈希不变；Adapter、解码器和夹爪头均有非零梯度并发生参数变化。真实Bridge输入、x0损失和优化链路通过，六步临时权重已丢弃，没有评价开发集。

正式训练因此获准，但仍保持结构单变量：从与192651相同的seed 42构造紧凑控制模型，逐项复制其初始Adapter与夹爪状态到多尺度模型；要求共享初始化哈希和20轮样本顺序哈希与192651一致。沿用1836个训练窗口、batch 2、20轮、18360次更新、AdamW、1e-4、余弦学习率、梯度裁剪1、x0监督。40个开发窗口只核对索引身份，本作业不读取其图像或动作目标，也不进行checkpoint选择。

训练完成后先保存最终权重、history、协议和报告，再由独立只读作业执行单步评价。这可避免评价脚本故障丢失训练成果。预计按预检暖更新时间约34分钟纯更新，作业额度90分钟；最终可训练权重约0.5GiB，只留集群，本地只取JSON。入口：[训练驱动](../code/run_bridge_official_unet_training.py)与[集群脚本](../scripts/cluster/bridge_official_unet_training.sh)。

证据：[193711完整报告](../reports/bridge-unet-real-193711.json)。该报告仍不包含精度或泛化结果。

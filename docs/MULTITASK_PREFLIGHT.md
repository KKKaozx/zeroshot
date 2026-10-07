# 五任务完整模型短预检

2026-10-07，包为 `training_cache/exports/multitask_preflight_v1.zip`，5,160,686 bytes（约4.9MiB）。只导出清单中每任务前两条训练演示的第0窗口，共10个；逐条检查原始payload指纹与此前验收一致。不依据动作大小或模型误差选样，不含开发/保留测试、模型权重或整批原始记录。生产代码、样本和脚本均记录SHA256。

复用现有 `RobotAdapterModel` 与 `train.policy_loss`，结构为冻结CLIP ViT-L/14、8层512注意力Adapter、CLS+patch均值、16步条件U-Net、100步余弦DDPM sample预测及独立夹爪头。8次batch=2的联合更新只用于检查各模块有限非零梯度、参数实际更新、CLIP不变及显存/速度；1e-4是预检学习率，不是已确定正式训练预算。不保存权重或宣称性能。

本地导出实跑及Python/bash语法检查通过。尝试加载本地CLIP时发现缓存只有配置、没有权重，模型尚未加载，未执行GPU更新。失败证据见 [本地报告](../reports/multitask_preflight_local.json)。先确认集群已有CLIP缓存与bridge-diffusion环境中的transformers，再运行GPU脚本；脚本禁止隐式下载/安装。若缺依赖，另准备隔离环境，不更改旧实验环境；若缺权重，只在集群按容量预算准备一份。

完整247窗口的训练manifest与主训练入口接入尚未完成；当前包只完成预检准备，不应当作正式训练包。候选清单保持未就绪，13条保留测试目标保持未读。

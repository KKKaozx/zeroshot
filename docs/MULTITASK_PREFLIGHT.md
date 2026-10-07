# 五任务完整模型短预检

最新集群盘点：bridge-diffusion有torch2.5.1+cu118、没有transformers，指定HF缓存路径未找到CLIP权重。已生成v2包（5,163,046 bytes），新增CPU安装/资源准备作业。项目内venv使用system-site-packages继承旧PyTorch，固定transformers5.17.0并约束torch版本，不修改旧环境。固定CLIP提交32bd64288804d66eefd0ccbe215aa642df71cc41，只下载safetensors及配置/tokenizer，排除PyTorch bin、TF、Flax重复权重；按HF文件元数据验大小，小于2GiB。下载前用200GB十进制规划限额、15GiB预留及2GiB依赖预算检查apparent占用，非实时配额查询。CPU加载通过后才运行8更新GPU预检。

v2安装和下载尚未在集群执行。安装脚本与GPU脚本bash语法检查通过，数据导出和源记录指纹核对通过；没有本地权重下载。上传v2 zip及sha256后，先提交setup_multitask_preflight.sh，取回日志验收，不直接启动正式训练。

2026-10-07，包为 `training_cache/exports/multitask_preflight_v1.zip`，5,160,686 bytes（约4.9MiB）。只导出清单中每任务前两条训练演示的第0窗口，共10个；逐条检查原始payload指纹与此前验收一致。不依据动作大小或模型误差选样，不含开发/保留测试、模型权重或整批原始记录。生产代码、样本和脚本均记录SHA256。

复用现有 `RobotAdapterModel` 与 `train.policy_loss`，结构为冻结CLIP ViT-L/14、8层512注意力Adapter、CLS+patch均值、16步条件U-Net、100步余弦DDPM sample预测及独立夹爪头。8次batch=2的联合更新只用于检查各模块有限非零梯度、参数实际更新、CLIP不变及显存/速度；1e-4是预检学习率，不是已确定正式训练预算。不保存权重或宣称性能。

本地导出实跑及Python/bash语法检查通过。尝试加载本地CLIP时发现缓存只有配置、没有权重，模型尚未加载，未执行GPU更新。失败证据见 [本地报告](../reports/multitask_preflight_local.json)。先确认集群已有CLIP缓存与bridge-diffusion环境中的transformers，再运行GPU脚本；脚本禁止隐式下载/安装。若缺依赖，另准备隔离环境，不更改旧实验环境；若缺权重，只在集群按容量预算准备一份。

完整247窗口的训练manifest与主训练入口接入尚未完成；当前包只完成预检准备，不应当作正式训练包。候选清单保持未就绪，13条保留测试目标保持未读。

# A40同卡配对骨干对照

用户已选择在A40上重新配对训练CLIP与DINO，各8192次更新，避免在新的对照里同时改变训练GPU型号。硬件通常首先影响计算速度、显存和数值实现；相同随机种子不保证跨设备/平台完全逐位复现。不能据此把大幅性能差异归因于显卡，也不能把CUDA尚未初始化的失败当作训练结果。官方依据：https://raw.githubusercontent.com/pytorch/pytorch/v2.5.1/docs/source/notes/randomness.rst 。

设计：一个Slurm作业申请一张A40，先CLIP后DINO，独立进程fresh seed42。两组保留原Bridge1836/40窗、405/10演示、同一8192×8日程、当前9D机械臂状态、CLS+patch读出、x0/sample目标、工具相对表示/裁剪3、原Adam阶梯LR、冻结夹爪与DDPM100/三采样seed。初始化与实际输入仍逐项对旧摘要核验；最终要求两组同allocation、同GPU、同初始参数、同目标/评价槽，独立复算全部原始预测，逐演示报告差异。

两份现有视觉缓存均在A6000编码：原CLIP缓存与228978通过的DINO缓存。两组复用完全相同的原文本特征；不在A40重新编码，不新增缓存，不重复16更新预检。作业在长训练前先做CUDA单元素检查和两份完整缓存payload检查。源预检228978的协议/原runner/helper/初始化和真实输入身份被显式绑定到新配对合同；新的helper身份按新合同声明，绝不伪造已在A40通过工程预检。原循环、损失、优化器、采样与度量代码保留；GPU与初始化/梯度/冻结检查在每组实际执行时继续强制。

旧A6000两次失败保留，均未训练。新CLIP训练是为这次同卡控制而获用户明确授权的额外对照，不是重新尝试旧拟合。旧207558和旧4.77cm留出结果只作历史背景，主效应来自新A40两组在原40开发窗的误差差。单训练seed、重复开发集和随机轨迹误差限制保留，不等于LIBERO或抓取成功。

本地23个Python文件编译、完整导入路径、继承SHA/原A6000包未改、真实输入loader AST和原context/check/advance函数AST一致、6个源准入/配对拒绝夹具、原NPZ复算及ZIP逐成员校验通过。没有真实模型/前向/参数更新/集群操作；仅合成验收数据用于测试守卫，未冒充A40结果。记录`reports/bridge_backbone_pair_a40_package_validation_20261027.json`。

新包`cluster_packages/bridge_backbone_pair_a40_v1.zip`为1,051,601字节，SHA256为077c2047628d8396d9418e5ec1e88af85b764afe564b45c1ced09247389785fc；无图片或模型权重。新最终权重合计约1.06GB，不生成17GB组合包。以此前A6000预检估计总2至3小时级，A40实际速度尚未测量，不含排队；wall上限4小时，实际步数每组仍固定8192，不自动扩展。

用户上传后在`/projects/Zeroshot`解压，进入`bridge_backbone_pair_a40_v1`执行`bash submit_once.sh`。提交前核实229016/229042均FAILED/NODE_FAIL；锁和marker防止重复。一组失败就停止，不自动循环或覆盖已完成组；保留输出再决定恢复。日志`logs/bridge-backbone-pair-a40-JOB_ID.out`，结果`runs/bridge-backbone-pair-a40-JOB_ID/pair_comparison.json`。其余主OXE/LIBERO批量工作仍暂停。

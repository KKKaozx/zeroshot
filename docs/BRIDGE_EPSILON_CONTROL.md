# Bridge epsilon训练目标配对对照

决策日期：2026-10-08。193087已证明当前x0预测权重在1至2步最好，4步起持续退化；继续调整该权重的采样步数不再是首要方向。

官方Diffusion Policy的图像DDIM配置使用100个训练扩散步、`squaredcos_cap_v2`、动作裁剪、epsilon预测和8步DDIM推理，并同时使用动作归一化、EMA、较大的Conditional U-Net与更长训练。官方来源：

- [DDIM图像策略配置](https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/config/train_diffusion_unet_ddim_hybrid_workspace.yaml)
- [训练目标与推理循环](https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/policy/diffusion_unet_image_policy.py)
- [真实机器人推理设置](https://github.com/real-stanford/diffusion_policy/blob/main/eval_real_robot.py)

本项目当前192651对照使用x0预测、20轮、batch size 2、无EMA和较小的单层下采样U-Net。两者差异很多，不能因官方8步成功就直接认为本项目8步也应成功；193087已经实证当前权重不成立。

下一项实验只改变`diffusion_prediction_type`：使用epsilon监督训练相同模型，固定以下项目不变：

- 相同1836个训练窗口、40个开发窗口与保留测试隔离；
- 相同随机初始化、20轮顺序、18360次更新、batch size 2；
- 相同冻结CLIP、8层Adapter、现有Conditional U-Net与独立夹爪头；
- 相同余弦噪声表、优化器、学习率和梯度裁剪；
- 开发评估使用8步确定性DDIM，同时保留静止、回归和当前x0一步结果作对照。

这样只能回答epsilon监督是否比当前x0监督更适配多步生成。若改善，再分别考虑EMA、网络容量和训练预算；若仍失败，不应一次性把这些因素全部加入。该实验尚未实现或运行，不把官方配置差异写成已解决问题。

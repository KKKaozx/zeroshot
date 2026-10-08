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

这样只能回答epsilon监督是否比当前x0监督更适配多步生成。若改善，再分别考虑EMA、网络容量和训练预算；若仍失败，不应一次性把这些因素全部加入。

独立驱动先按192651的确定性构造顺序重建共同初始状态，并要求Adapter/夹爪共同状态哈希、CLIP哈希、20轮批次顺序哈希与192651报告一致；正式训练必须读取完全相同协议且通过的6步GPU预检。正式结果固定使用最终第20轮，不用开发结果挑checkpoint。入口：[训练驱动](../code/run_bridge_epsilon_control.py)与[集群脚本](../scripts/cluster/bridge_epsilon_control.sh)。

## 193182预检与193195正式结果

193182六步GPU预检COMPLETED、0:0。193195正式训练COMPLETED、0:0，用时26分19秒，完成18360次更新；CLIP哈希不变、可训练参数改变，峰值保留显存2.80GiB。来源报告哈希全部匹配，没有读取保留测试目标。

8步DDIM三种子平均：训练13.689cm/23.501°、第16目标19.336cm/31.702°；开发13.108cm/21.549°、第16目标17.405cm/28.755°。开发终点夹爪73.3%，抓取时刻约1/20，释放平均约0.7/8。10条开发演示中，路径位置/旋转均0条优于x0一步或回归，仅位置2条、旋转1条优于静止。epsilon监督单独替换没有复现官方方案的优势。

后续193390冻结权重步数核对已完成。16、32、100步相对8步改善位置，但开发最佳仍为11.193cm/20.964°，差于静止8.212cm/15.098°、x0一步7.672cm/14.726°和回归6.626cm/12.108°。没有任何步数在任一开发演示的路径位置上超过x0一步或回归，因此当前epsilon分支正式停止。原始证据：[训练完整报告](../reports/bridge-epsilon-train-193195.json)、[训练核对摘要](../reports/bridge_epsilon_193195_verified_summary.json)及[步数核对](BRIDGE_EPSILON_STEP_SWEEP.md)。

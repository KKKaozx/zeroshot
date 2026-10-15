# 相近扩散机器人论文的学习记录

2026-10-09。范围：论文方法、实验协议及部分官方源码阅读；未复现论文结果，未安装环境、下载权重/数据或提交训练。

## 主参考：Scaling Up and Distilling Down

Huy Ha、Pete Florence、Shuran Song，CoRL 2023。[论文](https://arxiv.org/html/2307.14535v2)，[作者项目页](https://www.cs.columbia.edu/~huy/scalingup/)，[官方仓库](https://github.com/real-stanford/scalingup)。项目页旧columbia-ai-robotics仓库链接现重定向至real-stanford。

阅读第3.4节、策略学习对照及附录C。它将Diffusion Policy扩展为语言条件策略：CLIP文本、可训练双视角ResNet、末端状态历史作为条件，卷积扩散网络输出位姿与开合。动作使用3维位置、6维旋转及1维夹爪。论文的balance任务对照中，delta动作、空间池化、有本体状态条件下，MLP成功率29.0%，扩散69.5%；这是该论文任务内闭环结果，不是我们的Bridge泛化结果。论文报告训练50步、推理5步。

选择它的理由是语言条件加卷积动作扩散更接近当前目标，而不是认为它与冻结CLIP ViT加8层Adapter完全相同。

## 已核对的官方代码

固定官方提交：`7fb7891e9f1038ffb66a392b8458a2d836426884`。通过GitHub API读取源码与配置，未克隆仓库。以下链接固定该提交：

- [训练和推理](https://github.com/real-stanford/scalingup/blob/7fb7891e9f1038ffb66a392b8458a2d836426884/scalingup/algo/diffusion.py)：动作normalize后由scheduler.add_noise加噪，网络监督noise MSE；推理调用scheduler.step，再unnormalize；EMA随训练更新并用于推理。
- [核心配置](https://github.com/real-stanford/scalingup/blob/7fb7891e9f1038ffb66a392b8458a2d836426884/scalingup/config/algo/diffusion_core.yaml)：50训练步、5推理步；global_cond包含语言、视觉和本体状态；此文件本身还需Hydra配置组合，不能冒充完整实验配置。
- [6D旋转动作](https://github.com/real-stanford/scalingup/blob/7fb7891e9f1038ffb66a392b8458a2d836426884/scalingup/config/algo/diffusion_ee_control_6drot.yaml)：action_dim=10，upper_mat旋转。
- [观测/动作窗口默认配置](https://github.com/real-stanford/scalingup/blob/7fb7891e9f1038ffb66a392b8458a2d836426884/scalingup/config/algo/rollout_config/default.yaml)：16预测步，引用two_proprio、one_vision、four_actions配置。
- [调度器默认配置](https://github.com/real-stanford/scalingup/blob/7fb7891e9f1038ffb66a392b8458a2d836426884/scalingup/config/algo/noise_scheduler/default.yaml)：引用DDIM。

## 与当前项目的差异

| 环节 | 当前Bridge分支 | 主参考 |
| --- | --- | --- |
| 视觉条件 | 单幅图像，冻结CLIP ViT | 双相机、可训练ResNet |
| 语言融合 | 8层cross-attention Adapter后池化 | 视觉、语言与本体状态组合成global_cond |
| 本体状态 | 位姿网络没有数值末端位姿；当前开度只用于夹爪 | 末端状态历史进入动作生成条件 |
| 动作 | 输入时刻工具坐标系下未来到达位姿，3+4+1 | 末端控制序列，6D旋转配置为3+6+1；不可直接共用标签 |
| 监督 | 当前multiscale预测x0，7分量MSE；夹爪另用真实轨迹监督 | 所查代码预测noise MSE；动作维度由配置决定 |
| 归一化 | 位移固定除0.10m、单位四元数 | 独立action_norm_config，并在输出反归一化 |
| 推理权重 | 最终更新权重，无EMA | EMA权重 |
| 评价 | 离线轨迹、终点、开合与事件误差 | 闭环任务成功评价，也有离线诊断 |

推论：论文成功验证的是它的整套条件与控制方案，不能只拿U-Net层数或5步采样移植过来。以上差异是学习清单，不是已经确认的故障清单。不能直接换成6D旋转而继续称为老师指定的8维方案；需先核对课题约束。

## 辅助参考

### 3D Diffuser Actor

[论文](https://arxiv.org/html/2402.10885v2)，[官方代码](https://github.com/nickgkan/3d_diffuser_actor)。已读方法与官方Encoder、DiffuserActor：冻结视觉主干、语言注意力、RGB-D空间特征和末端状态条件；默认6D旋转，位置与旋转分别使用scaled-linear与cosine噪声表，预测epsilon。官方实现分别加权位置/旋转L1，夹爪在带噪轨迹特征上使用BCE。它启发我们检查空间条件和损失尺度，但不是U-Net方案，也不能用缺少深度的当前输入直接复现。论文CALVIN的A/B/C训练到D测试是场景泛化，不能等同于任意新任务零样本成功。

已核对：[Encoder](https://github.com/nickgkan/3d_diffuser_actor/blob/master/diffuser_actor/utils/encoder.py)，[动作扩散及损失](https://github.com/nickgkan/3d_diffuser_actor/blob/master/diffuser_actor/trajectory_optimization/diffuser_actor.py)。该辅助仓库阅读的是当日master，未固定commit，也未运行。

### MDT

[论文](https://arxiv.org/html/2407.05996v1)，[官方代码](https://github.com/intuitive-robots/mdt_policy)。论文在CALVIN和LIBERO讨论语言/图像目标条件，使用Transformer动作扩散及连续噪声形式，包含噪声依赖预条件skip和辅助任务。适合参考已有LIBERO数据与多模态条件设计；不是现有离散DDPM+U-Net的替换件。本次只读论文相关部分与仓库说明，没有审查完整MDT源码。

### Diffusion Policy

[作者项目页](https://diffusion-policy.cs.columbia.edu/)，[官方代码](https://github.com/real-stanford/diffusion_policy)。仍是离散动作扩散与U-Net的基础参考；其标准配置不等于冻结CLIP加语言Adapter，不能宣称本项目已经复现了它。

## 学习后如何收敛工作

先围绕主参考固定一份完整训练/推理配置，核对输入、动作语义、归一化、目标函数与输出控制的对应关系，再选一个可负担的最小运行对照。若运行参考方案，首先保存原始实现作为对照，再逐项加入项目要求的冻结CLIP和Adapter；不能一次更换归一化、旋转、损失与EMA后声称找到了根因。

复现验收必须区别：导入/形状通过，训练窗口可生成，独立演示评价，闭环任务成功。当前只完成文献与代码学习，尚未建立参考方案可运行性的证据。没有新增集群作业，本地只添加这份笔记。

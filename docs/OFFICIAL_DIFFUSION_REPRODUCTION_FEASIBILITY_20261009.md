# 官方扩散策略复现入口核查

核查日期：2026-10-09。此阶段只读取官方代码、配置与下载元数据；没有安装环境、下载权重、训练或执行仿真。

## Scaling Up：保留为语言条件结构参考

- 官方仓库：https://github.com/real-stanford/scalingup
- 固定代码版本：`7fb7891e9f1038ffb66a392b8458a2d836426884`。
- [官方复现说明](https://github.com/real-stanford/scalingup/blob/7fb7891e9f1038ffb66a392b8458a2d836426884/docs/reproduce.md)要求评估时传入自己的 checkpoint；训练前通过作者的数据生成流程产生轨迹。
- 查阅项目页、README、复现说明及 GitHub releases，未找到可直接下载的预训练扩散 checkpoint 或配套训练轨迹。这个结论仅限已核查入口，不意味着互联网上绝对不存在。
- walkthrough 提供的 `responses.zip` 是语言模型回答缓存，不是机器人策略权重或演示数据。
- 作者使用 MuJoCo。先前运行的原版 U-Net 单步更新只验证了该组件能够计算和反传，没有复现论文任务结果。

因此，暂不把它作为无需重训的成功率对照。它仍用于学习语言条件、动作表示、历史状态和扩散结构。

## 可直接执行的参考：Diffusion Policy 图像版 Push-T

选择依据：Scaling Up 官方 README 明确说明策略建立在 Diffusion Policy 的 Colab 实现之上；后者公开配置、日志、权重和任务评估入口。使用作者原版图像输入策略，避免把自己的 Bridge 数据处理或 Adapter 混进第一次复现。

- [官方仓库及使用说明](https://github.com/real-stanford/diffusion_policy)。固定代码版本：`5ba07ac6661db573af695b419a7947ecb704690f`。
- [完整实验配置](https://diffusion-policy.cs.columbia.edu/data/experiments/image/pusht/diffusion_policy_cnn/config.yaml)。CNN 图像编码、条件 U-Net、epsilon 预测、cosine 噪声表、100 步 DDPM、EMA、动作轨迹长度 16。
- [单个权重](https://diffusion-policy.cs.columbia.edu/data/experiments/image/pusht/diffusion_policy_cnn/train_0/checkpoints/epoch%3D0500-test_mean_score%3D0.884.ckpt)。HTTP HEAD 返回 200，Content-Length 为 **4,293,295,813 bytes，约 4.00 GiB**。不要递归下载整个实验目录。
- 文件名的 `0.884` 是作者记录的 mean score，不是我们复现得到的成功率，更不是 88.4% 抓取成功率。
- [原版 eval.py](https://github.com/real-stanford/diffusion_policy/blob/5ba07ac6661db573af695b419a7947ecb704690f/eval.py)加载 checkpoint 内配置、模型和 EMA，再调用原版 runner；此流程不调用 workspace 的训练方法，因而无需下载训练数据。
- 配置为 6 个训练初始化和 50 个测试初始化；测试起始 seed 为 100000，每个 episode 最多 300 步。保留作者协议，分别报告 train 与 test 指标。

作者原版评估命令形式（完成独立集群环境安装之后才能使用）：

```bash
python eval.py \
  --checkpoint /projects/Zeroshot/paper_reproduction/diffusion_policy/pusht_image.ckpt \
  --output_dir /projects/Zeroshot/runs/official-pusht-image-eval \
  --device cuda:0
```

这是原版入口的用法，当前尚未生成或验证集群安装脚本，也没有提交 Slurm 作业。

## 存储与兼容性

权重只下载到集群，本地仅保存配置、说明和结果摘要。独立环境、包缓存、权重和视频合计暂留 **20 GiB** 预算；这是规划额度，未实际安装测量。安装前重新查询项目占用，不能将之前的占用记录当成当前配额。

作者环境为 Python 3.9、PyTorch 1.12.1、CUDA toolkit 11.6、diffusers 0.11.1、gym 0.21.0 等旧版本。必须在项目目录建立独立环境，不覆盖已有 Bridge/Octo 环境。Push-T 使用 Pymunk/pygame，原版完整依赖文件还包含其他任务依赖；不能承诺集群无需兼容处理即可安装。

下一步先准备独立集群安装与评估作业。若必须变更依赖或 runner 配置，记录差异；不修改作者模型、训练目标或采样公式。不从头训练、不下载训练集。

## 这个实验能够说明什么

若复现成功：我们拥有一个在作者任务上实际完成闭环动作执行的扩散参考，可按训练加噪、噪声预测、归一化、调度器、EMA 和历史观测逐项对照自己的实现。

若执行失败：先根据 traceback 区分安装、加载、渲染和执行问题，不能归因于扩散理论。

Push-T 是二维推物体任务，没有 CLIP、语言指令、机器人夹爪或 Bridge 动作格式。因此成功也不能证明我们的 Adapter、夹爪、Bridge 泛化或三维抓取能力有效。它是论文方法的最小闭环结果复现，不是毕业项目最终实验，也不是完整论文复现。

# 官方预训练扩散策略闭环评估：195527

用户集群日志确认作业 COMPLETED，ExitCode 0:0。此次加载作者预训练权重，在作者的图像版 Push-T 环境实际执行闭环动作；未重训。完整 JSON 和六段视频已取回并核查，逐帧解码共 1380 帧，检查开始、中间和终止关键帧。

| 指标 | 日志结果 | 作者配置回合数 |
|---|---:|---:|
| train/mean_score | 0.7842293785 | 6 |
| test/mean_score | 0.9309259646 | 50 |

核对固定提交 `5ba07ac6661db573af695b419a7947ecb704690f` 中的原版 runner 和环境：每步 reward 为目标区域覆盖率除以 0.95 后裁剪到 [0,1]；每回合取最高 reward，最后对回合求均值。因此 0.9309 不是 93.09% 二元成功率，也不是未经缩放的平均覆盖率，更不是抓取成功率。

从完整 JSON 重算均值，与日志相符：50 个测试回合中 33 个满分（66%）、1 个零分；6 个 train 初始化中 3 个满分。满分率仍是 reward 达到上限的比例，不等同于从完整 done 日志核验的二元成功率。

实际测试 seed 为 4300000–4300049，train seed 为 0–5。之前网页配置的 test_start_seed=100000 与实际结果不同，不能将那份网页配置说成本次 checkpoint 中的完整配置。原版 eval.py 从 checkpoint 读取 cfg；如需逐项比较应另外取回其中保存的 cfg。

六段视频仅覆盖 4 个测试和 2 个 train 初始化，不代表所有 56 个回合。三段满分测试视频（4300001/2/3）的终止关键帧显示灰色物体与绿色目标基本重合；测试 4300000 得分 0.9917，300 帧视频中途接近目标、终止仍有偏差。train 0 得分 0.7112，300 帧终止画面仍偏离目标；train 1 为满分，终止画面基本对齐。这里只核查关键帧，未宣称逐帧评定碰撞、稳定性或物理误差。

本结果支持：预训练扩散策略在作者的匹配任务、观测、动作、归一化和调度器组合下能够执行并取得高任务得分。此前不是只有组件反传通过，现在已有完整闭环结果参考。

本结果不支持：自己的 CLIP、Adapter、Bridge dataloader 全部正确；自己的三维动作泛化成功；夹爪抓取有效；任何唯一根因已经确定。Push-T 不含语言和夹爪。只使用一个已训练权重，未复现从零训练与多个训练种子。

6 个 train 初始化与 50 个 test 初始化样本数不同，不能仅据 test 得分更高推断其泛化优于训练集。作者 checkpoint 已经经过测试表现筛选，本次评估也不是全新的、未被作者使用过的测试协议。文件名中 0.884 与本次 0.9309 的差异，不代表我们改进了算法。

下一步针对已有 Bridge 实现对照作者训练目标、归一化、采样调度、EMA 与历史观测；原版参考模型无需重新训练。已下载资料总计不到 250 KB；可靠关键帧图为 reports/official_pusht_195527/decoded_rollouts.png，数值重算为 verified_metrics.json。

原始代码入口：

- [作者 runner](https://github.com/real-stanford/diffusion_policy/blob/5ba07ac6661db573af695b419a7947ecb704690f/diffusion_policy/env_runner/pusht_image_runner.py)
- [作者环境 reward](https://github.com/real-stanford/diffusion_policy/blob/5ba07ac6661db573af695b419a7947ecb704690f/diffusion_policy/env/pusht/pusht_env.py)

集群结果：`/projects/Zeroshot/runs/official-pusht-eval-195527/eval_log.json`。

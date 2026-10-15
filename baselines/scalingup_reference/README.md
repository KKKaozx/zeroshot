# ScalingUp固定源码参考

固定提交 `7fb7891e9f1038ffb66a392b8458a2d836426884`，来源 https://github.com/real-stanford/scalingup 。保存的上游文件原样保留；文件SHA256、路径和永久链接见provenance.json。只保存阅读和核心运行所需的小型源码与配置，没有模型权重或数据。

## 本地已执行的范围

`verify_core.py`直接导入原版diffusion_utils.py；CPU执行10维动作、16步窗口的前向、反向和一次AdamW更新。官方通道256/512/1024、时间嵌入512；条件宽度32是合成占位，未使用真实图像/语言/机器人观测。输出[2,16,10]、条件梯度非零、参数发生变化。

在当前已有环境中可重复运行：

```powershell
& 'D:\miniconda\envs\zeroshot-gpu\python.exe' baselines/scalingup_reference/verify_core.py
```

证据：../../reports/scalingup_reference_core_20261009.json。没有执行scheduler、完整state encoder或闭环策略，不能将passed解释为拟合、泛化或任务成功。该核验无需集群，也不保存大checkpoint。

## 论文和仓库不能混为一谈

论文附录称语言投影512维；本提交state_sequence_encoder/default.yaml的投影是128维。medium U-Net配置和diffusion_core引用的默认值也需结合运行时Hydra覆盖，不能拿单个yaml推定作者所有实验。所存配置是部分配置，缺少完整依赖闭包，不是可直接启动论文训练的工程。原版diffusion.py已保存供训练/推理审查，本地核心核验不导入它。

当前仓库未在根目录发现许可证文件；不将它标成MIT或擅自将本地源码副本上传到项目GitHub。源代码链接和原创核验入口可以在研究记录中引用。

## 真数据运行前的输入契约

作者方案包含机器人状态历史、视觉/语言条件、动作normalizer与10维末端控制。当前Bridge分支用单帧图像与语言，工具相对到达位姿3+4+1。两者不能直接共用动作张量，也不能把10维输出覆盖老师规定的8维正式模型。

因此接下来的实际数据对照需要显式记录：沿用哪一份数据划分、当前状态历史如何取得、缺少的相机输入如何处理、控制动作和到达位姿的区别、旋转转换、训练集统计的normalize/unnormalize。这些适配完成前不提交训练。此前已有读取完整Bridge字段的检查，不能为重复验证shape再提交一轮GPU资源作业。

后续对照只服务于验证扩散学习链路，不取代老师要求的正式架构；不把从不同数据/表示/条件得到的成绩当成某单一模块的因果证明。

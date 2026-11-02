# 209420通过后：固定配对正式实验

用户提供209420完成日志和完整验证器输出：旧训练四窗上下文/预测均0差，两组各8次工程更新通过，峰值保留显存3.75GiB、暖更新0.172294秒、原估算2112.49秒。本地未读取其完整JSON；正式入口在集群调用原验证器，再核对实际报告。当前没有新泛化成绩，207558仍为主参照。

两组分别加载同一207558训练权重，请求投影归零，分别新建Adam；各4096新增更新，batch8、LR1e-5、不早停/选轮次。两组都有2048次K16和2048次K1，原固定抽样表不变。控制组K1用旧1836窗，候选组K1用3394窗；K16仍完全相同。K1是预测一个未来动作，仍100步去噪。CLIP/夹爪冻结。

复用旧与209420新增视觉缓存，不重解码/编码训练图像。运行时读取405条已审计记录的数值和原始文本，用冻结函数计算本机标签/状态，并核对参考；避免直接把Windows浮点参考当作Linux运行值。这是输入加载，不另做实验。

两份最终权重保存并加载校验后，才给新10演示/50完整窗打分。不重评旧40窗和已消费13演示。在同一新集合比较207558、控制、候选和不动；完整16步、三采样种子、演示等权为主，保留窗口等权、逐任务/演示/目标及终点指标，K1单独作辅助。包括辅助共750采样观测，实际预检速度修正预计约37分钟（不含排队），仍要求预算不超过两小时。

沿用既定四项合取采用条件：位置同时优于三参照、旋转不劣于三参照、同至少6演示位置同时胜控制/207558、窗口等权位置/旋转不劣控制/207558。失败保留207558并关闭分支，不加预算/种子。25条最终数据保持关闭；开发通过也不自动评最终集或部署。

本地新代码验证：误差单位、四元数符号/90度、计权、种子误差不抵消及退化四元数拒绝；采用门槛检查交集/窗口回退；实际训练函数AST用微型CPU模型跑8次更新，检查相同起点/新Adam/保存加载。实际输入函数AST仅移除CUDA搬运，验证3394训练数值/原始文本token路由（训练图像解码0），验证入口读取50开发画面、编码器使用CPU形状替身。无真实CLIP/策略/GPU训练或开发策略分数。Python语法/CLI、Bash语法和逐文件ZIP校验通过。

发布包：[bridge_prefix_formal_v1.zip](../cluster_packages/bridge_prefix_formal_v1.zip)，15,115字节，SHA256 `c7e25da759a3c49166637911752cdb108a0c5490962397960295170c80746825`。旧预检目录和ZIP未改；未发布草稿单独保留，不用于上传。证据：[代码检查](../reports/bridge_prefix_formal_preparation_checks_20261017.json)、[输入入口](../reports/bridge_prefix_formal_input_checks_20261017.json)、[发布清单](../reports/bridge_prefix_formal_release_20261017.json)。

本地PowerShell：

```powershell
scp "D:\ntu_related\dissertation\Zero_shot\cluster_packages\bridge_prefix_formal_v1.zip" zixiao005@10.97.216.128:/projects/Zeroshot/
```

集群：

```bash
cd /projects/Zeroshot
unzip -qn bridge_prefix_formal_v1.zip
bash bridge_prefix_formal_v1/submit_formal_once.sh
```

入口验证209420实际报告后才提交一个A6000作业，锁和已保存编号避免重复提交，不依赖可能过期的Slurm作业依赖。4CPU/40GiB/两小时，集群可能调整资源。无自动重试，不重跑预检或原8192次前缀。

日志：`/projects/Zeroshot/logs/bridge-prefix-formal-JOB.out`；结果：`/projects/Zeroshot/runs/bridge-prefix-formal-JOB/`。

完成后在集群复算（需当地两份大权重，日常不必下载它们）：

```bash
/projects/Zeroshot/envs/multitask-preflight-v1/bin/python \
  /projects/Zeroshot/bridge_prefix_formal_v1/verify_formal.py \
  /projects/Zeroshot/runs/bridge-prefix-formal-JOB
```

# 当前主入口 epsilon / x0 短预算配对

目的：回答“主入口的数据、CLS读出、无当前9D位姿和其他设置保持不变时，只把预测目标从epsilon换成x0是否改善生成误差”。231050只证明旧有效DINO+状态配方对目标敏感，不能替代当前主入口的证据。

## 固定设计

- seed42，两组各fresh 8,192更新，B8，沿用原145,812训练窗口的同一加权抽样前缀，共65,536输入呈现/组。不是重新挑一个更容易的训练子集。
- 原学习率时域仍65,536、warmup1,024、AdamW2e-4、WD1e-4；不把余弦压缩成8,192。两组均相同。
- 冻结DINOv2-L/14与CLIP文本，Adapter8/CLS、原多尺度U-Net、100步余弦扩散、source_cartesian/原时间近似与掩码、夹爪BCE.25全部保留。只改原生`diffusion_prediction_type`，x0内部键叫`sample`。
- 原65,536步模型没有8,192步检查点。因此必须新训一个相同短预算epsilon参照，不能用新短x0直接与旧完整模型作单变量结论。这不是重做旧Bridge窗口实验。
- 原始core文件逐字节保留。显式外部诊断协议承认8,192前缀，不篡改原完整训练协议；旧预检226133只绑定冻结资产/原初始化环境，不冒充x0性能预检。sample冻结profile仅有显式目标元数据派生，所有冻结张量/常量值与epsilon完全相同。

首更新前，两组都用同一张前向图重放原首批epsilon损失和梯度范数，逐值检查后清梯度，再执行各自目标的第一次真实更新；不额外优化、不重构噪声。原完整冻结身份必须精确匹配。仅已知两根派生常量表可按档案恢复，编码器和学习参数不会被修补。整段核对原批次/有效维度/LR，配对核对实际timestep、noise、noisy_actions全部流摘要；不同则停止。旧core守卫保留。

## 固定评价和停止规则

两组终点一次性原生DDPM100采样，三个固定采样seed1101/1102/1103。120个预先固定窗口：32个实际在此前缀见过的训练窗口（每来源8个）+88验证窗口（原64个不同Bridge验证组+其他三来源各8个旧锚点）。验证组与训练组不交叉。冻结特征只编码一次。分别加载各自实际训练的权重和其对应目标；不把旧epsilon权重改推理flag冒充x0模型。

按来源分别报告轨迹/终点生成位置、有效四元数旋转、不动基线。位置是原记录单位除0.1的归一化量，没有新增物理标定；Language Table只评价有效XY，没有Z/旋转监督，不伪造旋转误差。不合并四来源为一个物理厘米平均数。夹爪仍参加原训练，本轮不把位姿诊断称作完整抓取成功。

主读数为Bridge64等组均值、sample−epsilon差和组bootstrap95区间；三采样不是三个训练seed。不同目标的训练MSE不能直接横向评判。若x0的训练和验证生成均改善且超过不动，再讨论下一项资格检查；只有训练改善则关注泛化；若两者仍差，只能说该短预算未建立目标单独改动的优势，不能外推65,536步x0一定无效。任何结果都不自动延长、加seed、改结构、选择最佳中间点或恢复LIBERO大评估。

本实验预计合计80–100分钟，申请A40单卡3小时上限。保留两份完整8,192前缀Adam/RNG检查点（约3.2GB），预留8GiB；不得调用只适用于完整65,536训练的导出/删除逻辑。超过时间/身份检查失败保留输出，不自动重提。旧模型/记录完全不覆盖。

## 上传和提交

在本机PowerShell（小代码包，不含模型和数据）：

```powershell
scp "D:\ntu_related\dissertation\Zero_shot\cluster_packages\oxe_target_pair_short_v1.zip" zixiao005@10.97.216.128:/projects/Zeroshot/
```

在集群：

```bash
(
set -euo pipefail
cd /projects/Zeroshot
unzip -n oxe_target_pair_short_v1.zip
cd oxe_target_pair_short_v1
bash submit_once.sh
)
```

只提交一个配对作业，两个训练进程顺序运行后按已固定计划生成评价；不会提交其他作业。若目录已经存在，SHA256SUMS检查可阻止旧文件混入。marker阻止重复提交。

查看进度：

```bash
JOB=$(cat /projects/Zeroshot/oxe_target_pair_short_v1/job_id.txt)
tail -n 60 -f "/projects/Zeroshot/logs/oxe-target-short-${JOB}.out"
```

结束后验收同一个结果，不重复训练/推理：

```bash
JOB=$(cat /projects/Zeroshot/oxe_target_pair_short_v1/job_id.txt)
PYTHONNOUSERSITE=1 /projects/Zeroshot/envs/multitask-preflight-v1/bin/python -B \
  /projects/Zeroshot/oxe_target_pair_short_v1/verify.py \
  "/projects/Zeroshot/runs/oxe-target-short-${JOB}" --with-evaluation
```

结果目录含epsilon/sample的`arm_receipt.json`、`step_report.json`、`latest_step.pt`、有效插入源码；共同`training_pair_verification.json`、`evaluation/report.json`、`evaluation/predictions_and_targets.npz`、`pair_report.json`。下载报告无需打包两个大检查点。当地CPU夹具与源/数组检查不代表真实A40首批重放或训练已经通过；实际结果待用户手动提交后观察。

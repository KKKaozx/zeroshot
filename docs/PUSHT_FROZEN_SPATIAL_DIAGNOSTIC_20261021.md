# Push-T端点之后：冻结空间信息诊断

当前已知：同配方与84168次更新下，原生212158平均重叠奖励0.8459，项目前端215999为0.4455；这是整个前端家族的差距，不能单独归因于CLIP或Adapter。本诊断承接该结果，不再重复策略训练、在线接线或此前已关闭的Bridge分支。

## 本次回答的问题

冻结CLIP输出中是否容易读出当前T块的位置与朝向？这些信息经Adapter各层、全局池化、输出LayerNorm、1024→64投影后，线性可读性有何变化？用同一套读出方式读取已成功的原生RGB64作为参照，避免把诊断器自身不适用误认为模型无信息。

它只是一项定位工具，不能独立证明唯一根因，更不能视为已经改善泛化。

## 固定范围

| 项目 | 约定 |
| --- | --- |
| 权重 | 项目215999、原生212158固定端点EMA；全部冻结 |
| 读出器拟合数据 | 原来90条训练演示，每条均匀抽12帧，共1080帧 |
| 诊断留出 | 从116条非策略训练演示按固定SHA顺序选40条，每条12帧，共480帧 |
| 数据隔离 | 不跨演示混分；不读取策略开发20场景成绩和最终50场景 |
| 标签 | 同一帧state中的agent XY、block XY、block angle；不使用未来动作 |
| 读出器 | 独立线性岭回归；同一预定正则0.001，无超参数搜索；标准化仅使用拟合帧 |
| 指标 | 按演示等权的位置模拟像素误差、圆周角误差；不是厘米、重叠奖励或抓取成功率 |
| 训练范围 | 只拟合小型诊断读出器；无策略梯度、无CLIP更新、无新策略仿真 |

标签依据既有作者源快照：`pusht_env.py`的`_get_obs`为agent.position、block.position、block.angle模2π；512是模拟画布尺度。`pusht_image_dataset.py`也明确state前两维为agent、后三维为block pose。服务器运行时核对源文件与数据SHA，帧编号直接索引同一份图像和状态，不做时间偏移。

## 比较与限制

- CLIP全局池化、Adapter第1/4/8层同公式池化：比较相同1024维读出。
- CLIP patch与第8层patch：使用同样的四个固定空间矩（均值、横坐标、纵坐标、横纵乘积），各4096维，保留粗空间布局。
- 第8层池化前后LayerNorm、最终投影64维：检查信息的线性可读性变化。
- 原生RGB64、12×12原像素、agent XY、训练均值、固定打乱拟合标签：作为诊断校准。原生RGB特征不附加agent状态，并额外核查其不随输入agent坐标变化。

不同维数意味着不同读出能力，因此不能只比较1024和64维的误差就宣布投影存在逻辑错误。四个空间矩也会丢信息；读出器失败只说明这套固定线性读出没读好，不意味着CLIP中完全没有位置，也不意味着允许更换或解冻CLIP。单帧诊断不覆盖双帧时序理解和扩散解码器实际使用条件的能力。

诊断会保存所有预测及逐演示误差，给出预先约定比较的配对演示bootstrap区间。区间只描述这些演示，不代表多个独立训练种子的结论。不依照结果自动追加变体、种子、训练预算或访问最终集。

## 运行与交付

包：[pusht_spatial_diagnostic_v1.zip](../cluster_packages/pusht_spatial_diagnostic_v1.zip)。原训练包不修改，模型与大缓存只读，新产物仅写独立runs目录。

本地检查：[检查报告](../reports/pusht_spatial_diagnostic_checks_20261021.json)，包括数值读出、留出标签/特征不能改变拟合参数、整条演示隔离、角度环绕、被动钩子不改变原前向和冻结参数、Python3.9语法与Bash语法。它们是合成检查，未在本地加载真实端点，不能代替真实集群诊断。

上传（本机PowerShell）：

```powershell
scp "D:\ntu_related\dissertation\Zero_shot\cluster_packages\pusht_spatial_diagnostic_v1.zip" zixiao005@10.97.216.128:/projects/Zeroshot/
```

提交（集群终端）：

```bash
cd /projects/Zeroshot
unzip -n pusht_spatial_diagnostic_v1.zip
bash -n pusht_spatial_diagnostic_v1/diagnostic.sh && bash pusht_spatial_diagnostic_v1/submit_once.sh
```

只提交一个最长45分钟的诊断作业；45分钟是申请上限，不是实测预计时间。使用既有环境，无安装或下载步骤。

完成后：

```bash
DIAG=填入实际作业号
/projects/Zeroshot/envs/official-pusht-v1/bin/python /projects/Zeroshot/pusht_spatial_diagnostic_v1/verify_output.py \
  "/projects/Zeroshot/runs/pusht-spatial-diagnostic-$DIAG"
```

输出报告、预测数组、帧清单、固定24帧对齐样本和小型读出器。后续先查看这些证据再决定模块改动，不把读出器接到实际策略中。

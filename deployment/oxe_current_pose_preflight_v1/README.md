# 当前位姿条件对照：工程预检

目的：在当前主入口x0配方下，检验“当前末端位置/姿态是否能提供有效的动作条件”。232091发现幅度总体接近但方向对应弱，不能直接证明缺少位姿是根因。本包只做预检，不含正式训练入口或自动后续提交。

## 固定起点和变量

复用231297的`sample/latest_step.pt`（原8192终点、学习SHA9be1b13d…121fd、合同9cd86ab7…844ae），保留原core权重、两组Adam moments、原数据/抽样游标、原65536学习率时域和CPU/CUDA RNG。不是fresh训练或只加载权重后重置优化器。原完整检查点保留，不改原48文件或原冻结守卫；构造时沿用已知两根常量表精确恢复，原完整冻结SHA必须匹配。

两组都新增同一个**18→1024、无bias、18432参数**的零初始化线性分支，输出加到原图文context：

- `mask_control`：9个归一化位姿值置零，9个真实可用性mask保留。
- `real_pose`：9个真实当前位姿值与相同mask。

因此控制组也能通过mask学习偏置，不能称作原模型在续训后完全未变。两组容量、分支初始化、AdamW设置/额外冷参数组、梯度裁剪顺序均相同，唯一输入变量是位姿值。图像DINOv2-L/14与CLIP文本冻结，CLS读出、x0、标签坐标/尺度/时间、数据来源比例和夹爪损失都不改变。不添加source/robotID。

9个值为XYZ与旋转矩阵前两列（按列拼接，与旧Bridge状态支路几何一致）。只读原数字store中`episode_offset + 当前window.start`的实测状态，绝不读取未来状态作条件。XYZ均值/std仅用原145812个eligible训练窗的当前行，按观察维度统计并冻结；旋转列不做统计缩放。LT只提供XY，未测Z/旋转的值和mask为0；不把存储占位值当实测身份姿态。仍是原记录单位，未解决跨机器人单位/基座/TCP标定。

## 本次实际做什么

1. 提交前只核对旧包、232091结果和新包的小文件哈希与source终点收据。
2. GPU入口运行原`verify.arm`完整检查点守卫，再用原`restore_checkpoint`恢复core/Adam/plan/RNG。新状态分支保持在base模型之外，不伪装成原生checkpoint可恢复模型。
3. 用120个原固定窗口、1101/1102/1103三次采样重放原x0终点。真实/控制分支的零输出context均必须逐值等于原context；实际真实分支零端点预测必须与232091原数组bit-exact。控制分支依据同context/同sampler证明相等，不另做重复采样。该重放不是新增性能评价。
4. 每组执行8193、8194两次可丢弃更新。两组首批原core loss/梯度、批次、学习率和timestep/noise/noisy-action必须一致；新增真实值列须有非零梯度及更新，控制值列须严格保持0。
5. 每组将8193的完整core/分支/Adam/plan/schedule/RNG序列化进RAM，运行8194后恢复8193，再执行8194一次，所有状态摘要和loss/梯度必须逐值重现。总计**6次实际工程优化调用**，每组2个不同逻辑更新。预检学习权重丢弃，不迁移给正式模型。

新完整检查点只在RAM中序列化恢复，不写两份约1.6GB文件到磁盘；原source检查点不覆盖。提交申请A40/4CPU、40G（Slurm可能执行22G层级）、30分钟上限。原训练source为同一A40环境Torch2.5.1+cu118，源码/冻结/恢复/RNG环境有任何不一致就停止，不调大容差。新状态特征跨平台参考允许1e-6、统计量1e-12的数值比较，只针对新增特征的NumPy运算，不放宽模型/冻结/预测重放的逐值检查。

正式候选暂拟**从同一8192终点各续2048至10240**，原LR时域仍65536；约为以往从头配对成本的四分之一。具体资源由这次实际warm update估计，随后才发布正式代码。正式始终比较同预算mask_control与real_pose，并保留既有未变8192端点作为退化参照；不会把多训练2048带来的变化当作状态的单变量贡献。若不能恢复超过不动的基本能力，不能放行LIBERO。没有自动加预算/seed/窗口或后续任务。

## 上传与提交

本机PowerShell：

```powershell
scp "D:\ntu_related\dissertation\Zero_shot\cluster_packages\oxe_current_pose_preflight_v1.zip" zixiao005@10.97.216.128:/projects/Zeroshot/
```

集群仅提交预检：

```bash
(
set -euo pipefail
cd /projects/Zeroshot
unzip -n oxe_current_pose_preflight_v1.zip
cd oxe_current_pose_preflight_v1
bash submit_once.sh
)
```

运行后看日志：

```bash
JOB=$(cat /projects/Zeroshot/oxe_current_pose_preflight_v1/job_id.txt)
tail -n 60 -f "/projects/Zeroshot/logs/oxe-pose-preflight-${JOB}.out"
```

结束后只读回执：

```bash
JOB=$(cat /projects/Zeroshot/oxe_current_pose_preflight_v1/job_id.txt)
PYTHONNOUSERSITE=1 /projects/Zeroshot/envs/multitask-preflight-v1/bin/python -B \
 /projects/Zeroshot/oxe_current_pose_preflight_v1/verify_preflight.py \
 "/projects/Zeroshot/runs/oxe-pose-preflight-${JOB}"
```

本地8项CPU检查包含实际check-only入口、真实数字store/120特征与未来批次、当前行因果/缺失维度、旋转几何、零分支RNG、实际重放入口的CPU编码器/采样器替身，以及实际工程驱动两组6更新/原Adam迁移/序列化精确恢复。未加载真实编码器或使用CUDA；本地通过不冒充实际集群预检通过。此次包不含模型、图像或新大型缓存。

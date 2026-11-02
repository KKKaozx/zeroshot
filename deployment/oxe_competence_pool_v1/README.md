# 单组512窗训练能力诊断

只新增一组4,096次更新：四来源各128个不同演示组、每组一个窗口。
每来源包含原来固定评估的8个训练窗口，另120个按预先固定的随机数选取。
训练池合计512窗；在每个来源内部有放回抽样，平均每窗约63–65次。
起点是232587 real_pose的10,240步完整状态，终点14,336步。
233531 replacement的4,096步完整训练记录和端点预测直接作为对照，不重训。

模型、冻结编码器、当前位姿输入、x0目标、动作掩码、夹爪损失、三组Adam、
原生100步采样、逐步学习率、来源槽位和训练77token填充保持与对照一致。
改变的是训练池成员、多样性及由此带来的单窗重复曝光；不能单独归因于曝光次数。
旧控制与本组为不同作业分配；全程逐步比较原始噪声、时间步和模型随机流。

## 运行

在/projects/Zeroshot解压后执行：

```bash
cd /projects/Zeroshot/oxe_competence_pool_v1
bash submit_once.sh
JOB=$(cat job_id.txt)
tail -n 60 -f "/projects/Zeroshot/logs/oxe-competence-${JOB}.out"
```

A40，预计约26分钟（不含排队与启动）；Slurm上限35分钟，训练循环上限30分钟。
先CPU核对原始数据、计划和已完成对照的完整状态；再复现原始120窗预测及
对照首批次损失/梯度/噪声。探针仅反向传播，不调用optimizer.step。
训练中每2,048步原子保存参数、三组Adam、随机状态及完整实际轨迹。
不会自动追加预算或提交其他作业。

依赖现有oxe_sampling_pair_v1及其未修改的原有依赖、原始数据和DINO/CLIP权重。
必须保留runs/oxe-pose-pair-232587和runs/oxe-sampling-pair-233531，特别是
两组来源/对照latest_state.pt。包内不包含模型权重或图像，无需下载新数据。
启动要求项目预算内至少4GiB余量，检查依据为du与文件系统空余，并非账户quota查询。

## 输出与解释

```bash
ROOT=/projects/Zeroshot
JOB=$(cat "$ROOT/oxe_competence_pool_v1/job_id.txt")
cd "$ROOT/oxe_competence_pool_v1"
LD_LIBRARY_PATH="$ROOT/envs/multitask-preflight-v1/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  "$ROOT/envs/multitask-preflight-v1/bin/python" -B competence_verify.py \
  --root "$ROOT" "$ROOT/runs/oxe-competence-${JOB}"
```

只评估原来固定的32个已见训练锚点（每来源8窗）和88个开发窗口，三个固定采样种子。
不声称测完512窗。训练能力标准为每来源可用的位置/旋转轨迹及终点均值均优于不动，
并检查是否不差于旧有放回对照。Language Table只计可用XY，无伪造旋转/夹爪监督。
位置是源记录的归一化单位，没有新增厘米标定。
即使通过，也仅支持这些训练锚点的学习能力；不批准主模型、泛化或LIBERO评估。
未通过则报告结果，不自动延长、缩池、换损失、换来源或追加种子。

## 中断恢复

不重复运行submit_once.sh。先确认原作业已停止，再人工检查完整断点。
只有原作业因中断且已保存latest_state.pt，才可显式提交同一输出目录的恢复：

```bash
cd /projects/Zeroshot/oxe_competence_pool_v1
# ORIGINAL_JOB替换为本组最初作业号；先检查sacct确认没有运行/等待中的续跑
sbatch job.sh ORIGINAL_JOB
```

恢复仍以原有4,096步终点为限；完整断点内的合同、池身份、实际轨迹、Adam和RNG必须一致。
该命令不会绕过失败的资格检查。运行时间超限后的新分配需要另行明确安排。

本包交付前仅运行CPU替身集成/恢复/拒绝错误输入测试；未在本机执行实际DINO/CUDA训练。

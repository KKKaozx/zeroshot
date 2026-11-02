# 512窗单组诊断代码与运行

用户已选定这笔约26分钟的新预算。只准备和交付代码，尚未上传或提交集群。
原233531来源内轮转试验仍关闭；本次两侧都是有放回抽样，只新增512窗训练池的一组。

代码包：`cluster_packages/oxe_competence_pool_v1.zip`（约280KiB，13成员，无权重和图像）。
源码：`deployment/oxe_competence_pool_v1/`。验证记录：`reports/oxe_competence_pool_validation_20261029.json`。

## 上传与提交

本地PowerShell：

```powershell
scp "D:\ntu_related\dissertation\Zero_shot\cluster_packages\oxe_competence_pool_v1.zip" zixiao005@10.97.216.128:/projects/Zeroshot/
```

集群终端：

```bash
(
set -euo pipefail
cd /projects/Zeroshot
unzip -n oxe_competence_pool_v1.zip
cd oxe_competence_pool_v1
bash submit_once.sh
)
```

提交后查看同一任务：

```bash
JOB=$(cat /projects/Zeroshot/oxe_competence_pool_v1/job_id.txt)
tail -n 60 -f "/projects/Zeroshot/logs/oxe-competence-${JOB}.out"
```

约26分钟估算不含排队和部分启动核对；A40作业上限35分钟，训练循环上限30分钟。
预提交先CPU核对源状态、原对照完整状态和512窗实际标签/计划，并检查保守4GiB空间余量。
若缺少依赖或状态不一致会在提交前停止；不要删除源232587或对照233531的完整状态。

## 训练和端点

从232587 real_pose第10,240步原始模型、位姿端口、三组Adam和RNG恢复，
保持与233531有放回组同一来源槽位、实际Linux逐步LR、77token、所有网络与监督，
只新增4,096步到14,336。四来源各128不同训练组的一窗，512窗均会被抽到；
每窗新增42–95次，均值约63–65次。选样事先冻结，不用误差或未来标签挑选。
源池145,812窗不删改；减少池多样性与增加单窗曝光同时发生，不能单独归因曝光。

GPU上首先复现源模型原120窗×3种子预测，与232587及233531存档逐值核对。
再做0优化探针，复现旧控制首批次损失/梯度/噪声/RNG，检查新批次原始噪声/t/RNG；
新组标签不同，不要求含标签的noisy_actions相同。全4096次更新与旧实际轨迹核对随机流。
复用原已测试训练、保存与恢复引擎；每2048步原子保存完整独立断点。

端点仍只评估原32个已见训练锚点和88个开发窗口，不声称评价整个512池。
按来源报告四项均值、配对训练组区间、位置幅度与方向；LT仅有效XY。
可用轨迹/终点位置与旋转均值全优于不动，并不差于旧控制，才支持这些锚点的有限训练能力。
位置是源记录归一化单位，不是新厘米标定；区间不包含训练种子不确定性。
无论结果如何，都不自动加预算、变池、换来源/模块、追加种子或放行LIBERO。

## 本地验证边界

5项新增CPU检查通过：真实512窗/掩码/隔离、完整控制mmap与真正check-only入口、
单组主入口与原120窗输入/原生100步采样/输出核验、半程Adam/RNG精确恢复与错误拒绝、
真实LT掩码及训练门槛/幅度方向。集成测试使用短4步和CPU替身编码器/解码器；
完整4096行控制元数据夹具仅验证状态校验，不冒称实际训练4096次。
另有8项原引擎/实际源加载器回归通过，包含已知Linux LR差异修复及原生512步CPU夹具。
包内Python编译、SHA清单和ZIP每成员逐字节核验通过；Shell只做静态头/LF核对，未执行本地Bash。
没有执行真实冻结编码器、CUDA推理/训练、远端提交或清理。

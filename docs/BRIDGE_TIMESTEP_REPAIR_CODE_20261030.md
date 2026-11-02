# 源端联合能力：时间步损失修复代码与固定预算

**后续用户已选择保持229116有效DINO位姿、补独立夹爪；本候选停用留档，不能按下方旧命令提交。当前路线见 `DINO_GRIPPER_SIDECAR_CODE_20261030.md` 和 `configs/active_capability_extension_20261030.json`。**

2026-10-30，待用户审阅预算，未提交。参数核查支线已结束；用户同意的单位/名义频率/工程迁移约定继续适用，未知物理标定不作为这次源端训练候选的启动障碍。

## 现在交付什么

已实现 `code/bridge_timestep_repair.py` 和独立部署 `deployment/bridge_timestep_repair_v1/`：完整两组训练入口、权重/源身份绑定、批次计划、断点、端点评价/验证、防重复提交、Slurm脚本和命令。小ZIP约2.42MB，不复制0.526GB源模型、编码器或数据，复用集群已有234100和资产。历史部署不修改。

具体候选是x0损失的**时间步权重**，不是再扫位置/旋转通道系数。使用[Min-SNR作者x0实现](https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/main/guided_diffusion/gaussian_diffusion.py)的`min(SNR,5)`形状，再除固定100步网格的平均值；固定均值归一化是披露的项目变体，不是逐批自适应或开发集拟合。目的是改变不同噪声阶段在现有x0目标中的相对作用，尝试改善已有局部重建偏差。不能从该图像领域方法或235095结果推断机器人已修复、或唯一根因已找到。

对照与候选都从234100完整65536端点的base+pose-port起跑，恢复源训练统计；两组Adam都从空状态开始，共同常LR1e-5。与原234100相比这是有意的权重微调新阶段；**不是原Adam/LR日程无缝续训**。两组间仅时间步损失权重不同。原始单位/动作标签、H16名义5Hz、冻结DINO/CLIP文本、架构、18D状态端口、夹爪loss和100步采样保持。

## 固定预算

| 内容 | 冻结安排 |
|---|---|
| 训练池 | 原Bridge10352窗/1151演示，不缩池、不添目标域监督 |
| 两组预算 | 每组8192更新、batch8，总16384更新，顺序占用一张A40 |
| 抽样 | 固定seed20261030计划，两组同一批次；65536次呈现/组，实际覆盖10332/10352窗，另20窗本计划未抽到，不说每窗都见过 |
| 训练随机性 | 两组共同训练seed42，每更新匹配indices/LR/t/加噪输入/反向后RNG；不是两个独立训练seed |
| 时间 | 按234100约0.329秒/更新外推，纯更新约90分钟，预计总95–110分钟；Slurm2小时硬上限，父进程6900秒，每组训练累计3000秒上限 |
| 状态/空间 | 每2048原子latest，含Adam/抽样/日程/RNG；每组一份latest和一个final，预留8GiB，空间失败不自动清理 |
| 评价 | 只在8192终点评价原128训练组+64开发组，各1101/1102/1103采样；旧234100数组直接复用，不重新采样旧结论 |

启动重复loss/梯度守卫与正式更新在同一作业内，不另占一轮GPU预检。停止/异常保留证据，不自动续作业、续训、换seed或扫gamma。此方案现在没有任何集群执行授权或远端操作。

## 看什么结果才算有效

两个人群均先过原能力门槛：四项轨迹/终点位置/旋转均值优于不动，位置轨迹配对组bootstrap95上界<0。另要求候选两项旋转均值优于同期原损失控制，开发轨迹旋转candidate-control的配对组CI上界<0；两个位置均值在训练及开发都不得比234100恶化超过5%。完整报告所有指标及未达项，不以训练loss、一次采样或一项改善代替联合能力。

失败则本候选结束，不自动增步/改表示/换权重。通过仅支持源端下一阶段，不自动解除空间/5Hz接触问题或放行论文13模型。小数据/语言工具和新增无更新诊断不接着展开。

## 验证边界和操作

3项CPU检查已通过：公式/归一化、尾部mask与独立梯度；实际冻结`models.diffusion_loss`及`train.policy_loss`接线，原`bcap_core`启动守卫、两组各3次toy优化和完整两权重组/Adam落盘；随机流配对和篡改拒绝。没有真实DINO/A40/新采样/物理仿真或真实训练。包内同名测试验证隔离导入，不能把重复运行计成更多测试或模型结果。

包内README包含完整命令：[上传、准入、唯一提交、tail和小结果下载](../deployment/bridge_timestep_repair_v1/README.md)。核心命令也列于下方，**仅供审阅后由用户手动执行**。

本机PowerShell：

```powershell
scp "D:\ntu_related\dissertation\Zero_shot\cluster_packages\bridge_timestep_repair_v1_20261030.zip" "zixiao005@10.97.216.128:/projects/Zeroshot/"
```

集群终端，首次解压/校验/提交：

```bash
cd /projects/Zeroshot
test ! -e bridge_timestep_repair_v1
unzip -n bridge_timestep_repair_v1_20261030.zip
cd bridge_timestep_repair_v1
sha256sum -c SHA256SUMS
bash submit_once.sh
```

`submit_once.sh`先运行CPU身份准入，再提交唯一2小时作业，打印`BRIDGE TIMESTEP REPAIR: JOB`，保存job_id锁。已有目录/标记时先查看状态，不删除绕过保护。日志：

```bash
JOB=$(cat /projects/Zeroshot/bridge_timestep_repair_v1/job_id.txt)
tail -n 60 -F /projects/Zeroshot/logs/bridge-timestep-repair-${JOB}.out
```

本次实际解除的是“没有可执行的源端修复候选与明确预算”的准备缺项；**B1性能尚未解除**，不能把代码包完成算成论文主模型结果。

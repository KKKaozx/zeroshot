# 2026-10-31待审阅：恢复未执行的源端时间步损失候选

此前v1因转去完成夹爪而搁置，从未提交或取得训练结果；本轮没有重跑已关闭实验。最新236840是独立LIBERO监督正对照，不能据其5/5认定OXE只差一个接口，也不能当此方法有效的因果证据。

下一项只针对B1：保留234100已学位置，尝试改善同一模型的旋转/联合生成。两组仍为原损失与固定Min-SNR5均值归一化；每组8192×8、同234100权重、新空AdamW、LR1e-5。学习输出仍8维未来相对位姿，原标签、源统计、18D当前位姿、100步采样与执行器均未改；不加载LIBERO监督模型。监督数据仅完整现有Bridge子池10352窗/1151演示。

v1r1仅补提交前完整CPU依赖导入、原绑定数值/原图文件/源状态准入，避免作业开始后才发现缺模块或材料。vendor原字节、损失实现、批次计划、验收门槛均保留；这是运行准备修订，不是第二种算法。本地依赖检查不是远端Torch2.5.1/A40已通过的证明，不执行集群或真实模型。

固定单A40预计95–110分钟，硬2小时；总16384更新，无自动续训或扫参数，预留8GiB。结束只保存固定端点、全状态/Adam与原预测，按原门槛报告。控制若独立改善也如实保留；本候选通过不自动准入LIBERO或13模型，失败即收口。源码和预算供用户审阅，助手不SSH、不提交。

以下为沿用并更新路径的完整方法、预算和手动命令。

# Bridge当前配方：单一时间步损失修复候选，待审阅

2026-10-30。代码已准备，未提交训练。物理参数不再作为本源端候选的前置审批条件；沿用用户同意的原记录单位按米、名义5Hz和现有标签。未知空间标定作为解释边界保留。

目的：直接尝试改善B1源端位置/旋转联合能力。234100训练位置有优势而旋转仍弱；235095已有低噪声局部重建偏差，复用这些证据，不再跑诊断。选定一个尚未验证的干预：x0训练的时间步权重。不是宣布根因已找到，不重复epsilon/x0、位姿、骨干、旋转通道权重、池大小或5Hz插值扫描。

候选采用 `min(SNR(t),5)`，再除以固定100步网格的均值；控制组权重1。Min-SNR的x0公式见[作者实现](https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/main/guided_diffusion/gaussian_diffusion.py)和[ICCV2023论文](https://openaccess.thecvf.com/content/ICCV2023/html/Hang_Efficient_Diffusion_Training_via_Min-SNR_Weighting_Strategy_ICCV_2023_paper.html)。固定网格均值归一化是本项目明确披露的变体，目的是保留与夹爪BCE的期望量级；不按batch或开发集调整。图像生成证据不证明机器人有效。

两组各从已有234100的65536步base+pose-port权重起跑，恢复源训练统计和冻结编码器身份；**两组都新建空Adam**，不是原Adam无缝续训。共同常LR1e-5、batch8、完整当前Bridge子池10352窗/1151演示，共同批次和训练随机流，各新增8192步。两组唯一受控差别为时间步权重；这不等于相对原234100的所有设置只差一项。权重原样保留，不覆盖历史部署。

沿用冻结DINO+CLIP文本、原Adapter/扩散/夹爪架构、x0、18D当前位姿、H16、100步采样、原mask/夹爪损失。源轨迹/归一化/执行接口不改，不使用LIBERO监督。

## 固定预算和验收

- 单个A40作业内顺序完成两组，共16384更新，预计95–110分钟；Slurm硬上限2小时，父进程6900秒，每组训练累计上限3000秒。
- 每2048步原子保存最新完整Adam/计划/RNG；每组只留一份latest和一个final，额外空间预留8GiB。不自动清理旧模型，不自动续作业/续训/换seed/扫gamma。
- 只评价固定8192终点：复用原128训练组+64开发组、采样seed1101/1102/1103。234100预测直接复用旧数组，不重新采样；3次采样不是3个训练seed。
- 每个更新核对两组的批次/LR/t/加噪输入/反向后RNG一致。启动守卫只在同一作业内检查无更新重放，失效即停止，无单独GPU预检作业。
- 候选两个人群均须通过原联合门槛（四项位姿均值优于不动，位置轨迹配对组CI上界<0）；两项旋转均值优于同期控制，开发轨迹旋转candidate-control的95%组CI上界<0；两项位置均值在训练和开发均不能比234100恶化超过5%。所有指标仍完整报告，不只看一个数。
- 未通过则此固定候选收口，不自动增步、改权重、缩池或恢复语言工具。通过仅支持源端下一阶段，不自动放行13模型或正式LIBERO。

本地3项CPU检查已覆盖权重公式、尾部mask及梯度；实际冻结model.diffusion_loss+train.policy_loss接线、两组各3步toy优化和完整Adam/两组权重落盘；逐步随机流匹配/篡改拒绝。使用小型替身网络，**没有加载真实DINO或执行真实训练**。真实A40/原始图像运行仍须作业内完成；Shell只检查LF/shebang，未声称本机执行bash。

## 手动上传、准入、提交和日志

以下命令供用户审阅固定预算后执行。助手没有执行SSH或sbatch。

本机PowerShell上传（压缩包不含原数据/大编码器/完整源权重，使用集群现存资产）：

```powershell
scp "D:\ntu_related\dissertation\Zero_shot\cluster_packages\bridge_timestep_repair_v1r1_20261031.zip" "zixiao005@10.97.216.128:/projects/Zeroshot/"
```

集群登录终端，首次部署和CPU身份准入：

```bash
cd /projects/Zeroshot
test ! -e bridge_timestep_repair_v1r1
unzip -n bridge_timestep_repair_v1r1_20261031.zip
cd bridge_timestep_repair_v1r1
sha256sum -c SHA256SUMS
/projects/Zeroshot/envs/multitask-preflight-v1/bin/python -B run_pair.py --root /projects/Zeroshot --check-only
```

审阅后唯一提交入口（有job_id标记及锁，不重复提交）：

```bash
cd /projects/Zeroshot/bridge_timestep_repair_v1r1
bash submit_once.sh
```

日志：

```bash
JOB=$(cat /projects/Zeroshot/bridge_timestep_repair_v1r1/job_id.txt)
tail -n 60 -F /projects/Zeroshot/logs/bridge-timestep-repair-v1r1-${JOB}.out
```

完成后同一终端验证已保存结果，无新模型前向：

```bash
/projects/Zeroshot/envs/multitask-preflight-v1/bin/python -B /projects/Zeroshot/bridge_timestep_repair_v1r1/verify_results.py /projects/Zeroshot/runs/bridge-timestep-repair-v1r1-${JOB}
```

本机只取小结果，按实际JOB替换：

```powershell
scp "zixiao005@10.97.216.128:/projects/Zeroshot/runs/bridge-timestep-repair-v1r1-JOB/small_results.zip" "D:\ntu_related\dissertation\Zero_shot\reports\"
```

若CPU身份/版本/空间或GPU守卫失败，保留具体错误，不降守卫或自动提交。目录已存在时先看已有job_id和状态，不删除标记绕过。输出模型使用独立repair格式，不能冒充原始65536端点或论文13模型之一。

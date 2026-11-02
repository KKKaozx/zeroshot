# 229116相对位姿输入配对候选

2026-10-31。**已准备代码和材料，未提交训练，尚无新能力提升结论。** 保留原229116位姿及235928夹爪文件。本轮针对当前绝对位姿跨机器人迁移的问题，不重开夹爪、骨干、旧拟合或插值实验。

## 唯一干预与实际实现

原策略以源训练均值/标准差归一化绝对XYZ，并输入绝对姿态两列。上轮LIBERO初始Z经该公式约为+17.7标准差；这是在已披露工程坐标约定下的输入事实，不是唯一失败根因的证明。

同一229116第8192步权重分为两组，全部初始参数一致、两组均空Adam：

- absolute：原9D绝对状态，原始RPY公式及保留168窗特征字节完全一致。
- relative：`R0.T @ (p_t-p0) / old_xyz_std`与`R0.T @ R_t`的前两列。源训练使用演示index0已观测位姿，部署使用本回合第一个实际观测；参考固定，回合切换必须显式reset。沿用源训练XYZ标准差，不拟合LIBERO统计。

只改当前状态表示和它所需的因果初态参考。原图像、文本、工具相对/右相对H16动作标签、100步x0 DDPM、夹爪输入/权重、名义时距和执行规则不变。初态参考是新增可用的过去信息，本试验不把它包装成“只证明了某个单一世界坐标错误”。同一旧权重对新输入可能初始不适应，固定2048步失败也不代表所有相对表示都无效。

`code/episode_relative_pose_state.py`提供公共特征和固定参考；`pose_core.py`完整恢复adapter/diffusion/history/state四模块，直接继承归档原生loss/sample，使用原冻结DINO/CLIP文本token缓存，不加载编码器训练。原生sample最后的夹爪占位列被丢弃，不训练/评价夹爪。`online_policy.py`按新checkpoint格式恢复并附加保留旧头；调用者必须先`policy.begin_episode(observation)`，然后接现有`bind_panda_measured_observation`和`execute_policy_chunk`。旧权重不能直接重命名为相对输入模型，默认不加载未准入夹爪，真实后续工程检查仍需显式工程模式。

保留旧夹爪不等于其与新上下文已经兼容；本轮不再训练夹爪。相对表示消除了共同刚性世界坐标变换的影响，不解决工具轴/TCP、相机、接触时序、初态分布差异或任务泛化。没有新增LIBERO监督训练。

## 固定预算、评价与停止

- 原1836训练窗/405演示，开发40窗/10演示；同一原批次计划的前2048批，batch8，两组各新增2048更新，总4096。所有模型/优化器RNG按同一update显式重置，常LR1e-5、Adam/wd0、clip1。它是空Adam的配对微调，不是原优化器精确续训。
- 更新adapter/扩散/history/state的副本。原检查点及旧夹爪前后SHA必须相同；编码器更新0、夹爪更新0、图像重编码0。不得覆盖原文件。
- 一张A40，预计20–35分钟，Slurm硬上限60分钟，配对主体/验证打包3300秒alarm；该alarm在CPU文件/cache准入之后启动，整个作业仍受60分钟限制。依据229116实际每步约0.1419秒：4096步约9.7分钟，其余为固定采样、I/O和保存；不是完成时间保证。预计6–8GiB显存，24GiB主机内存，输出/原子保存预留8GiB，`du`仅是配额估算，不自动清理。
- 每512保存完整四模块+Adam/计划/RNG，保留两组终点及最后状态。超时/信号/非有限/身份或原生恢复守卫失败就结束；不续步、换seed/阈值、放宽容差、重建cache或自动重试。
- 仅训练原train池；原128个固定训练锚点和40开发窗各3个采样draw1101/1102/1103评价。3draw不是3训练seed。source原预测直接复用。absolute开始前只做2窗/1draw的实现恢复守卫（上下文绝对差≤1e-5、轨迹分量绝对差≤1e-4），不是重新开展旧能力实验；失败不自动改容差。
- 新候选的前瞻规则：relative的train128/dev40按演示平均的位置/旋转误差分别不超过原229116相应值的1.05倍；开发两项也不超过同期absolute的1.05倍。满足只允许审阅一次有限工程回合，不自动运行、不等于显著改善，更不算正式准入。失败保留结果，本候选结束。
- 保留relative切换输入后的零更新预测、两组终点原预测/标签/行号和完整指标，能区分输入切换冲击与训练后结果。原夹爪F1门槛不改，不将其设为本轮源位姿试验的无限前置项。13模型和大评估不自动放行。

## 本地已验证与未验证

现有211个分片只读提取初始/当前位姿，原payload、1876目标/当前测量/编码图像身份绑定；没有解码或编码图像，准备约70.7秒。绝对控制保留168窗状态特征与原SHA完全一致。5项CPU检查通过，包含刚性变换不变性、参考复制/显式reset、新旧格式拒绝，以及两组各2步合成CPU更新、原生100步采样与Adam读回；这些不是策略成绩。真实229116的132,309,639个可训练参数已在新缓存训练核心中CPU完整恢复，无真实前向或更新。原加载器3项回归通过。

本机Torch2.11+cu128，实际集群仍需Torch2.5.1+cu118/A40。未SSH，未核查此刻远端cache余量/存在性，未真实训练/新仿真。提交前CPU准入核对已有文件和冻结缓存，失败会在申请GPU前给出具体文件；没有自动下载或重编码退路。原生CUDA恢复守卫及真实优化结果留待该唯一作业，不声称本地合成检查已经覆盖。

依赖已有文件：

```text
/projects/Zeroshot/runs/bridge-backbone-pair-a40-229116/dino/final_trainable.pt
/projects/Zeroshot/runs/dino-gripper-sidecar-235928/gripper.pt
/projects/Zeroshot/runs/bridge-dino-only-cache-v1/full_tokens.pt
/projects/Zeroshot/envs/multitask-preflight-v1
```

包不含大模型、原数据、冻结特征缓存或新环境。后续在线加载另复用原`dino_gripper_sidecar_v1`和编码器资产；本轮训练不需要再上传它们。

## 用户审阅后手动上传与唯一提交

本机PowerShell：

```powershell
scp "D:\ntu_related\dissertation\Zero_shot\cluster_packages\episode_relative_pose_pair_v1_20261031.zip" "zixiao005@10.97.216.128:/projects/Zeroshot/"
```

集群终端：

```bash
cd /projects/Zeroshot
test ! -e episode_relative_pose_pair_v1
unzip -n episode_relative_pose_pair_v1_20261031.zip
cd episode_relative_pose_pair_v1
sha256sum -c SHA256SUMS
bash submit_once.sh
```

唯一编号与日志：

```bash
JOB=$(cat /projects/Zeroshot/episode_relative_pose_pair_v1/job_id.txt)
tail -n 60 -F /projects/Zeroshot/logs/episode-relative-pair-${JOB}.out
```

若只想先核对已有依赖、不申请GPU：

```bash
/projects/Zeroshot/envs/multitask-preflight-v1/bin/python -B /projects/Zeroshot/episode_relative_pose_pair_v1/train_pair.py --root /projects/Zeroshot --check-only
```

`job_id.txt`及输出目录存在就拒绝重提，保留锁和标记，不删除绕过。完成后`small_results.zip`只含指标、原预测和回执，不含约529MB/组的完整模型或Adam：

```powershell
scp "zixiao005@10.97.216.128:/projects/Zeroshot/runs/episode-relative-pair-JOB/small_results.zip" "D:\ntu_related\dissertation\Zero_shot\reports\episode_relative_pair_results_20261031\small_results.zip"
```

把JOB换成真实编号；本机接收目录已创建。先审阅该原结果，再决定是否取回候选`relative/policy.pt`，不自动运行下一项。

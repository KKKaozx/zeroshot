# 保持229116位姿策略：独立夹爪补全，待审阅

2026-10-30。当前已完成代码和训练材料准备，**未提交真实训练或新仿真**。Min-SNR候选留档，本次不执行。原PyBullet抓取3/3与229116 DINO离线位姿改善作为独立既有成果保留，不重新运行或拼接成一个模型的成绩。

## 本次改变与保护

只训练一个独立`GripperSidecar`，约27万参数，读取原1024D上下文、**原模型已生成的**H16x7位姿、原连续实测开度输入`2*x-1`，输出16个绝对开正命令。结构沿用原轨迹条件夹爪的投影+时间Embedding+LayerNorm/Mish/Linear，不重新训练DINO/Adapter/扩散/9D状态端口，不覆盖229116检查点。源位姿模型在夹爪训练中根本不加载，优化器只持有独立夹爪参数。

训练数据复用已保存的229116的1836训练窗/405演示、40开发窗/10演示，3个既有采样draw1101/1102/1103的生成轨迹和原eval上下文。不是用专家未来位姿作为部署输入。1876条连续实测夹爪输入已从已有记录提取并与原逐行SHA一致；没有重编码图像、生成新位姿或重训旧实验。打包输入9.54MB，不需约2GB视觉token缓存或新源数据下载。

输入语义保持原样：**连续实测开度，不是上一条命令；9D状态，不是234100的18D端口；工具坐标/右相对旋转，不是直接当世界左相对轨迹。** 新回调显式处理这些区别。原XYZ统计来自旧128训练窗，原样恢复，不用LIBERO重拟合。

组合推理调用原模型`sample`一次，只替换第8列夹爪命令，前7列逐元素保持。CPU检查实际调用原100步采样方法，比较独立夹爪更新前后同输入/同seed的7D轨迹和采样后RNG完全相等。它证明的是**相同观测输入下的接口保护**；新夹爪会改变物理接触及之后的观测，不能保证完整闭环轨迹仍相同或一定抓取成功。

## 固定训练预算与停止条件

- 一个训练seed42，2048次更新，batch64，Adam常LR1e-4、wd0、clip1；无早停选权重/扫seed/扫结构。
- 每批固定随机抽取训练窗口及一个既有生成draw，131072次呈现；不使用开发窗口训练。3个生成draw不是3次独立训练。
- 估计1–4分钟，单A40 Slurm硬上限10分钟；进程480秒上限，每512步保存独立头+Adam/计划/RNG。不创建位姿模型副本，输出/原子写预留256MiB。
- 只在2048终点评价：开/闭平均召回率即balanced accuracy≥0.80，严格优于始终开/始终关/当前实测阈值保持三个基线的最佳值；窗口内部开、闭切换的精确步F1均≥0.50。两个事件类别若无真值事件则不准入。完整报告训练/开发、每类召回、事件TP/FP/FN与整窗一致率，不用多数类准确率冒充会夹取。
- “当前实测阈值保持”只是预测基线，不把实测宽度冒充既知历史命令；不评价当前→第一个未来步的切换，只评价未来16步内部切换。
- 失败则本候选停止，不自动续训、换头、改门槛或运行仿真。通过只解除本源端夹爪离线准入项，不算LIBERO/13模型成绩。

## 本地验证边界

3项CPU检查已通过：原100步采样的7D输出/RNG/冻结权重保护，独立头一次synthetic更新；原9D状态与连续测量、工具右相对→目标世界左相对几何；事件指标与同一观测绑定。没有加载真实229116权重或DINO、没有真实数据训练/实际LIBERO运行。源模型权重本地尚未下载，集群训练直接复用现存路径并核对SHA，缺失则在提交前明确报文件路径。

完整策略加载器和一次有界工程回合入口已经提供，实际加载和仿真仍未验收；不得把接口代码存在写成闭环通过。

## 用户审阅后手动上传与提交

本机PowerShell：

```powershell
scp "D:\ntu_related\dissertation\Zero_shot\cluster_packages\dino_gripper_sidecar_v1_20261030.zip" "zixiao005@10.97.216.128:/projects/Zeroshot/"
```

集群终端，首次部署和唯一提交（脚本先CPU身份准入，再请求10分钟A40；不另开预检作业）：

```bash
cd /projects/Zeroshot
test ! -e dino_gripper_sidecar_v1
unzip -n dino_gripper_sidecar_v1_20261030.zip
cd dino_gripper_sidecar_v1
sha256sum -c SHA256SUMS
bash submit_once.sh
```

脚本打印`DINO GRIPPER SIDECAR: JOB`并保存job_id，已有标记不重复提交。日志：

```bash
JOB=$(cat /projects/Zeroshot/dino_gripper_sidecar_v1/job_id.txt)
tail -n 60 -F /projects/Zeroshot/logs/dino-gripper-sidecar-${JOB}.out
```

完成后的已保存结果复核，无位姿模型采样：

```bash
/projects/Zeroshot/envs/multitask-preflight-v1/bin/python -B /projects/Zeroshot/dino_gripper_sidecar_v1/verify_head.py /projects/Zeroshot/runs/dino-gripper-sidecar-${JOB}
```

只取小结果（按实际JOB替换），供审阅后续是否准入：

```powershell
scp "zixiao005@10.97.216.128:/projects/Zeroshot/runs/dino-gripper-sidecar-JOB/small_results.zip" "D:\ntu_related\dissertation\Zero_shot\reports\"
```

## 已准备但不自动执行的LIBERO接线

`load_policy.py`严格绑定229116原模型源码/配置、9D状态统计、源完整权重SHA和通过离线门槛的独立夹爪。`legacy_dino_gripper_execution.py`从相同观测读取Panda两指关节；以`sum(abs(qpos))/.08`作为向源开度比例迁移的**工程约定**，不宣称物理标定。显式给定的空间约定仍为s1/A=C=I/零原点与TCP差；不因未知参数继续停工，也不把它们改写成测量事实。

旧工具相对输出先依当前源姿态转换为源左相对位移/旋转，再应用显式空间映射；原生20Hz/每4tick重规划不变，没有追加插值/控制扫描。头离线通过后，才可手动运行**一个**已排除LIBERO90工程任务，最多200tick/50调用/含加载600秒，不读取专家动作或未来观测，无中途状态纠正。

本机取回原229116位姿模型（约529MB，原文件保持，SHA应为`730f06ecb417ff98d7e5270c114c2b062ce38347f0987c136b9d0359e8c4a6f5`）并从small_results.zip提取gripper.pt。该下载不是训练夹爪的前提：

```powershell
scp "zixiao005@10.97.216.128:/projects/Zeroshot/runs/bridge-backbone-pair-a40-229116/dino/final_trainable.pt" "D:\ntu_related\dissertation\Zero_shot\reports\bridge_backbone_pair_a40_229116\cluster\extracted\dino\final_trainable.pt"
```

明确选定通过的head文件后，本地入口先不带`--run`只列文件状态；只有显式追加`--run`才加载完整模型/启动固定回合。命令中的HEAD应替换实际gripper.pt路径：

```powershell
Set-Location "D:\ntu_related\dissertation\Zero_shot"
$py = "D:\ntu_related\dissertation\Zero_shot\training_cache\libero_native_env_v1\Scripts\python.exe"
& $py -B deployment/dino_gripper_sidecar_v1/run_engineering.py --root "D:\ntu_related\dissertation\Zero_shot" --head "HEAD" --source "D:\ntu_related\dissertation\Zero_shot\reports\bridge_backbone_pair_a40_229116\cluster\extracted\dino\final_trainable.pt"
```

输出固定`reports/dino229116_sidecar_libero_engineering_20261030/run1`，已存在即拦截，不删除绕过。此任务结果仍不能当正式40任务成功率；没有自动后续训练或仿真。

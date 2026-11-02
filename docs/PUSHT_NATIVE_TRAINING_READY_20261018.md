# Push-T在线接线已通过，原生2万次训练入口

211193完整报告已本地读取，SHA256为9030306a0e10c498659322ad3203336aeebce5555c5817dece0be58acca82942。
CLIP全部参数冻结且前后权重未变；缓存/在线的文本、原批8视觉、归一化往返补批视觉、Adapter条件、DDPM100动作五项差异均0；真实环境2次决策/16步通过。策略未训练，0次更新，没有奖励或泛化结论。此预检收尾，不再重复。

下一步是从头训练**作者原生组**，先建立可用的闭环参照，再用相同训练预算对比冻结CLIP+项目Adapter组。不是直接再跑195527预训练权重，也不是同时启动一天半训练。

## 锁定的第一阶段

- 使用既有90训练演示、10726窗口，16步动作/2帧观察/每次执行8步；11356训练帧单独拟合normalizer。使用已有float32数据，原图只除255一次，不安装或下载。
- 作者原生策略、完整[512,1024,2048]U-Net、原compute_loss/epsilon/cosineDDPM100、AdamW及EMA。相同新初始化后端和train-only统计的SHA必须匹配210763。
- 有效batch64，microbatch8。每个打乱轮次所有训练窗恰好出现一次，最后38样本按实际样本数加权。明确固定打乱顺序与独立于前端的扩散噪声/时间步；仍直接调用作者compute_loss，不另造扩散目标。
- 保留原计划3050轮对应512400次更新的cosine学习率时域、warmup500；2万是阶段检查点，不把学习率提前压到0。确定性计算也会用于后续项目组。
- 入口只训练20000次；不开放自动延长或项目组训练。原预算上限84168仅保留在协议中，后续若需要接续，另行检查20k结果后准备。
- 启动时审查一次原损失/计划噪声及梯度，保存第1步后重放第2步，要求raw/EMA/Adam/LR/EMA计数/全部RNG及loss完全相同。额外重算1次仅作恢复自检，不计已完成训练预算；通过即继续，不再另交预检。
- 每2000次原子保存latest.pt及SHA侧车；约4.2GB完整状态，需要15GiB余量。恢复必须显式指定该文件、核对协议/样本计划/训练原图内容及检查点SHA，不重新初始化或重训已完成前缀。

## 泛化评价

只评估恰好2万次后的EMA，固定20个新开发初始场景5000000..5000019；原作者legacy图像环境、[多步执行包装器](https://github.com/real-stanford/diffusion_policy/blob/5ba07ac6661db573af695b419a7947ecb704690f/diffusion_policy/gym_util/multistep_wrapper.py)、最多300步、DDPM100。每次决策采样seed固定，后续项目组沿用。

报告每回合最大归一化重叠奖励的平均、满奖励回合比例及置信区间，同时保存逐步奖励便于CPU复算。只保存前两个固定种子的视频，不根据成功与否挑视频。原生参照门槛预定为平均奖励至少0.8且满奖励至少50%；这是本次操作门槛，不是论文规定，更不是机器人抓取率。

若原生20k达标，再准备同20k预算项目组；若未达标，不归罪Adapter，不自动训练项目组。最终50种子封存，Bridge207558及最终25不动，CLIP硬冻结。

## 本地验证与实际边界

CPU小模型直接提取原作者compute_loss AST、使用真实einops及原EMA实现，验证目标/代表参数梯度、前端多消耗随机数后的噪声不变、磁盘恢复所有状态精确重放、协议变更拒绝、全窗覆盖/尾批计权/边界补帧，以及逐步奖励复算与坏摘要拒绝。Python3.9 AST、CLI、Bash语法和ZIP/SHA检查通过。

本地没有运行真实原生大模型的这条完整训练入口，也未评分任何开发或最终回合；其真实GPU启动检查将在任务开始时执行。既有资源和在线预检不重复；没有远程提交作业。

## 上传与提交

包：[pusht_native_training_v1.zip](../cluster_packages/pusht_native_training_v1.zip)。机器协议见包内training_protocol.json，发布记录见[release](../reports/pusht_native_training_release_20261018.json)。单A6000/4CPU/22GB主存，预计纯训练约2.8小时，保存和评价另计，作业上限8小时。

Windows PowerShell：

```powershell
scp "D:\ntu_related\dissertation\Zero_shot\cluster_packages\pusht_native_training_v1.zip" zixiao005@10.97.216.128:/projects/Zeroshot/
```

集群终端：

```bash
cd /projects/Zeroshot
if [ ! -e pusht_native_training_v1 ]; then
  /projects/Zeroshot/envs/official-pusht-v1/bin/python -m zipfile -e pusht_native_training_v1.zip .
  PUSHT_NATIVE=$(sbatch --parsable /projects/Zeroshot/pusht_native_training_v1/train.sh)
  echo "Push-T原生2万次训练：$PUSHT_NATIVE"
else
  echo '目录已存在，请先检查已有作业，避免重复提交。'
fi
```

查看：

```bash
sacct -j "$PUSHT_NATIVE" --format=JobID,State,ExitCode,Elapsed,NodeList
tail -n 30 "/projects/Zeroshot/logs/pusht-native20k-$PUSHT_NATIVE.out"
```

结束后用包内verify_training.py对report执行CPU核验：核对checkpoint SHA/完整启动自检、逐步奖励复算和两个视频存在性；不加载模型、不开新作业。报告执行通过与原生能力门槛是两个不同结论。

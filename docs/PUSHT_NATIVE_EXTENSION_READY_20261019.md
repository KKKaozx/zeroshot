# 原生20k已核验，按预先上限接续至84168

211309完整小工件已本地读取，逐步奖励独立汇总仍为0.7958856487、满奖励3/20，预设参照门槛保持未通过。启动原损失/计划噪声/代表梯度、完整磁盘恢复状态摘要均通过且前后一致；GPU设备修复记录匹配发布合同。检查点SHA在report/侧车一致，用户已报告集群原验证器通过；4GB权重未下载或在本地重新散列。两个固定视频收到、非空及MP4文件头/SHA已记录，本地没有解码画面，不能冒称逐帧审片。详见[完整小工件分析](../reports/pusht_native_analysis_211309_20261019.json)。

## 唯一下一步

从已保存的20000步检查点接续到**事先已声明的84168上限**，新增64168次；原模型、EMA、Adam动量、学习率调度位置、EMA计数和所有RNG一起恢复，原样本/独立噪声计划接续。初始化后构造的对象只用于接收完整状态，不能把恢复视为随机重训或重置优化器。

数据仍90演示/10726窗，有效batch64；原normalizer、扩散损失、模型、AdamW/EMA及512400时域cosine/warmup500均不变。原training_protocol.json与已验证的修复core逐字节相同；新extension_contract.json只声明原定预算接续、参考文件SHA及终点。原20k权重、报告和视频保持，输出到新的native84k目录。

只在84168的EMA上评估一次原20个开发初始场景，采样种子和成功门槛不变；中途只存盘，不评分挑最佳。若上限仍不达标，停止本次模块对照分支，不继续加原生预算；项目组和最终50均不自动运行。训练更多不保证提高得分。

## 新入口检查

- 原20k的5个小JSON及检查点侧车绑定SHA，实际加载时再核对4GB权重SHA。
- 使用已在A6000/211309跑通的恢复设备修复，加载后检查raw/EMA张量均在目标设备；原模型状态完整摘要必须与源检查点完全一致。
- 原损失短审查不执行优化器更新，随后直接执行20001..84168，不另交GPU预检。
- 每2000次及终点保存完整状态，侧车绑定本次extension；中断时可显式恢复该extension的最新检查点，拒绝无关合同。
- 本地小CUDA策略使用实际作者normalizer与EMA，在20k计数边界的下一次更新loss、完整状态、LR与EMA计数精确匹配不中断路径。实际生成的循环用轻量替身执行了全部64168个新计数，最后正确保存84168；原runner接线、错误恢复合同拒绝、样本计划SHA、Python3.9和CLI/Bash/ZIP检查通过。未在本地执行真实原生大模型/训练数据/开发回合，也不声称新入口已经在集群Torch1.12执行。

## 时间与提交

按211309实际训练时间投影，新增纯训练约9.91小时，加载/保存与约16分钟开发评估另计，排队另计。A6000/4CPU/22GB，作业上限16小时，保留完整源权重；不安装、不下载、不生成CLIP全缓存。

发布包：[pusht_native_extension_v1.zip](../cluster_packages/pusht_native_extension_v1.zip)；[发布SHA](../reports/pusht_native_extension_release_20261019.json)；[本地检查](../reports/pusht_native_extension_checks_20261019.json)。**尚未提交。**

Windows PowerShell：

```powershell
scp "D:\ntu_related\dissertation\Zero_shot\cluster_packages\pusht_native_extension_v1.zip" zixiao005@10.97.216.128:/projects/Zeroshot/
```

集群：

```bash
cd /projects/Zeroshot
if [ ! -e pusht_native_extension_v1 ]; then
  /projects/Zeroshot/envs/official-pusht-v1/bin/python -m zipfile -e pusht_native_extension_v1.zip . &&
  PUSHT_NATIVE_CAP=$(sbatch --parsable /projects/Zeroshot/pusht_native_extension_v1/train.sh) &&
  echo "Push-T原生接续训练：$PUSHT_NATIVE_CAP"
else
  echo '目录已存在，请先检查已有作业，避免重复提交。'
fi
```

```bash
sacct -j "$PUSHT_NATIVE_CAP" --format=JobID,State,ExitCode,Elapsed,NodeList
tail -n 30 "/projects/Zeroshot/logs/pusht-native84k-$PUSHT_NATIVE_CAP.out"
```

先看PUSHT_RESTORE_DEVICE PASSED和PUSHT_NATIVE_EXTENSION_RESUME PASSED20000/84168，再看更新进度。收尾用包内verify_extension.py核对完整工件；不另交评价作业。

# 作者最终EMA：冻结图像／当前状态对照

203876完成，1分32秒，A6000，两个小文件独立复算通过。原40窗三个采样逐元素精确重放，最大差0，9窗四观察结果核验一致；8/9窗对换图像更敏感，但窗口146两种替换变化接近。已确认作者模型在一对有效条件下未复现专家轨迹差异，未定位唯一层或修复泛化；本轮收口，不追加作者组合。见[独立结果与项目条件向量复查](BRIDGE_AUTHOR_OBSERVATION_RESULT_203876.md)。起点是[西兰花只读审计](BRIDGE_BROCCOLI_OBSERVATION_AUDIT_203699.md)的范围内失败例子，不更新参数，不评价PyBullet、抓取或混合观察的“成功率”。

当前作业固定查看命令，不用重复提交：

```bash
squeue --start -j 203876
sacct -j 203876 --format=JobID,State,ExitCode,Elapsed,NodeList
if [ -f /projects/Zeroshot/logs/bridge-author-obs-203876.out ]; then
  tail -n 15 /projects/Zeroshot/logs/bridge-author-obs-203876.out
fi
```

## 固定四种观察

保留全部9个西兰花开发窗口，近邻预先按同任务冻结CLIP视觉描述选择，未用未来目标或预测误差选择。每个窗口两种替换使用同一训练近邻，语言保持原值。作者模型实际使用可训练ResNet，CLIP在这里只是审计的近邻选择依据。

| 保存结果名 | 图像 | 当前末端9维状态 | 可作何种判断 |
| --- | --- | --- | --- |
| original | 原窗口 | 原窗口 | 必须重放原预测，可对原目标计算误差 |
| image_only | 训练近邻 | 原窗口 | 仅比较配对输出变化；可能物理不一致 |
| state_only | 原窗口 | 训练近邻 | 仅比较配对输出变化；可能物理不一致 |
| both | 训练近邻 | 同一训练近邻 | 是有效训练观察，可对该训练近邻的目标计算误差 |

状态是观测时刻XYZ及旋转6D，经原训练统计构造的9维特征，不是未来动作或夹爪标签。全部替换只读原缓存，不使用审计导出的未来画面。

冻结检查点：`/projects/Zeroshot/runs/bridge-author-reference-203699/ema_final.pt`，SHA256为`c5c7401054ffa1fb159d86f136a27e1a62895fbdecb40d81c91449414ec5a975`。固定源码提交5ba07ac、原环境Torch1.12.1、100步原生DDPM及种子1101/1102/1103。

## 随机数和验收

先以原来的batch8、行序和每个采样种子重放全部40个开发窗口。每次原生调用必须完整经过99至0的100个去噪时间步。与保存预测最大分量差不得超过1e-4；XYZ单位为10cm，该阈值对应每个XYZ分量0.001cm。重放失败即停止，不能继续解释干预结果。

四种观察使用同一批大小及原批开始时完全相同的CPU/CUDA随机状态，因此初始噪声与逐步随机噪声配对。每批只更换9个指定窗口涉及的行，其他行的预测须保持在同一数值容差内；权重、归一化统计、语言和采样公式不变。结束后逐项比对模型状态与加载检查点，确认参数与buffer未改变。

完整重放15次采样调用，干预24次，总计39次原生采样。三个采样种子仍不是三个独立训练初始化。最终报告`execution_passed`只表示执行契约通过，不表示模型改善或根因已确认。

结果分支：图像替换敏感而状态不敏感，支持在这些配对条件下继续检查状态读取；状态也敏感，则不能说作者忽略了状态。两者同时替换回训练观察可以检验已见输入读出，但其更好不证明单个模块有错。跨图像／状态观察可能不一致，其输出变化只能作敏感性诊断，不能用错误变小宣称修复。某一路敏感也不等于它在有效任务中更重要。

## 准备证据和资源

本地用实际168输入缓存的图像、状态和语言核验9对替换，CPU玩具函数检验配对Torch随机数和后续状态恢复；玩具结果不作为作者模型效果。本地未加载作者模型、未进行CUDA重放或训练。Python3.9语法、真实启动器参数、厘米换算与四元数符号等价及Bash语法通过。[准备记录](../reports/bridge_author_observation_probe_preparation.json)。

冻结ZIP8,533字节、6个文件，SHA256为`f431dd156895817f88fe92edff7e5ff800edb0d3379494d26e9a6c39a6d67226`。复用已有作者环境／源码、203498输入缓存、203699EMA权重与结果，不安装包、不下载权重、不复制大数据。仅保存小预测和JSON，输出预算5MiB，不保存新checkpoint。Slurm上限10分钟，是执行上限，不是排队时间或完成保证。仅提交一个作业。

保留以下集群路径：`bridge_author_preflight_v1`、`bridge_author_reference_v1`、`runs/bridge-author-preflight-203498/inputs`、`runs/bridge-author-reference-203699`及已有作者源码／环境。

## 上传和运行

本地PowerShell：

```powershell
scp "D:/ntu_related/dissertation/Zero_shot/cluster_packages/bridge_author_observation_probe_v1.zip" "D:/ntu_related/dissertation/Zero_shot/cluster_packages/bridge_author_observation_probe_v1.sha256" zixiao005@10.97.216.128:/projects/Zeroshot/
```

集群Bash：

```bash
cd /projects/Zeroshot
mkdir -p logs
sha256sum -c bridge_author_observation_probe_v1.sha256 &&
unzip -n bridge_author_observation_probe_v1.zip &&
AUTHOR_OBS_JOB=$(sbatch --parsable bridge_author_observation_probe_v1/probe.sh) &&
echo "作者图像状态对照：$AUTHOR_OBS_JOB"
```

查看同一终端变量；断连后换成实际作业号：

```bash
squeue --start -j "$AUTHOR_OBS_JOB"
sacct -j "$AUTHOR_OBS_JOB" --format=JobID,State,ExitCode,Elapsed,NodeList
if [ -f "/projects/Zeroshot/logs/bridge-author-obs-$AUTHOR_OBS_JOB.out" ]; then
  tail -n 15 "/projects/Zeroshot/logs/bridge-author-obs-$AUTHOR_OBS_JOB.out"
fi
```

完成后在集群打包小结果：

```bash
cd /projects/Zeroshot/runs
tar -czf "/projects/Zeroshot/bridge_author_obs_results_$AUTHOR_OBS_JOB.tar.gz" "bridge-author-obs-$AUTHOR_OBS_JOB"
```

本地用实际编号替换`JOBID`：

```powershell
scp zixiao005@10.97.216.128:/projects/Zeroshot/bridge_author_obs_results_JOBID.tar.gz "D:/ntu_related/dissertation/Zero_shot/reports/"
tar -xzf "D:/ntu_related/dissertation/Zero_shot/reports/bridge_author_obs_results_JOBID.tar.gz" -C "D:/ntu_related/dissertation/Zero_shot/reports/"
```

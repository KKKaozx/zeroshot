# 同Bridge作者实现：固定预算正式参照

2026-10-14：预检203498的六份小文件已下载核验。168窗口输入身份、四次真实更新日志、训练统计归一化及反变换通过；独立计算归一化系数最大差约2.33e-7，报告的反变换最大分量差约2.39e-7。峰值保留显存5.91GiB。编码器、U-Net和EMA变化从报告核对，未下载权重独立复算。此次不评价训练拟合或开发泛化。见[核验记录](../reports/bridge-author-preflight-203498/verification.json)。

正式参照203699完成，19分12秒，A6000，八份小结果独立核验通过。8192更新、配置/输入目标/行索引与预测指标复算通过；最终raw/EMA训练拟合门槛均未通过，主要EMA开发路径8.378cm/15.220°仍略差于静止。当前不追加训练，下一项转向共享观察与监督适用性审计。见[独立结果、任务与步长拆分](BRIDGE_AUTHOR_REFERENCE_RESULT_203699_20261014.md)。它改变多个模块，不能据此直接定位一个错误层，也不是完整论文复现。

当前作业固定检查命令（不用重复提交）：

```bash
squeue --start -j 203699
sacct -j 203699 --format=JobID,State,ExitCode,Elapsed,NodeList
if [ -f /projects/Zeroshot/logs/bridge-author-reference-203699.out ]; then
  tail -n 15 /projects/Zeroshot/logs/bridge-author-reference-203699.out
fi
```

## 固定内容与结果分支

- 原128训练窗口/64演示、40开发窗口/10演示，初始化43，batch8，512轮、8192次优化更新。每个训练窗完整参与512次；不换窗、不加数据、不追加种子或轮数。
- 复用203498导出的浮点图像、当前9维末端状态、冻结CLIP文本768维特征及原7维相对位姿目标。16步轨迹；不训练或评价夹爪，不执行仿真。
- 使用固定作者源码中的Workspace、可训练图像编码器、U-Net、epsilon目标、训练集limits归一化、AdamW、500步warmup/cosine、EMA及100步DDPM。与项目Adapter方案有多个差异，已在预检协议列明。
- 固定最终EMA为主要结果，最终原始参数为辅助诊断；各用三个固定采样种子完整评价训练和开发。三个采样种子不是三次独立训练。不按开发结果选择权重或种子。
- 训练门槛沿用原项目：至少90%窗口在三个采样结果中均达到平均路径位置≤5mm、旋转≤5°，每个采样结果的整体路径及终点也满足既定门槛。
- 有效参照须主要EMA通过训练拟合，而且开发路径位置、旋转均优于静止，至少6/10开发演示位置优于静止。否则如实报告本预算下的拟合或泛化限制，不将失败归因于唯一模块。

只有建立有效的同数据参照后，才设计条件编码与扩散实现边界的匹配对照。开发集已经反复用于诊断，不能称独立最终测试。

## 资源和文件

复用已有环境与作者源码，不安装包或下载权重。新ZIP为11,978字节，7个文件；SHA256为`6b5edbd228061b04d7ae414b0c8722b3569e9bfd03384016fae0544ead034b1a`。[本地准备检查](../reports/bridge_author_reference_preparation_20261014.json)包含Python3.9语法、真实启动参数、与已有预测一致的物理指标以及预算门槛检查；Bash语法检查通过，本地未运行正式训练。

Slurm上限3小时，含训练、最终评价和保存，不是预计耗时。前8次正式更新内测量迭代速度；保守估计超过预留预算则停止，报告原因，不自动延长作业。只保存最终raw/EMA策略权重，不保存优化器或每轮权重，两份合计限制3GiB，留在集群。上传包和小结果不会显著占用本地硬盘。

**保留集群原目录**`/projects/Zeroshot/bridge_author_preflight_v1`及`/projects/Zeroshot/runs/bridge-author-preflight-203498`；正式作业会校验源码、报告和输入缓存，不重新导出大输入。

## 上传与提交

本地PowerShell：

```powershell
scp "D:/ntu_related/dissertation/Zero_shot/cluster_packages/bridge_author_reference_v1.zip" "D:/ntu_related/dissertation/Zero_shot/cluster_packages/bridge_author_reference_v1.sha256" zixiao005@10.97.216.128:/projects/Zeroshot/
```

集群Bash（只提交一次）：

```bash
cd /projects/Zeroshot
mkdir -p logs
sha256sum -c bridge_author_reference_v1.sha256 &&
unzip -n bridge_author_reference_v1.zip &&
AUTHOR_BRIDGE_TRAIN=$(sbatch --parsable bridge_author_reference_v1/train.sh) &&
echo "作者Bridge正式参照：$AUTHOR_BRIDGE_TRAIN"
```

查看进度，变量须在同一终端保留；断连后将变量换成实际作业号：

```bash
squeue --start -j "$AUTHOR_BRIDGE_TRAIN"
sacct -j "$AUTHOR_BRIDGE_TRAIN" --format=JobID,State,ExitCode,Elapsed,NodeList
if [ -f "/projects/Zeroshot/logs/bridge-author-reference-$AUTHOR_BRIDGE_TRAIN.out" ]; then
  tail -n 15 "/projects/Zeroshot/logs/bridge-author-reference-$AUTHOR_BRIDGE_TRAIN.out"
fi
```

完成后只取小结果：

```bash
cd /projects/Zeroshot/runs
tar --exclude='*.pt' --exclude='*.ckpt' -czf "/projects/Zeroshot/bridge_author_reference_results_$AUTHOR_BRIDGE_TRAIN.tar.gz" "bridge-author-reference-$AUTHOR_BRIDGE_TRAIN"
```

本地使用实际作业号替换下面的`JOBID`：

```powershell
scp zixiao005@10.97.216.128:/projects/Zeroshot/bridge_author_reference_results_JOBID.tar.gz "D:/ntu_related/dissertation/Zero_shot/reports/"
tar -xzf "D:/ntu_related/dissertation/Zero_shot/reports/bridge_author_reference_results_JOBID.tar.gz" -C "D:/ntu_related/dissertation/Zero_shot/reports/"
```

`execution_passed`只表示执行验收，是否有效须独立复算保存的预测并按以上门槛判定。

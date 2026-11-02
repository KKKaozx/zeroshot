# 原生参照工件核对与冻结CLIP训练缓存

用户报告原生cap验证器通过。已下载的development.json逐步奖励复算仍为均值0.84587825、满奖励10/20，原门槛通过；latest.json的84168计数、base/extension合同SHA、startup源20k侧车及GPU恢复记录一致。视频文件已收到，尚未解码。**目录缺少report.json**，不能称本地完整summary已审阅或4GB权重已下载。详见[可用工件分析](../reports/pusht_native_analysis_212158_20261019.json)。

请先补齐报告：

```powershell
scp zixiao005@10.97.216.128:/projects/Zeroshot/runs/pusht-native84k-212158/report.json "D:\ntu_related\dissertation\Zero_shot\reports\pusht_native_212158\report.json"
```

## 下一项实际准备

项目前端需输入冻结CLIP的patch/text特征。一次性编码同90演示的11356训练帧，产生FP32[11356,257,1024]、约11.133GiB的视觉缓存及原始全局frame ID、固定文本token/mask。使用原生相同的10726取窗映射和原图SHA，后续训练每次仍取相同2帧观察。

这是实际训练输入准备，0优化更新、无模型效果评分。不会重复原资源5次更新或在线16步接线；复用211193原worker/client/codec，CLIP视觉与文本严格冻结，前后完整state hash不变。仅传原像素与固定文本，原始Zarr不修改，不安装或下载。原128参考帧新缓存须与既有特征在原容差内一致，失败不放宽门槛。

内存映射写11GiB文件，按32帧请求、内部8帧编码，短尾沿用原补批方式；worker在长I/O散列前关闭。未完成文件保留writing名称，不标记完成；最终文件SHA/形状/类型/全局索引均可由CPU验证器检查。写入前检查32GiB磁盘余量，预留后续完整训练检查点空间。缓存留在集群，不需下载。

后续匹配训练预算84168，原U-Net结构/新初始化后端/数据/statistics/样本与噪声计划/AdamW/LR/EMA与原生一致，只替换前端家族。项目Adapter和1024→64映射可训练，CLIP硬冻结；不接收原生已经学好的后端权重。资源测试投影约24小时纯训练，计划分段完整恢复、只终点评价同20开发场景，最终50封存。此包没有正式训练入口，也不自动提交训练；正式入口须在缺失cap报告和缓存核验后放行。

## 本地验证与提交

数值替身检查了memmap发布、32+短尾请求、重复/乱序参考查询、缺失ID/错误特征/dtype/worker失败拒绝；原native TrainingData AST、原worker五文件字节一致。Python3.9 AST、CLI/Bash、ZIP/SHA通过。未在本地生成真实全缓存或执行CLIP/训练/仿真，真实11356帧缓存仍待本集群作业完成。

包：[pusht_project_cache_v1.zip](../cluster_packages/pusht_project_cache_v1.zip)，[发布记录](../reports/pusht_project_cache_release_20261019.json)，[CPU检查](../reports/pusht_project_cache_cpu_checks_20261019.json)。预计数分钟，A6000/4CPU/22GB，作业上限30分钟；尚未提交。

PowerShell上传：

```powershell
scp "D:\ntu_related\dissertation\Zero_shot\cluster_packages\pusht_project_cache_v1.zip" zixiao005@10.97.216.128:/projects/Zeroshot/
```

集群：

```bash
cd /projects/Zeroshot
if [ ! -e pusht_project_cache_v1 ]; then
  /projects/Zeroshot/envs/official-pusht-v1/bin/python -m zipfile -e pusht_project_cache_v1.zip . &&
  PUSHT_CACHE=$(sbatch --parsable /projects/Zeroshot/pusht_project_cache_v1/cache.sh) &&
  echo "Push-T冻结特征缓存：$PUSHT_CACHE"
else
  echo '目录已存在，请先检查已有作业，避免重复提交。'
fi
```

日志logs/pusht-clip-cache-JOBID.out，报告runs/pusht-project-cache-JOBID/report.json；完成后调用包内verify_cache.py，无新训练/评分作业。

# DINO训练缓存与在线接线包

220882的三个小报告已接收并核对，原128帧资源探针不再重复。本次只完成正式骨干对照所需的训练缓存与真实新worker接线，零次参数更新、零次环境执行、无奖励或泛化结论。完整长训练入口尚未生成。

## 固定范围

- 沿用原90训练演示、11356帧、10726窗口来源；原训练原始像素/帧ID/固定CLIP文本均绑定散列。不扫描新任务或抽取新窗口，不读最终集。
- 冻结DINO-L/14官方固定资产不变；保留旧CLIP的224resize/224crop画面几何，DINO原生ImageNet归一化及最终LayerNorm差异公开。原八层Adapter、池化、投影、作者U-Net与训练专用归一化保留。
- 训练缓存通过新DINO worker直接生成；原Python3.9读Zarr/调策略，新Python3.10进程运行DINO。不安装、升级或下载依赖。不复制原始图像、CLIP视觉缓存或大权重。
- 对照220882原128帧；重新启动worker，以1/2/4批量检查原作者RGB往返、pad8、token、Adapter context，固定seed1101比较2窗完整16步及执行8步DDPM100动作。策略从原42/43种子新建，只验证接口，没有训练或闭环能力结论。
- 保存`backbone_binding.json`供未来正式checkpoint绑定。当前没有正式保存/恢复入口，不能把身份文件生成当成正式恢复已验证。

## 空间和资源

缓存约11.13GiB；提交前及作业内检查`du`占用和文件系统剩余量，按保守200十进制GB项目上限再保留24GiB余量。该检查不是权威账户quota查询，写盘异常仍直接失败。不自动执行清理、降精度或重新提交；旧清理脚本尚未收到实际执行回执，不能声称已经释放空间。

A40、4CPU、40GiB内存、30分钟硬上限。220882吞吐推算纯编码约186秒，另有读数据、IPC、文件散列和接线开销；30分钟不是预计训练时长。本包不启动25.5小时正式训练。

## 本地验收

20文件小ZIP，10个原核心模块字节不变。29项检查覆盖包装、Python3.9语法、空间上限/余量、缓存及身份绑定损坏拒绝、数值socket worker请求/关闭及错误dtype；两个Bash脚本语法通过。新通信路径使用极小确定性替身验证生命周期，真实预训练骨干/Adapter接口证据沿用已有本机检查；实际跨两Python环境的DINO一致性必须由本次集群作业完成，尚无新的服务器结果。最初沙箱限制本机socketpair，允许本机回环检查后通过；不是集群CUDA故障。

## 上传、提交和核验

本机PowerShell：

```powershell
scp "D:\ntu_related\dissertation\Zero_shot\cluster_packages\pusht_dino_cache_v1.zip" zixiao005@10.97.216.128:/projects/Zeroshot/
```

集群：

```bash
cd /projects/Zeroshot
unzip pusht_dino_cache_v1.zip
bash /projects/Zeroshot/pusht_dino_cache_v1/submit_once.sh
```

仅首次提交此新目录；`.submission`防重复提交。空间检查失败则不排队，先查空间/清理回执，不直接绕过保护。若sbatch失败，保存的防重投目录留待核对，不自动删除或重投。

完成后将下面编号改为本次真实编号：

```bash
DINO_CACHE_JOB=实际编号
LD_LIBRARY_PATH=/projects/Zeroshot/envs/official-pusht-v1/lib \
  /projects/Zeroshot/envs/official-pusht-v1/bin/python -B \
  /projects/Zeroshot/pusht_dino_cache_v1/run_cache.py verify \
  --output "/projects/Zeroshot/runs/pusht-dino-cache-${DINO_CACHE_JOB}"
```

验收后只打包小报告，不下载11.13GiB缓存：

```bash
tar -czf "/projects/Zeroshot/pusht_dino_cache_${DINO_CACHE_JOB}_reports.tar.gz" \
  -C "/projects/Zeroshot/runs/pusht-dino-cache-${DINO_CACHE_JOB}" \
  report.json backbone_binding.json training_data.json live_action_identity.npz \
  cache_worker/dino_worker.log live_worker/dino_worker.log
```

通过后再审阅同84168次更新、两段42084的正式入口、完整状态恢复和身份绑定；保留原CLIP组0.445及原生0.846参考，固定终点同开发场景评价，不追加无关结构分支，不访问final50。

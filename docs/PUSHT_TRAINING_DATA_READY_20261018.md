# Push-T训练数据与实际环境准备包

承接210557资产报告及用户“好”（2026-10-18），仅准备CPU数据获取/完整读取及实际项目环境元信息检查。没有本地真实下载、模型执行、环境安装或集群提交，没有新的泛化结果。

## 范围

- 仅获取作者[训练目录](https://diffusion-policy.cs.columbia.edu/data/training/)中的`pusht.zip`，目录列示约29MB；不下载4GB权重或pusht_real。最多三次尝试/总10分钟、256MiB下载上限；用Python HTTPS，保留证书验证，避开此前curl共享库问题，不运行Conda/Pip。
- 补查实际`multitask-preflight-v1`包版本，纠正210557查早期bridge-diffusion的遗漏，不重读检查点。元信息不等于运行兼容验证。
- ZIP路径/重复项/符号链接/CRC/空间检查后暂存解压；核对Zarr96×96RGB、二维动作、状态及演示边界。显式检查所有预期块存在，然后每次最多256帧逐块解码图像/动作/状态、检查有限值、保存内容指纹。既有合格数据复用，不覆盖作者原目录。
- 输出作者原保存配方的演示/窗口计数，仅描述旧配方，不建立新训练/开发划分、不拟合归一化统计。后续公平协议仍须两组训练数据统计一致。
- 原作者环境只用于现有NumPy/Zarr读取；不改任何环境，不加载CLIP，不读Bridge最终25，不请求GPU，不自动训练。207558保持。

数据新位置：`/projects/Zeroshot/paper_reproduction/pusht_module_data_v1/`。新作业报告：`/projects/Zeroshot/runs/pusht-data-prepare-JOBID/report.json`及`active_environment.json`。中断/损坏中间产物保留，不递归删除。数据SHA是本地身份记录，不冒称已与作者公布SHA核对。

## 本地验证

包：`cluster_packages/pusht_training_data_v1.zip`，8643字节、6文件，SHA256 `12c35123c3faf6ca857c0266849094a3b3884d880f8aa87ee065b8a08e98db08`。

Python3.9语法、CLI帮助、Bash语法、ZIP/SHA；五种危险归档拒绝、正常目录识别、合成缺块/非有限值拒绝、模拟HTTP下载发布/复用/超限拒绝通过。窗口计数帮助函数原样复制210557已用版本，18组与固定作者采样代码对照再次通过。Windows本地无Zarr，本地没有验证真实codec/Zarr解码；由本次集群CPU作业实际执行，失败不自动安装。

资源：CPU4核/12GB/30分钟上限。完整读取图片是数据完整性检查，不是CLIP特征编码、评估或模型训练。准备通过也不等于训练资源或兼容通过。

## 上传和提交

Windows PowerShell：

```powershell
scp "D:\ntu_related\dissertation\Zero_shot\cluster_packages\pusht_training_data_v1.zip" zixiao005@10.97.216.128:/projects/Zeroshot/
```

集群：

```bash
cd /projects/Zeroshot
if [ ! -e pusht_training_data_v1 ]; then
  /projects/Zeroshot/envs/official-pusht-v1/bin/python -m zipfile -e pusht_training_data_v1.zip .
  PUSHT_DATA=$(sbatch --parsable /projects/Zeroshot/pusht_training_data_v1/prepare.sh)
  echo "Push-T数据准备：$PUSHT_DATA"
else
  echo '目录已存在；先核对已有作业，避免重复覆盖或提交。'
fi
```

查看：

```bash
sacct -j "$PUSHT_DATA" --format=JobID,State,ExitCode,Elapsed,NodeList
tail -n 20 "/projects/Zeroshot/logs/pusht-data-$PUSHT_DATA.out"
```

下一阶段依据真实报告确定隔离接口、训练/开发/最终回合、仅训练统计以及有界资源测量；本包没有正式训练入口，不把原0.9309预训练得分当成短预算对照成绩。

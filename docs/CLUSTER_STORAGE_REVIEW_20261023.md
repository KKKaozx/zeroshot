# 集群空间清理前的依赖核对

用户报告200GB空间接近满额。当前没有服务器目录大小清单，未执行删除，也未批准按作业状态批量删除。下一项是只读du/find盘点，再按具体文件做保留/归档/删除清单。

## 盘点收到后的首批方案

2026-10-23用户清单：项目157G，runs98G，.tmp22G，envs19G，logs仅2.7M。日志并非主要占用。首批只退役以下6个权重文件，按用户du舍入值合计约25.6G，不删除目录；实际释放量以文件链接数/磁盘占用为准。

| 文件 | 约占用 | 退役依据 |
|---|---:|---|
| runs/pusht-bypass-preflight-217778/global_control/replay.pt | 4.6G | 关闭旁路分支的短预检状态；217820正式最终权重保留 |
| runs/pusht-bypass-preflight-217778/spatial_candidate/replay.pt | 4.6G | 同上 |
| runs/pusht-native20k-211279/latest.pt | 4.0G | 旧失败启动状态，原生212158完整终点保留；若缺校验侧车则跳过 |
| runs/pusht-native20k-211309/latest.pt | 4.0G | 已完整恢复并训练到212158终点；核对原始父检查点SHA |
| runs/pusht-project-part1-214273/latest.pt | 4.6G | 已完整恢复并训练到215999终点；核对原始父检查点SHA |
| runs/pusht-zero-preflight-219260/replay.pt | 3.8G | 两步预检状态；219466正式完整终点保留 |

`scripts/cluster_storage_cleanup_v1.py`默认只读预览，`--apply`才执行。只允许已解析的`/projects/_ssd/Zeroshot`，拒绝路径转向/目录/越界；待删文件与5个保留终点均先读完整SHA，最终终点报告和父状态恢复证据一致后才允许删除。有任何Slurm排队/运行作业则停止，检查失败不自动放宽。每个删除写入`storage_cleanup_receipts`账本；保留原JSON/YAML、代码、视频、所有正式终点和其他文件。没有递归删除、通配批删或模型载入。

20项本地极小自建文件测试通过，包括活跃作业、损坏候选、损坏终点、路径越界、缺失败侧车、账本和重复执行；没有连接集群或实际删除集群文件。脚本SHA见`reports/cluster_storage_cleanup_checks_20261023.json`。

清理不创建这些旧大权重的字节备份，删除后不能从相应旧中间时点恢复，也不能原样重跑依赖replay.pt的历史预检完整验收；原已验证的报告/恢复摘要和正式终点保留，历史报告不被改写为当前仍保留所有中间状态。DINO下一流程不依赖这些旧预检大状态。未执行前不能称已释放25G。

本地上传：

```powershell
scp "D:\ntu_related\dissertation\Zero_shot\scripts\cluster_storage_cleanup_v1.py" zixiao005@10.97.216.128:/projects/Zeroshot/
```

集群：

```bash
python3 /projects/Zeroshot/cluster_storage_cleanup_v1.py
python3 /projects/Zeroshot/cluster_storage_cleanup_v1.py --apply
du -sh /projects/_ssd/Zeroshot
```

`.tmp`22G尚不知内容，本批不动；同样不动11.13GiB旧CLIP视觉缓存、环境、数据集和Bridge参照。需要进一步空间时，先`du -xhd2 /projects/_ssd/Zeroshot/.tmp | sort -hr | head -n 30`读取内容大小，不整体删除。

## Bridge第二批可清理项

用户追问旧Bridge是否也能删。可以，上一批只选择了最明确的Push-T旧状态，并非要求保留所有Bridge大权重。本次核对保存代码与收到的原报告，确认以下6个是优化器/恢复快照，各组另有较小的完整学习参数推理权重；固定上下文组的保存函数按参数前缀导出，包含冻结的上下文学习参数，不会因清理Adam状态丢失它们。

| 明确文件 | 已有报告字节数 |
|---|---:|
| runs/bridge-late-lr-fit-200150/common_u1024_with_adam.pt | 1587979022 |
| runs/bridge-lr-rep-fit-200208/seed42/common_u1024_with_adam.pt | 1587979150 |
| runs/bridge-lr-rep-fit-200208/seed44/common_u1024_with_adam.pt | 1587979150 |
| runs/bridge-budget-fit-208586/training_state_u08192.pt | 1588139196 |
| runs/bridge-budget-fit-208586/training_state_u16170.pt | 1588139196 |
| runs/bridge-fixed-fit-208964/training_state_u16170.pt | 1588159356 |

合9528375070字节，约8.87GiB。保留上述五个实验目录中的8份最终推理权重（每个学习率对照的constant/reduced各一份，budget/fixed各一份）、全部报告/数组/日志/源码。207558主参照、207746旋转对照、207348覆盖初始化参照、203699作者Bridge参照以及数据/缓存均不在删除清单。

`scripts/cluster_bridge_state_cleanup_v1.py`重用首批脚本的路径/哈希/Slurm/账本机制，新增固定Bridge配置；对五份报告SHA及完成字段、8份保留推理权重的SHA和6份待删状态SHA校验后才删除。Bridge层13项自建小文件检查通过，原20项通用保护不重复执行。删除后不能原样恢复这些旧Adam状态或重放旧分叉训练，但仍保留现成最终推理权重和历史指标；不创建大状态备份。当前尚未执行任何远程删除。

上传两个脚本以确保基础版本匹配：

```powershell
scp "D:\ntu_related\dissertation\Zero_shot\scripts\cluster_storage_cleanup_v1.py" "D:\ntu_related\dissertation\Zero_shot\scripts\cluster_bridge_state_cleanup_v1.py" zixiao005@10.97.216.128:/projects/Zeroshot/
```

集群先预览，再执行本批Bridge清理：

```bash
python3 /projects/Zeroshot/cluster_bridge_state_cleanup_v1.py
python3 /projects/Zeroshot/cluster_bridge_state_cleanup_v1.py --apply
du -sh /projects/_ssd/Zeroshot
```

此命令只清理Bridge六状态，不会再次执行Push-T首批。两批都实际执行后预计合计约34G，最终释放量以服务器占用为准。

## 其余已确认的保留依赖

- Push-T实际数据在`runs/pusht-data-prepare-210631/extraction_staging/pusht/pusht_cchi_v7_replay.zarr`。210631历史状态FAILED不代表目录无用，210690的有效数据报告明确preserve_origin_run_data=true。禁止按FAILED状态删除旧目录。
- `runs/pusht-float-data-210690/report.json`绑定数据路径；`runs/pusht-front-resource-210763/report.json`和`inputs.npz`提供固定128帧/64窗；`runs/pusht-online-211193/report.json`是主协议所需参考。小报告应保留原路径和字节，不能先删再改脚本来规避验证。
- `runs/pusht-project-cache-213439/report.json`和`text.npz`仍由新DINO资源路径读取。原`visual.npy`约11.13GiB，可再生成，但原CLIP训练器/验证器读取完整缓存；若退役该数组，需明确后续DINO入口只依赖报告/文本且保留重建来源，不能删整个213439。
- 已验证端点212158原生、215999项目8块、219466零块的大检查点暂留服务器，本地只收到小报告/视频，没有本地大权重备份。Bridge主参照207558及论文需要的权重/配置/源码快照也先保留。不能声称“已下载小报告”就等于权重已备份。
- 当前DINO220882报告/128帧token和预训练DINO资产是下一步全缓存与身份绑定的依据。既有CLIP资产、实际使用的official-pusht-v1和multitask-preflight-v1环境、作者源码暂不动。

## 优先候选类别

无后续依赖且已有结果归档的旧预检完整replay权重、失败/关闭分支大权重、已被后段完全恢复且无继续使用计划的前段检查点、重复压缩/上传包、可重建的旧特征缓存。是否能删除要按实际路径/大小/现有备份核对，不使用`runs/*/latest.pt`等通配批删。

日志、JSON/YAML、源脚本及两段论文示例视频通常比完整Adam/EMA检查点小得多，保留它们有利于结果追溯。200GB是否主要被这些候选占用尚不确定，不能先给节省容量承诺。

集群只读盘点（项目路径先解析符号链接）：

```bash
PROJECT_ROOT=$(readlink -f /projects/Zeroshot)
du -xhd1 "$PROJECT_ROOT" | sort -hr
du -xhd1 "$PROJECT_ROOT/runs" | sort -hr | head -n 30
find "$PROJECT_ROOT/runs" -xdev -type f -size +500M -printf '%s\t%p\n' | sort -nr | head -n 30 | numfmt --field=1 --to=iec
```

这些命令不创建备份或删除文件。收到盘点后再给具体清理清单；清理前还需核对其他正在运行的作业是否使用候选路径。

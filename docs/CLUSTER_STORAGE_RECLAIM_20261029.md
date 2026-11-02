# 集群空间清理（2026-10-29）

用户提供的盘点显示项目约 179 GiB，其中 runs 约 103 GiB、.tmp 约 22 GiB、envs 约 20 GiB；用户称额度只剩约 9 GB。这是目录占用清单，不是权威配额查询。盘点时唯一作业为 PENDING 的 233531。

用户明确授权“所有能删的都删掉”。已准备单文件脚本 `scripts/cluster_reclaim_space_20261029.py`，不申请 GPU，不安装包、不取消作业。尚未访问集群或执行实际清理；在集群执行 `--apply` 才会删除。

## 清理范围

- pip 的 HTTP、wheel、自检缓存，约 11 GiB。
- Conda 下载压缩包和索引；保留解压包，防止破坏仍通过软链接引用缓存的环境。[Conda 官方说明](https://docs.conda.io/projects/conda/en/latest/commands/clean.html)、[pip 缓存说明](https://pip.pypa.io/en/stable/topics/caching/)。
- 两份 Push-T 冻结训练 visual.npy，共 22.266 GiB；保留帧索引、文本、报告和权重。以后重跑相应旧训练需要重建这两份缓存。
- 已关闭 Bridge 分支的 full_tokens.pt、delta_visual_tokens.pt 等已列名特征缓存，约 9 GiB；原数据、模型、程序及历史报告保留。若重放旧缓存依赖脚本，需要先重建缓存；不把历史通过状态改写成当前可直接重放。
- 失败的 211279 检查点；已完整接续到终点的原生 20k、CLIP 第一段、DINO 第一段检查点；旧一次性预检 replay 和已关闭监督预检的优化器状态。已不存在的跳过。
- 先前已审核的六个 Bridge Adam 中间状态，仅在对应报告和独立最终权重的 SHA256 仍一致时删除；先前已清掉的跳过。
- 项目根目录上传 ZIP：仅当全部成员在解压目录中存在且逐字节相同时删除压缩包。缺文件、内容改动、路径异常、软链接或无法核对则保留归档。

按本次四舍五入清单估计约 50 GiB，取决于仍存在的目标及重复包；不保证具体释放量。脚本记录逻辑删除量和估计释放块数，实际项目额度仍以集群查询为准。硬链接可能使回收量小于文件逻辑大小。

## 保留范围

当前 233531 整个输出目录、232587 和 231297 整个源目录、OXE seed42/43 最终模型、有效 Bridge 207558/207746/A40 骨干对照，以及 Push-T 三方/零块/旁路最终模型全部保留。原始 data、envs、baseline_setup、DINO 资产、official_pusht、全部源码、报告、预测和日志不删除。

特别保留 226133 的原始主预检检查点：新的主训练条件首次绑定时，`bind_preflight.py` 仍可能读取其合同；不因它只有两步就贸然删除。`.tmp/conda` 的解压包、HF/TFHub 模型资产也没有被当成普通下载压缩包清空。

运行时允许只有已核对依赖的 233531 在队列中或运行；发现其他作业则停止，避免删除未知任务缓存。先验证保留模型、源文件和所有删除目标，再逐文件 unlink，检查路径、inode、大小和修改时间；不递归清空目录。每次删除前后写 fsync 的 JSONL 回执，重复执行只处理仍存在的目标。

## 本地验证

10 项微型文件回归通过，覆盖白名单精确删除、所有保留字节不变、重复运行、作业守卫、保留模型/候选变化拒绝、错误父检查点摘要拒绝、缺失后继模型保护、逐成员 ZIP 比对、路径越界拒绝、Bridge 独立最终权重门槛。Windows 限制实际硬链接创建，该分支使用 inode/stat 夹具测试，未声称做过真实硬链接集成。临时文件仅创建在本工作区 `.tmp` 的已核对专属目录。

## 执行

本机 PowerShell：

```powershell
scp "D:\ntu_related\dissertation\Zero_shot\scripts\cluster_reclaim_space_20261029.py" zixiao005@10.97.216.128:/projects/Zeroshot/
```

集群：

```bash
(
set -euo pipefail
cd /projects/Zeroshot
python3 -B cluster_reclaim_space_20261029.py --apply
du -sh .
)
```

结束打印 `CLEANUP_COMPLETE`，详细删除回执位于 `/projects/Zeroshot/storage_cleanup_receipts/reclaim_20261029_*.jsonl`。脚本中途若拒绝，不要绕过守卫，保留输出用于定位被保护的具体文件。

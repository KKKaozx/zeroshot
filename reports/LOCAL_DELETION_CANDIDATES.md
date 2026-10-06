# 可删除候选及代价

用户随后授权删除。2026-10-06 已删除下面三个 latest.pt，以及重复副本 `results/bridge_diffusion_target_pair_smoke_v1/epsilon/final_head.pt`，合计释放约 2.018 GiB。保留项已核对。逐文件原大小、SHA256 和删除记录见 [local_checkpoint_cleanup.json](local_checkpoint_cleanup.json)。以下清单保留删除前的判断依据。

## 放弃旧实验续训后可删除的候选

| 文件 | 可释放 GiB | 保留项与代价 |
|---|---:|---|
| `results/oxe_core_pilot/latest.pt` | 0.670 | 保留 best.pt、history 和其他配置；无法直接从该最终状态恢复训练，也不能重新评价该最终权重 |
| `results/oxe_core_8shard_v1/latest.pt` | 0.671 | 保留 best.pt、history 和其他配置；无法直接从该最终状态恢复训练，也不能重新评价该最终权重 |
| `results/bcz_overfit_cosine_v1/latest.pt` | 0.670 | 保留 best.pt、history 和其他配置；无法直接从该最终状态恢复训练，也不能重新评价该最终权重 |

合计约 **2.01 GiB**。这些不是完全重复文件，不能称作无损删除。当前已搜索 code/experiments/configs/reports/results 的 Python、JSON、Markdown 和 args 文件，未发现对上述完整相对 latest.pt 路径的直接引用；该搜索不排除动态路径、外部引用或未来复查需求。

## 字节完全重复的文件

SHA256 核对发现一个约 6.66 MiB 的重复副本对：

- `results/bridge_diffusion_fit_smoke_v2/final_head.pt`
- `results/bridge_diffusion_target_pair_smoke_v1/epsilon/final_head.pt`

两者字节相同，可保留其中一个作为权重备份，但直接删掉另一原路径可能影响历史脚本读取该路径。该项释放空间很少。核验记录见 checkpoint_duplicate_audit.json。

## 暂不建议删除

当前完整 Bridge 权重、冻结上下文、隔离动作头、扩散导出的初始权重/输入及核心报告。大量其他旧权重仍需按实验用途核对，不能把 28.7 GiB 全部当作垃圾。若需大量释放空间，优先转存到已检查有余量的 E 盘，逐文件校验后再处理本地副本；此处未执行转存。

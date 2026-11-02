# patch_readout_v1：让策略真正看到空间与速度信息

2026-11-01。新目录，不改动任何已冻结部署或旧结果。主视觉骨干为冻结 DINOv2 ViT-L/14（老师第七封邮件），CLIP 文本保持冻结。仍是 OXE(Bridge) 源端训练，不使用任何 LIBERO 示教或目标域统计。

## 为什么改（证据）

1. **旧 Adapter 只读 CLS，patch token 完全没被用到。** `code/adapter.py` 里每个视觉 token 各自做“视觉查询文本”的交叉注意力和 FFN，token 之间没有交互；`pooling='cls'` 只取 CLS，所以 256 个 patch 对输出梯度恒为 0（`cpu_tests.py` 第 1 项直接验证）。策略实际只看到一个全局向量加文本，物体/夹爪位置信息基本丢失。
2. **旧离线门槛对扩散策略不公平。** 它拿一次随机采样的误差比确定性零命令。`reanalyze_old_runs.py` 用已有预测重算：

   | 首命令位置(归一化) | 单次采样 | 3次平均 | 零信息随机策略 | 零命令 |
   |---|---|---|---|---|
   | 237424 训练/开发 | .177/.235 | .161/.212 | .237/.290 | .157/.224 |
   | 237579 训练/开发 | .216/.262 | .176/.229 | .237/.290 | .157/.224 |

   237424 明显有信息（比零信息随机策略好 20% 以上），只是单次采样过不了“优于零命令”。5Hz 旋转增量近似噪声：即使知道上一条命令，旋转误差也只从 2.66° 降到 2.60°，所以旋转不再设门槛。
3. **缺速度信息。** 已知上一条命令可让首命令位置误差从 .163 降到 .118（约 28%），而只用当前位姿+夹爪的线性回归没有任何信息。单帧+绝对位姿看不到“正在往哪动”。

## 改了什么

| 改动 | 实现 | 位置 |
|---|---|---|
| patch 注意力读出 | 保留老师设计的逐 token 视觉→文本交叉注意力（8 层），之后用“文本条件的可学习查询”对全部 patch(+可学习位置编码)做注意力读出；`readout='cls'` 保留作对照 | `code/patch_policy.py` |
| 速度/历史条件 | 上一时刻位姿差（位置/0.1 + 旋转向量）与上一条命令，带可用性掩码，训练时 30% 随机丢弃以防抄历史 | `build_cache.py`, `code/patch_policy.py` |
| 内部旋转表示 | 位置3+旋转向量3 一起标准化（原来 qw 的 std≈0.001，标准化放大噪声）；外部仍输出 8 维 | `code/geometry.py` |
| 扩散后端 | Diffusion Policy 式三尺度 ConditionalUnet1D（每层 FiLM），x0 预测，余弦噪声表，EMA 权重评估 | `code/patch_policy.py` |
| 标签 | 两种都支持：command8 控制增量（与 237424/237579 同一标签）和 234100 未来到达位姿（旧有正信号的标签） | 配置 `target` |
| 特征缓存 | 冻结编码器输出一次性缓存到集群，训练不再每步跑 ViT-L；每个配置约 10–25 分钟 | `build_cache.py`, `code/cached_data.py` |
| 公平指标 | 同时报告单次采样、8 次采样均值(点预测)、零信息随机策略、零命令、训练均值 | `code/fair_metrics.py` |

窗口完全沿用 237424/237579：10352 训练窗/1151 演示 + 64 开发组，固定评估 128 训练组 + 64 开发组。

## 2026-11-01 本机CPU检查后的更正：A1–A3 改用 spatial 读出

本机 `cpu_tests.py` 第 3 项（动作只取决于亮patch位置的合成任务）发现：原 `readout='patch'`（可学习查询+可学习位置编码的注意力读出）在 3 个随机种子下注意力都停留在均匀 1/64，找不到亮块，只会背训练样本；位置编码初始化 0.02 或 1.0 都一样，与扩散后端、步数无关。改用空间softmax关键点读出（`readout='spatial'`，Levine 2016；robomimic/Diffusion Policy 的标准读出，坐标来自固定网格）后，3 个种子都能在开发集学会（误差 0.009–0.046，零命令 0.234），完整扩散模型开发集 0.048（CLS 0.289）。因此 A1–A3 改为 `spatial`；配置文件名沿用 `patch` 以免改动 `job.sh`。`readout='patch'` 代码保留未删。

## 一次作业里的 5 个配置（单 seed 42，诊断阶段）

| 配置 | 读出 | 历史 | 标签 | 回答的问题 |
|---|---|---|---|---|
| A0_cls_cmd8 | CLS | 无 | command8 | 缓存版复现旧瓶颈（对照） |
| A1_patch_cmd8 | spatial | 无 | command8 | 只换读出有多大作用 |
| A2_patch_hist_cmd8 | spatial | 有 | command8 | 再加速度信息 |
| A3_patch_hist_future | spatial | 有 | 未来位姿 | 换回可学标签 |
| A4_cls_hist_future | CLS | 有 | 未来位姿 | 未来位姿下读出的作用 |

## 预先固定的判定（看结果前写定）

- **位置门槛**：两个角色中，点预测的首命令位置和全段位置都 ≤ 0.9×零命令，且点预测−零命令的首命令位置逐窗差 bootstrap 95% 上界 < 0。旋转只报告。
- **只取最终端点**，不按开发集挑 checkpoint。
- **通过**：取通过的最佳配置，进入 LIBERO 最简单任务的工程回合（需另写在线接线，本包未包含），之后才进入老师的五条件/13 模型矩阵。
- **全部不通过**：Bridge 源端离线能力问题按负结果结案写入论文，不再开新的局部修补。
- 若 A1/A2 明显优于 A0，就直接证明“只读 CLS”是主要瓶颈之一，这本身就是论文里一个有说服力的消融。

## 运行

**1. 本机 CPU 检查**（不需要 GPU，临时文件写在本目录 `.tmp_tests`，不写 C 盘）：

```powershell
cd D:\ntu_related\dissertation\Zero_shot\deployment\patch_readout_v1
D:\ntu_related\dissertation\Zero_shot\training_cache\libero_native_env_v1\Scripts\python.exe cpu_tests.py
D:\ntu_related\dissertation\Zero_shot\training_cache\libero_native_env_v1\Scripts\python.exe reanalyze_old_runs.py D:\ntu_related\dissertation\Zero_shot
```

**2. 上传并提交**（本机 PowerShell）：

```powershell
scp "D:\ntu_related\dissertation\Zero_shot\cluster_packages\patch_readout_v1r1_20261101.zip" zixiao005@10.97.216.128:/projects/Zeroshot/
```

集群：

```bash
set -e
cd /projects/Zeroshot
test ! -e patch_readout_v1
unzip -n patch_readout_v1r1_20261101.zip
cd patch_readout_v1
bash submit_once.sh            # 先CPU检查数值与路径，再唯一一次sbatch
JOB=实际编号
tail -n 80 -F /projects/Zeroshot/logs/patch-readout-${JOB}.out
```

完成标记 `PATCH_READOUT_ALL_COMPLETE`。结果小包：

```powershell
$job = "实际编号"
$dest = "D:\ntu_related\dissertation\Zero_shot\reports\patch_readout_$job"
New-Item -ItemType Directory -Force -Path $dest | Out-Null
scp "zixiao005@10.97.216.128:/projects/Zeroshot/runs/patch-readout-$job/small_results.tgz" $dest
```

## 预算与资源

单 A40，4 CPU，48G 内存，Slurm 上限 3.5 小时。缓存约 5.5GB（集群 `/projects/Zeroshot/caches/`，不占本机）；建缓存预计 10–20 分钟；每个配置训练上限 25 分钟（10000 步×64），超时即按当前端点评估，不续训。

## 已知边界

- 本包由助手编写，**在云端环境里只跑通了 NumPy 部分**（几何、指标，并复现了 237424/237579 的原始数值）；PyTorch 部分因环境限制未执行，所以提交前请先跑 `cpu_tests.py`。
- `build_cache.py` 依赖集群上已有的 `bridge_command8_fresh_scaled_v1/`、`oxe_dino_main_preflight_v1/inputs/` 和原 Bridge 分片；它会逐条核对当前命令与已验收 command8 标签第一步一致（相位检查），不一致就停止。
- LIBERO 在线推理接线（实时 DINO token、位姿差、上一命令）尚未写，等离线门槛有结论后再做。
- 单 seed 诊断，不是老师矩阵的正式结果。

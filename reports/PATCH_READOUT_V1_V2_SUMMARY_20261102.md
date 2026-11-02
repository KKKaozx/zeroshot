# Patch readout v1 / v2 阶段总结（2026-11-01 至 11-02）

OXE(Bridge) 源端训练 → LIBERO 零样本。冻结 DINOv2 ViT-L/14 + 冻结 CLIP 文本 + Adapter + 扩散策略。
全部为单 seed 诊断，不是正式结果；LIBERO 只用于预先排除的工程任务，未参与任何训练或调参。

## 1. v1：让策略看到图像小块和速度信息（作业 239393）

代码 `deployment/patch_readout_v1/`，结果 `reports/patch_readout_239393/`。

- 发现旧 Adapter 只读 CLS，patch token 梯度恒为 0；改为空间 softmax 关键点读出（spatial）。
- 加入速度/历史输入（上一时刻位姿差 state_delta、上一条命令 prev_command），冻结特征一次性缓存。

| 配置 | 开发集首命令位置 点预测 / 零命令 | 离线门槛 |
|---|---|---|
| A0 CLS + command8 | 0.195 / 0.224 | ❌ |
| A1 spatial + command8 | 0.188 / 0.224 | ❌ |
| A2 spatial + 历史 + command8 | 0.130 / 0.224 | ✅ |
| **A3 spatial + 历史 + 未来位姿** | **0.080 / 0.187** | ✅ |
| A4 CLS + 历史 + 未来位姿 | 0.087 / 0.187 | ✅ |

结论：历史（速度）是主要收益来源，读出方式影响较小。

### A3 在 LIBERO 工程任务上（`reports/patch_readout_libero_engineering_20261102/`）

| 试验 | 末端净位移 x/y/z (m) | 成功 |
|---|---|---|
| run1 原样 | −0.04 / +0.20 / +0.15 | 否 |
| run2 夹爪状态固定为未知 | y +0.11，z +0.24 | 否 |

离线诊断：
- 绝对位姿输入 pose18 在 LIBERO 下超出 Bridge 范围：z 约 +21σ，x 约 −8σ。
- LIBERO 画面离 Bridge 远：DINO 最近邻余弦 0.36，Bridge 内部为 0.69。
- 在 LIBERO 画面上，"夹爪已知张开"会推动 +y 方向 3–11 mm；在 Bridge 画面上，这一输入几乎不起作用（<1 mm）。
- 推断：模型认不出陌生画面时，会转而依赖本体输入。

## 2. v2：去掉本体输入 + 通用图像增强（作业 241155，方向 1）

代码 `deployment/patch_readout_v2/`（计划见 `PLAN.md`），结果 `reports/patch_readout_v2_241155/`。

- B1：不使用任何本体输入（不读 pose18、current2、prev_command）。
- B2：在 B1 基础上，只保留 state_delta，丢弃率 0.5。
- B3：在 B2 基础上，缓存时为每张图生成 4 个版本（裁剪、颜色、模糊、低分辨率），训练时随机取一个；第 0 版是原图，评估只用原图。
- 本机建同样的缓存：10416 窗 × 4 版，28 分钟，5.3 GB，标签核对偏差为 0。增强效果预览图在 `reports/patch_readout_v2_local/augment_preview.png`。

### 三关判定（看结果前写定）

**第 1 关：Bridge 离线门槛。** 要求首命令和全段位置都 ≤ 0.9×零命令，且 bootstrap 95% 上界 < 0。

| 配置 | 开发集首命令 | 全段 | bootstrap 上界 | 夹爪准确率 | 结果 |
|---|---|---|---|---|---|
| B1 | 0.182 / 0.187 | 0.777 / 1.024 | +0.020 | 0.791 | ❌（死记训练集） |
| B2 | 0.123 / 0.187 | 0.690 / 1.024 | −0.039 | 0.789 | ✅ |
| **B3** | **0.108 / 0.187** | **0.650 / 1.024** | **−0.051** | 0.794 | ✅ 最好 |
| 参考 A3 | 0.080 / 0.187 | 0.588 / 1.024 | −0.077 | 0.821 | ✅ |

**第 2 关：LIBERO 离线检查**（`offline_gate2.py` → `gate2_offline.json`）。B3 勉强通过：散布比 0.71（要求 ≥ 0.7），漂移比 0.48（要求 ≤ 0.5）。但对照 A3 在"夹爪已知张开"设定下也能通过（0.89 / 0.39），所以这一关区分不了模型好坏。

**第 3 关：LIBERO 工程仿真**（`run_libero_engineering.py` → `libero_gate3/`）。要求单方向净位移 < 0.10 m。

| 试验 | 末端净位移 x/y/z (m) | 结果 |
|---|---|---|
| B3_run1 | +0.07 / −0.24 / −0.14 | ❌ |
| B3_run2 速度输入始终关闭 | +0.06 / −0.22 / −0.15 | ❌ |

- 开环重放（`offline_velocity_check.py`）：关掉速度输入后，步长从 12.7 mm 变成 12.5 mm，几乎不变；把速度输入反向后，步长降到 7.2 mm。
- 注意：`velocity_check.json` 里 off 一行的余弦值 0 是脚本 bug，用实际轨迹重算后是 0.65。
- 闭环试验确认，漂移主要由画面驱动，与速度输入基本无关。

## 3. 结论与下一步

- 方向 1 按预先写定的规则记为负结果：去掉本体输入并加通用增强后，Bridge 上可学（第 1 关通过），但 LIBERO 中仍然漂移（第 3 关不过）。
- v1 是被夹爪状态带偏，v2 改为被画面带偏。根本原因是 Bridge 与 LIBERO 之间的画面域差距；这类通用增强不足以弥补。
- 未解决的问题：
  - Bridge→LIBERO 的坐标轴对应未校准。
  - 老师的 5 条件/13 模型矩阵若采用缓存方式，步数是 10000 而不是 65536，需要在论文里说明并告知老师。
- 下一步方向待讨论。

## 4. 不在仓库里的大文件（只在本机 D 盘）

| 文件 | 位置 |
|---|---|
| A3 权重 | `reports/patch_readout_239393/A3_policy/policy.pt` |
| B3 权重 | `reports/patch_readout_v2_241155/B3_policy/policy.pt`（各约 530 MB） |
| 增强缓存 | `caches/patch_readout_bridge_dinov2_aug4_v2_local/`（5.3 GB） |
| 集群包 | `cluster_packages/patch_readout_v1*.zip`、`patch_readout_v2_20261102.zip` |

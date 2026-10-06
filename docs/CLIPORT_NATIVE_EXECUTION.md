# CLIPort 原生执行核查

2026-10-06。只使用现有训练演示 `000000-0.pkl`，没有模型、训练或测试集评估。这是单场景工程检查，不能作为正式任务成功率。

## 实际结果

| 检查 | 结果 | 能支持的结论 |
| --- | --- | --- |
| 作者资源与代码 | 固定提交的 239 个文件均与 Git blob 一致；UR5、吸盘、底座和方块可加载 | 该版本资源可用，任务/环境/primitive/奖励源码未修改 |
| 当前场景的作者脚本专家 | 7 个动作，总奖励 1，`done=True` | 当前环境的原生执行链路在一个训练场景上能工作 |
| 按旧演示 seed=0 重建 | 底座及初始语言相同，刚体位置最大差 0.3313 m | 旧场景不能仅凭 seed 在当前软件组合中完全复现；未执行旧动作 |
| 恢复旧记录中的初始刚体位姿 | 恢复后位置/旋转误差为零；前 6 个动作奖励一致；最后一次放置奖励 0，总奖励 5/6 | 记录动作格式可进入原生接口；完整历史物理回放仍未通过 |

恢复初始状态只调整已核对尺寸的刚体物体位置、姿态与零速度。底座必须先匹配，任务目标仍由作者任务根据底座生成；没有修改目标、奖励、动作或控制器。这个检查与按 seed 重建分开报告。

前四个执行后物体位置差约 0.0003 mm，第五个约 0.1 mm，第六个约 2.8 mm；最后约 12.7 mm、26.2°。最后一步是第一次奖励与旧记录不一致。尚未确定具体原因；源码、依赖、渲染及接触仿真差异仍需要区分，不能凭此确认某一项为根因。

证据：[cliport_native_execution_check.json](../reports/cliport_native_execution_check.json)。作者原始 `env.step` 不直接暴露 primitive 超时标志；初轮只记录空观测。后续检查通过观察包装器记录原生 primitive 返回值与 `check_grasp` 返回值，保持参数及返回值不变，得到可靠的该次调用日志；没有修改作者函数体。

## 使用的环境与限制

作者代码来自 [cliport/cliport](https://github.com/cliport/cliport/tree/2be5c47b5bb9bb7040ad90693288b87b1e18e7ad)，保存于 Git 忽略的 `training_cache/cliport_author`。本地独立 venv 为 `training_cache/cliport_replay_env`，使用 `--system-site-packages` 复用已有 Torch/PyBullet；没有修改原训练环境。

测试组合是 Python 3.10、NumPy 1.26.4、PyBullet 3.2.7、Gym 0.17.3、OpenCV 4.10.0.84。不是作者原 requirements 的完整旧版本复现，也不是通用干净环境。继承的 TensorFlow/ml-dtypes 与 venv 的 NumPy 存在依赖冲突；本检查未导入 TensorFlow，此 venv 仅用于原生仿真检查。

作者包入口会自动导入学习模型；检查脚本仅跳过该包入口，直接加载作者环境/任务模块。未替换其物理控制、任务逻辑或奖励。没有加载 CLIP、Adapter、扩散头或作者模型权重。

源码和 venv 保留约 419 MiB；未复制数据集，没有下载 GPU 库或模型权重。正式基准仍需固定部署环境并验证更多任务。

## 复现命令

先获取上述固定提交的作者源码及自带资源；不要直接把旧 requirements 装进项目训练环境。下列包是本次独立 venv 中测试的组合：

```powershell
python -m venv --system-site-packages training_cache/cliport_replay_env
training_cache/cliport_replay_env/Scripts/python.exe -m pip install numpy==1.26.4 gym==0.17.3 hydra-core==1.3.2 opencv-python==4.10.0.84 meshcat==0.0.18 kornia==0.4.1 transforms3d==0.4.2 scipy==1.10.1 matplotlib==3.7.5 imageio==2.34.2
```

在仓库根目录运行，输出路径必须不存在：

```powershell
# 默认严格检查 seed 重建；初始场景不一致则停止执行该演示。
training_cache/cliport_replay_env/Scripts/python.exe diagnostics/replay_cliport_native.py --author-root training_cache/cliport_author --dataset-dir D:/ntu_related/dissertation/dataset/cliport --output-json results/new-seed-check.json
# 作者专家针对当前场景生成动作，不执行旧记录动作。
training_cache/cliport_replay_env/Scripts/python.exe diagnostics/replay_cliport_native.py --author-root training_cache/cliport_author --dataset-dir D:/ntu_related/dissertation/dataset/cliport --output-json results/new-oracle-check.json --oracle-smoke
# 明确恢复记录初始物体位姿，再执行原始记录动作。
training_cache/cliport_replay_env/Scripts/python.exe diagnostics/replay_cliport_native.py --author-root training_cache/cliport_author --dataset-dir D:/ntu_related/dissertation/dataset/cliport --output-json results/new-state-replay.json --restore-recorded-reset
```

## 下一步的边界

当前已确认的是原生读取和一个当前场景的作者专家执行链路。下一步先核对旧演示生成时的作者提交和依赖记录；如果无法恢复旧环境，应把固定当前版本的专家场景作为独立工程对照，并保留旧回放失败记录。不要通过改奖励或放宽成功标准让旧回放“通过”。

这仍不等于 Bridge 相对连续动作能够转换成 CLIPort 世界坐标拾取/放置，也不说明学习模型已泛化。正式 OXE→CLIPort 的观测/动作合同尚待解决，不应接入失败的 Bridge 模型或直接开始完整训练。

## 最后一步的受控检查（同日后续）

没有在检查到的项目、数据及集群产物目录找到旧 Git 提交或完整依赖清单。`.hydra/hydra.yaml` 留下 Hydra 1.3.6 和旧工程路径；`demos.log` 为零字节，旧工程路径不存在。当前 Conda 环境元数据不能证明旧生成环境版本。上述搜索不等于全盘穷尽；若存在备份，可继续核对。

| 条件 | 最后一步抓取 | 原生 primitive 超时 | 最后一步奖励 | 总奖励 |
| --- | --- | --- | --- | --- |
| 恢复初始场景后连续执行旧记录动作 | 成功 | 否 | 0 | 5/6 |
| 前 6 个动作相同，仅在最后动作前恢复记录物体位姿并清零速度 | 成功 | 否 | 1/6 | 1 |

干预前物体位置最大偏差 2.814 mm、旋转最大偏差 2.298°；干预后均为零。两次都沿用同一个最后动作、作者任务目标、控制器和奖励。普通回放 7 个动作均记录到抓取成功且无 primitive 超时。

这支持“累积物体状态偏差参与了最后一次放置失败”，不是仅凭日志猜测。干预同时恢复位置、姿态并清零速度，不能把结果归因于某一种状态，也没有证明具体的依赖或数值根因。成功的干预结果不算无辅助任务成功率；完整连续旧演示回放仍未通过。

检查命令（仅工程诊断）：

```powershell
training_cache/cliport_replay_env/Scripts/python.exe diagnostics/replay_cliport_native.py --author-root training_cache/cliport_author --dataset-dir D:/ntu_related/dissertation/dataset/cliport --output-json results/new-intervention.json --restore-recorded-reset --restore-before-step 6
```

证据：[cliport_native_state_intervention.json](../reports/cliport_native_state_intervention.json)。本轮没有下载或更换依赖，没有训练模型。

当前可以确认单场景中的原生动作读取、抓取和执行接口具有运行证据；不能把这个结论推广为统一 OXE 动作转换、学习模型泛化或全部任务可靠。旧版本追查先限定在可找到的备份；若缺少备份，后续采用固定当前版本新生成的作者专家训练场景做独立工程对照，旧演示保持历史记录。不要持续试版本或放宽奖励来追求旧回放的表面通过。

## 固定当前版本的新演示读写与回放（同日）

**类型：无学习模型的单演示工程检查，不是泛化实验，也没有训练集/测试集模型预测比较。** 作者脚本专家在训练模式 seed=0 场景生成 7 个动作，使用作者 `RavensDataset.add` 保存一条完整演示；项目原生读取器加载后，在同一固定软件组合下从 seed 重建场景、连续执行记录动作，没有恢复初始物体位姿或中间状态。

结果：初始与各步物体位姿完全相同，7 个动作奖励与记录一致，最终奖励 1 且 `done=True`。8 个观测的 RGB 数组完全一致；深度按作者保存时的 `float32` 精度完全一致。原始渲染深度与保存深度并非逐位相同，属于作者显式精度转换，不能宣称所有原始浮点数无损保存。

新增数据约 49.23 MiB，仅保留于 Git 忽略目录 `training_cache/cliport_fresh_control`，带文件哈希及工程用途说明，没有覆盖旧数据。证据：[cliport_fresh_expert_roundtrip.json](../reports/cliport_fresh_expert_roundtrip.json)。未下载新依赖或模型权重。

生成命令要求新输出目录；专家未完成任务时不会将其保存为成功演示：

```powershell
training_cache/cliport_replay_env/Scripts/python.exe diagnostics/replay_cliport_native.py --author-root training_cache/cliport_author --dataset-dir D:/ntu_related/dissertation/dataset/cliport --output-json results/new-expert-export.json --oracle-smoke --save-oracle-dir training_cache/new-cliport-control
training_cache/cliport_replay_env/Scripts/python.exe diagnostics/replay_cliport_native.py --author-root training_cache/cliport_author --dataset-dir training_cache/new-cliport-control --output-json results/new-unassisted-replay.json
```

这只确认一个当前版本场景中的原生读写/相机记录/执行链路。CLIP、Adapter、动作头未参与，不构成其有效性的证据；OpenX 统一加载器与 OXE→CLIPort 动作转换也没有被验证。旧版本差异未确定根因，不再盲试依赖；后续工作回到模型输入/动作合同。

## 连续 TCP 控制工程对照（2026-10-06）

新增 `code/cliport_controller.py`，执行 1–16 个 `[局部位置/0.1m, 相对四元数 xyzw, 吸盘命令]` 目标；整个块使用一个固定输入 TCP 参考系。目标先全部检查，再执行；越界不裁剪，超时停止，不添加额外抓取辅助或物体状态恢复。移动完成后才应用吸盘命令，这还不是固定频率控制协议。19 项读取/转换/控制合同测试通过，单元测试不能替代物理任务验证。

仍用当前版本训练 seed=0 的一个工程场景，不使用 CLIP、Adapter 或动作头。在已有回放入口记录作者底层 `movep` 与吸盘命令，并区分移动事件和吸盘事件，避免开关吸盘时引入额外运动。

| 对照 | 总奖励 | 任务完成 |
| --- | --- | --- |
| 作者原生 primitive，记录底层命令 | 1 | 是 |
| 原样回放底层绝对命令，无相对转换或路径拆分 | 1 | 是 |
| 转为相对动作，经范围约束、路径拆分、连续控制器执行 | 1/3 | 否 |

连续控制完成前两次放置；第四个 primitive 边界本应奖励 1/6，实际为 0，随后语言目标与记录分岔，停止剩余命令。已执行目标未超时；物体状态已发生偏差。去掉吸盘事件的额外运动没有解决该失败。原样命令通过，支持问题位于本次改变的执行链路；尚未单独区分路径拆分、坐标数值误差和累积物理状态影响，不能声称已经确定某一根因。

作者失败抓取尝试中的命令最低为负高度；命令目标不能当作实际到达的 TCP。连续对照单独使用 z 下界 −0.1m 的作者命令包络；默认模型控制器仍为 z≥0。两种条件不能混用。对照还使用已知作者 primitive 边界结算奖励，不是正式连续策略评测。完整轨迹仅在忽略的工程缓存内保留，不上传数据或权重。

结论：连续控制接口尚未验收，不接入失败的 Bridge 权重，也不启动正式 rollout。下一项限定检查是分别隔离坐标转换与路径拆分，随后才确定实时反馈、控制周期及吸盘触发协议；避免反复调整奖励、工作空间或网络追求通过。证据：[连续控制对照摘要](../reports/cliport_continuous_tcp_control.json)。Bridge 离线泛化失败与此控制接口问题分别记录。

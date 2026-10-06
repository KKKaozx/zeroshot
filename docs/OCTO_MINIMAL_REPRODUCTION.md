# Octo 最小官方复现

2026-10-07。目标是运行已发布完整策略的官方第一项推理示例，作为正对照。暂停继续调整 CLIPort 专家回放或旧动作头。**安装188861与GPU推理188862均COMPLETED/0:0，完整JSON已由用户取回并保存至[结果报告](../reports/octo_inference_188862.json)。**

直接核对JSON：backend=gpu、devices=[cuda:0]、model_used=true、passed=true，原生动作[1,4,7]共28个有限数值，固定模型/代码revision与语言资源版本匹配；记录耗时26.05秒包括首次加载及采样，不能当作稳态控制延迟。没有训练、机器人执行或泛化评价。环境du -sh约5.7GB、项目约26GB；不是实际配额查询。TensorFlow GPU提示、Transformers缓存警告、缺少可选腕部观测提示未阻止该官方单图示例完成，不据此增加安装或观测修补。没有动作正确性参考，也没有用本项目CLIP/Adapter/动作头。

下一步使用现有Bridge子集建立原生单步离线对照：先核对固定Octo代码对Bridge动作的变换语义、图像/语言、坐标及夹爪开关约定，再评估316个训练分区窗口和55个开发分区窗口，并按演示拆分。该导出是稀疏相邻状态对，不是连续轨迹，不能将相邻记录拼成4步历史或4步动作真值；仅比较预测第一步与同一源时间的单步标签。保留Octo checkpoint动作统计，不用子集统计替换它，也不将7维Euler增量直接与项目16步8维四元数轨迹混比。未启动该评估，无需再安装/下载/训练；Bridge可能参与预训练，结果不作为未见数据泛化证据。

## Bridge 单步离线评估包

集群评估188886已完成，316/55窗口覆盖通过，官方模型冻结；三采样种子均值从用户日志核对并保存[日志摘要](../reports/octo_bridge_188886_log_summary.json)。训练分区位置1.395cm优于静止1.920cm，旋转3.090°差于保持姿态2.702°；开发位置1.453cm优于静止1.820cm，旋转2.935°差于保持姿态2.446°，夹爪准确率95.15%、平衡94.12%优于一直关闭83.64%/50%。完整集群JSON尚未取回，不能判断逐演示及采样种子稳定性。此处“训练分区”仅是项目既有划分，不表示Octo此次参与训练。结果支持位置/夹爪的离线改善，不能声称旋转、机器人成功、未见数据泛化或单个Adapter模块归因通过。下文未运行描述为当时准备记录。

已准备 `scripts/cluster/octo_bridge_single_step.py` 和 `.sh`，上传包 `results/octo_preparation/octo_bridge_single_step_v1.zip`。本地实际读取现有两个TFRecord、验证文件SHA256和17/3条演示的316/55个窗口；逐窗口运行固定官方源文件中的原始 `relabel_actions` 函数，371项完全一致。该本地检查仅提取该函数执行，不冒充完整Octo环境或GPU模型运行。原夹爪标签来自此前完整有效序列的作者扫描核验，不在稀疏样本上重新扫描；没有重读完整原始序列证明所有末尾处理一致。RGB已是256×256×3 uint8，原样输入，使用真实语言 `sweep into pile`。运行无增强、历史长度1、不拼接稀疏记录，保留checkpoint动作统计，评价4步输出中的第0步。

姿态评价是把预测Euler分量增量加到当前实测Euler，再计算与下一实测姿态的SO(3)角误差；不用Euler分量差绝对值当旋转角。真实标签代替预测时误差接近零、夹爪准确率1；±π分支跨越检查和100例独立SciPy姿态核对通过。详细结果见[本地检查](../reports/octo_bridge_single_step_local_check.json)。单步开发静止基线1.81968cm/2.44578°；一直关闭夹爪46/55=83.636%，平衡准确率50%。不能与先前16步9.61cm/12.94°/91.9%混比。

集群只提交一次GPU作业，无安装或权重/数据下载。批大小8，尾批重复最后样本仅用于固定编译形状；统计仅保留真实样本，覆盖计数仍316/55。采样seed0/1/2，各种子RNG再按分区编号及批起点fold_in；这不是三次训练。逐采样种子和逐演示计算指标，再平均指标，不平均动作。输出小JSON及仅第一步预测的压缩NPZ，不保存图片、权重或4步真值。30分钟是Slurm时间上限，尚未测得实际运行时长。

固定Octo的[Bridge标准化代码](https://github.com/octo-models/octo/blob/241fb3514b7c40957a86d869fecb7c7fc353f540/octo/data/oxe/oxe_standardization_transforms.py)说明其 `bridge_dataset` 实际使用更近期自有Bridge发布，当前子集来自较早OXE Bridge V2。因此此次是原生动作语义下的诊断参照，不是同分布论文复现；输出好坏均不能单独定位Adapter或证明未见演示泛化。当前只证明源标签计算一致，没有实际机器人坐标/单位物理标定或控制执行验收。

本地上传：

```powershell
scp "D:/ntu_related/dissertation/Zero_shot/results/octo_preparation/octo_bridge_single_step_v1.zip" "D:/ntu_related/dissertation/Zero_shot/results/octo_preparation/octo_bridge_single_step_v1.sha256" zixiao005@10.97.216.128:/projects/Zeroshot/
```

集群提交：

```bash
cd /projects/Zeroshot
sha256sum -c octo_bridge_single_step_v1.sha256 && unzip -n octo_bridge_single_step_v1.zip &&
OCTO_BRIDGE_JOB=$(sbatch --parsable octo_bridge_single_step_v1/octo_bridge_single_step.sh) &&
printf '单步评估作业：%s\n' "$OCTO_BRIDGE_JOB"
```

随后查 `sacct -j "$OCTO_BRIDGE_JOB" --format=JobID,State,ExitCode`，日志位于 `logs/octo-bridge-step-JOBID.out`，完整报告位于 `runs/octo-bridge-step-JOBID/report.json`。`passed`仅表示覆盖/有限值/流程检查通过，模型是否优于基线由报告实际误差判断。子集GPU评估尚未运行。

用户提供的集群作业 188824 报告确认 x86_64、RTX A6000 可见，项目 apparent use 23.18GiB，inventory_passed=true；octo_inference_ready=false。旧环境 JAX0.4.13/Flax0.7.0/TF2.13 与官方依赖不同，另建 `/projects/Zeroshot/envs/octo-small-v1`。200GB 配额按 200,000,000,000 bytes 做规划；原预检用 200GiB，余量略高估，新脚本已纠正。仍未读取实际配额。

安装作业 188827 在 pip 依赖解析阶段失败：未锁定的新版 wandb 要求 protobuf≥5，与 TF2.15/protobuf4.23.4 冲突；尚未进入模型下载或推理。修复为 wandb==0.16.6，其 PyPI 元数据支持 Linux/Python3.10 下 protobuf≥3.19,<5。补锁后仍需集群安装验证，不能称修复已运行通过。旧依赖等待任务 188828 应取消，再按 afterok 重新提交安装和推理；复用已创建的独立环境。

重试 188832 仍在依赖解析阶段失败：tensorflow-metadata1.14.0 要求 protobuf≥3.20.3,<4.21，与人为固定的4.23.4冲突。保留 wandb0.16.6，将 protobuf 改为3.20.3；检查其满足日志列出的 TF、TFDS、TensorBoard、wandb、TFHub、Orbax、TFMetadata 全部 protobuf 范围。此检查不等于完整依赖安装或运行通过。取消旧等待任务188833，再复用安装环境提交作业。

安装188838成功：pip check通过，官方模型与语言/图片资源已缓存，独立环境约5.2GB、项目约25GB（du -sh）。GPU推理188839失败且model_used=false：JAX找不到匹配cuDNN并回退CPU。安装清单为nvidia-cudnn-cu11 9.10.2.21，固定jaxlib使用cuDNN8 ABI；JAX0.4.20的cuda11_pip仅限定cuDNN>=8.8，未排除9。新增约束nvidia-cudnn-cu11==8.9.6.50，确认PyPI存在x86_64 Linux wheel（约700MB），通过原CPU安装作业修复，再提交GPU推理。环境、权重继续复用；修复后GPU可用性仍待实跑，不能把运行库错误归因于模型预测。

固定代码 `241fb3514b7c40957a86d869fecb7c7fc353f540`，模型 `rail-berkeley/octo-small-1.5` revision `dc9aa3019f764726c770814b27e4ab0fc6e32a58`，T5 资源 revision `a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1`。checkpoint 546,696,551 bytes。模型和环境只在集群下载；T5 只下载配置及 tokenizer，语言编码器参数来自 Octo checkpoint。官方代码未替换为本项目网络。

CPU 安装作业使用官方 requirements 加关键兼容约束，固定 JAX0.4.20 CUDA11 GPU wheel、NumPy1.24.3/Flax0.7.5/TF2.15；记录 pip freeze 并执行 pip check。其余传递依赖仍由 pip 解析，不能称所有依赖均已锁定或安装验证通过。安装前要求 15GiB 规划余量，安装后检查新增占用不超过 15GiB；此检查不限制安装期间的峰值占用。保留原有环境，不下载完整 OXE 数据。本地包只有脚本和配置。

GPU 作业复现 [官方 notebook 第一项](https://github.com/octo-models/octo/blob/241fb3514b7c40957a86d869fecb7c7fc353f540/examples/01_inference_pretrained.ipynb)：Bridge 示例 JPEG、RGB256×256、`pick up the fork`、历史长度1、有效时间掩码、seed0，使用 `bridge_dataset` 动作统计反归一化。GPU 阶段离线读取缓存；JAX 回退 CPU 则失败。检查输出 `[1,4,7]` 且全部有限，保存原生动作和统计。未运行 notebook 下载整条远程数据的第二项，不使用项目 `[16,8]` 或 CLIP 预处理。

在本地 PowerShell 上传：

```powershell
scp "D:/ntu_related/dissertation/Zero_shot/results/octo_preparation/octo_official_smoke_v1.zip" "D:/ntu_related/dissertation/Zero_shot/results/octo_preparation/octo_official_smoke_v1.sha256" zixiao005@10.97.216.128:/projects/Zeroshot/
```

在 VS Code 集群终端执行；安装成功后 GPU 推理才具备运行条件：

```bash
cd /projects/Zeroshot
mkdir -p logs baseline_setup
sha256sum -c octo_official_smoke_v1.sha256 && unzip -n octo_official_smoke_v1.zip &&
OCTO_SETUP=$(sbatch --parsable octo_official_smoke_v1/octo_setup.sh) &&
OCTO_RUN=$(sbatch --parsable --dependency=afterok:"$OCTO_SETUP" octo_official_smoke_v1/octo_inference.sh) &&
printf '安装作业：%s\n推理作业：%s\n' "$OCTO_SETUP" "$OCTO_RUN"
```

在同一终端查询：

```bash
sacct -j "$OCTO_SETUP,$OCTO_RUN" --format=JobID,State,ExitCode
tail -n 60 "/projects/Zeroshot/logs/octo-setup-$OCTO_SETUP.out"
tail -n 60 "/projects/Zeroshot/logs/octo-inference-$OCTO_RUN.out"
```

排队时日志可能不存在，正常。安装失败时推理可能保持依赖等待，先检查安装日志；不要重复提交。完成后取回 `baseline_setup/octo-inference-JOBID.json` 小报告，无需下载权重。CPU 下载成功与 GPU 加载成功分别验收，运行前不宣称复现通过。

推理通过仅证明官方模型加载、GPU执行和有限值动作输出，没有动作数值参考对照，也没有机器人任务成功率。Octo 预训练包含 Bridge、BC-Z、Fractal、Language Table，在本地 Bridge 上表现好不能证明未见数据泛化。后续比较需统一划分、动作合同和评价指标。来源：[官方代码](https://github.com/octo-models/octo)、[权重](https://huggingface.co/rail-berkeley/octo-small-1.5)、[论文项目](https://octo-models.github.io/)。

# 同Bridge作者实现参照：接口与资源预检

最新状态：203498为COMPLETED/0:0，3分31秒，在gpu-a6000-1运行；六份小结果已本地下载核验。168输入身份、四更新日志、训练统计归一化及反变换通过，峰值保留显存5.91GiB；参数/EMA变化按报告核对，未下载权重独立验算。这只验证资源与接口，不判断训练拟合或泛化。固定8192更新的正式作者参照包现已准备，尚未提交；见[正式协议与提交命令](BRIDGE_AUTHOR_REFERENCE_TRAINING_20261014.md)。以下下载与准备内容保留为历史记录。

集群Bash打包，明确只选择author结果与导出报告，不带浮点输入缓存：

```bash
cd /projects/Zeroshot/runs
tar --exclude='*.pt' --exclude='*.ckpt' -czf /projects/Zeroshot/bridge_author_preflight_results_203498.tar.gz bridge-author-preflight-203498/author bridge-author-preflight-203498/inputs/export_report.json
```

本地PowerShell：

```powershell
scp zixiao005@10.97.216.128:/projects/Zeroshot/bridge_author_preflight_results_203498.tar.gz "D:/ntu_related/dissertation/Zero_shot/reports/"
tar -xzf "D:/ntu_related/dissertation/Zero_shot/reports/bridge_author_preflight_results_203498.tar.gz" -C "D:/ntu_related/dissertation/Zero_shot/reports/"
```

以下提交和准备状态是历史记录。

当前提交状态：用户已提交预检203498，尚未收到Slurm状态或结果。仅此一个预检，无正式训练作业。固定作业号检查命令：

```bash
squeue --start -j 203498
sacct -j 203498 --format=JobID,State,ExitCode,Elapsed,NodeList
if [ -f /projects/Zeroshot/logs/bridge-author-preflight-203498.out ]; then
  tail -n 15 /projects/Zeroshot/logs/bridge-author-preflight-203498.out
fi
```

状态配对已在两个初始化上显示部分收益，但仍未优于静止。下一阶段先检验作者训练/生成实现能否接收同一Bridge监督，而非继续扫状态接口参数。当前仅准备一个资源预检，不包含正式训练启动器。

## 本次到底运行什么

直接使用已有集群源码中固定提交`5ba07ac6661db573af695b419a7947ecb704690f`的作者图像策略、训练Workspace、AdamW、学习率调度、EMA、归一化器和DDPM采样。GPU进程调用作者`TrainDiffusionUnetHybridWorkspace.run()`实际更新4次，不另写一个完整训练循环。作者代码不修改。[作者训练入口](https://github.com/real-stanford/diffusion_policy/blob/5ba07ac6661db573af695b419a7947ecb704690f/diffusion_policy/workspace/train_diffusion_unet_hybrid_workspace.py)

作者原配置包含双帧历史、epsilon目标、512/1024/2048宽度、AdamW、warmup/cosine及EMA。本次迁移保持其算法组件，修改任务输入与预检规模，明确记录差异。[作者配置](https://github.com/real-stanford/diffusion_policy/blob/5ba07ac6661db573af695b419a7947ecb704690f/diffusion_policy/config/train_diffusion_unet_hybrid_workspace.yaml)

数据接入和空离线Runner是薄封装：复用原168输入身份，不进行环境交互；作者原策略本身没有项目的八层跨模态Adapter。整套参照同时改变多个模块，其结果不能直接证明某一模块故障。它是Bridge适配参照，不能称完整论文复现，也还不能称成功基线。

## 固定输入与监督

| 字段 | 本次定义 | 防止混淆 |
| --- | --- | --- |
| 划分 | 原201663的128训练窗/64演示、40开发窗/10演示 | 不换窗、不扩数据；预检不评价开发策略 |
| 图像 | 同一224×224浮点letterbox画面，反转已有CLIP均值/标准差回RGB[0,1] | 不重新缩放、不裁剪、不量化为uint8；作者图像归一化再映射至[-1,1] |
| 当前状态 | 同一观测索引XYZ与旋转6D，共9维；沿用原训练统计构造后由作者训练统计归一化 | 不用未来状态、演示编号或窗口进度 |
| 语言 | 5个训练任务规范化指令的冻结CLIP文本EOS池化768维特征 | 来自已有本地权重，缓存约15KB；没有在线上传、开发拟合或文本编码器训练 |
| 观察历史 | 一帧当前观测，`n_obs_steps=1` | 不把重复当前画面当真实历史，不偷加前后帧 |
| 动作 | 原16个未来相对位姿的前7维：XYZ/0.1m + XYZW四元数 | 首目标仍是t+1；作者输出反归一化后回到这7维，计算物理位置时再乘0.1m |
| 夹爪 | 本项只做位姿参照，不监督或判断夹爪 | 不用它报告抓取成功 |

作者动作和状态归一化器只拟合128训练窗。开发沿用训练统计，不能按开发范围重新拟合。语言为固定条件向量，使用恒等归一化。所有16目标均有效，不引入padding、过去动作或未来观测条件。

## 与项目的实际差异

| 模块 | 项目当前参照 | 作者Bridge适配参照 |
| --- | --- | --- |
| 视觉与融合 | 冻结视觉CLIP + 可训练八层Adapter | 可训练作者ResNet/SpatialSoftmax观察编码器，图像/状态/固定语言向量合并 |
| 文本读出 | token序列经跨注意力 | 已有CLIP的固定EOS池化向量作为low_dim输入 |
| U-Net | 256/512/1024宽度，时间嵌入256 | 作者512/1024/2048宽度，时间嵌入128 |
| 扩散目标 | x0/sample | 作者epsilon |
| 动作训练尺度 | 既定XYZ/0.1m与四元数 | 同一原动作再经作者训练集limits归一化；输出逆变换 |
| 优化 | Adam + 中途降低学习率 | 作者AdamW、500更新warmup、cosine和EMA |

以前的同权重U-Net/调度数值一致性审计仍有效；本项检验的是完整作者训练链在相同数据上的行为，不重复声称已找到U-Net公式错误，也不能将整套差异视为单变量比较。作者参照有效后才能设计固定观察条件下的模块替换。

## 验收与预算

本地已完成168原始输入核验、图像转换和状态对齐。实际执行作者纯归一化函数并独立检查limits公式；动作归一化/反归一化最大分量差2.384e-7，改变40开发目标不会改变训练统计。这个小差值是数值往返误差，不是预测成绩。

启动参数及Bash语法通过；本地未安装缺失的作者依赖、未执行完整作者GPU入口、未训练策略或保存PT。作者完整环境只在集群复用现成`official-pusht-v1`，预检成功与否尚未知。额外公开源码核验文件仅20,745字节，没有新增模型或数据下载。

集群预检先用现有`multitask-preflight-v1`读取/核对原TFRecord并缓存约100MB浮点输入，再用已有`official-pusht-v1`运行作者Workspace。检查4次真实优化、编码器和U-Net梯度/参数变化、EMA变化、训练统计/反归一化、100步采样的[2,16,7]有限输出和显存。只在训练样本上做接口采样，预检不计算开发泛化成绩，也不保存权重。

预检脚本申请A6000、4CPU、22GiB内存、30分钟；这些是资源上限，不是排队或完成承诺。启动时按200,000,000,000字节规划限额预留新增1GiB及15GiB余量，`du`不是实时配额查询。缓存只写集群，勿下载整个inputs目录到本地。

后续正式参照的暂定有界预算为初始化43、batch8、512轮、8192更新，每窗512次曝光，与原项目样本曝光预算一致；沿用作者优化与EMA。它不等于论文原3050轮预算。正式启动器尚未准备，需先看预检的接口及资源结果再冻结训练和最终评价，不自动排入长作业。正式若未拟合，先报告预算/观察与任务适配限制；不把作者也失败解释成问题无解。

## 上传和提交一个预检

ZIP29,875字节（约29.2KiB）、10文件，SHA256：`c417ae6c08c68fc63e170ff7e0f1ec55f17395fa68938d3aa39791c629d1fa26`。不含图像、原始数据、模型权重或新环境。

Windows PowerShell：

```powershell
scp "D:/ntu_related/dissertation/Zero_shot/cluster_packages/bridge_author_preflight_v1.zip" "D:/ntu_related/dissertation/Zero_shot/cluster_packages/bridge_author_preflight_v1.sha256" zixiao005@10.97.216.128:/projects/Zeroshot/
```

集群Bash：

```bash
cd /projects/Zeroshot
mkdir -p logs
sha256sum -c bridge_author_preflight_v1.sha256 &&
unzip -n bridge_author_preflight_v1.zip &&
AUTHOR_BRIDGE_CHECK=$(sbatch --parsable bridge_author_preflight_v1/preflight.sh) &&
echo "作者Bridge预检：$AUTHOR_BRIDGE_CHECK"
```

```bash
squeue --start -j "$AUTHOR_BRIDGE_CHECK"
sacct -j "$AUTHOR_BRIDGE_CHECK" --format=JobID,State,ExitCode,Elapsed,NodeList
tail -n 12 "/projects/Zeroshot/logs/bridge-author-preflight-$AUTHOR_BRIDGE_CHECK.out"
```

只有一个预检，无正式依赖作业。如果失败，保留报告/日志核对原因，不跳过数据或接口断言；没有新环境安装脚本。完成后只下载author报告、配置、normalizer及小预测和inputs/export_report.json，原浮点image.npy缓存留在集群。

工件：[冻结ZIP](../cluster_packages/bridge_author_preflight_v1.zip)、[协议](../cluster_packages/bridge_author_preflight_v1/protocol.json)、[本地原始输入验收](../reports/bridge_author_contract_20261014/export_report.json)、[独立数值与启动验收](../reports/bridge_author_contract_20261014/verification.json)、[生成器](../scripts/build_bridge_author_preflight_v1.py)。

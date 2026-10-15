# 最后一次当前设置诊断：真实历史输入的有界对照

本轮目的：检验原202278设置缺少真实过去画面是否影响开发演示动作预测。保留128训练/40开发窗口，不换任务、不扩大数据、不改动作表示或去噪方法。本轮有明确结束条件，不将每个阴性结果继续扩展成下一次调参。

最终状态：204258共17个文件已独立验收。真实历史比精确重放的控制位置平均退步4.79%，旋转也退步，仅4/10演示位置改善，三个候选收益条件均失败。**本项及当前设置追加诊断关闭，不复核新种子或扫描帧间隔/参数。**[最终结果](BRIDGE_HISTORY_GENERALIZATION_RESULT_204258.md)、[本阶段总结与下一阶段范围](BRIDGE_DIAGNOSTIC_PHASE_CLOSE_20261015.md)。下文提交/待下载状态均为历史记录。

2026-10-15正式更新：204258已收到COMPLETED/0:0、A6000、1小时30分29秒日志，真实历史组训练1.965mm/0.542°、拟合通过，开发8.749cm/15.614°。新控制具体数字尚待报告，但包装器成功要求其位置在旧8.349cm±0.1cm内，因此真实历史不可能满足事前至少10%位置改善条件。本方案按约定关闭，不追加训练或种子复核；下一步仅下载两组结果独立复算。[终端结论及固定下载命令](BRIDGE_HISTORY_GENERALIZATION_RESULT_204258.md)。下文未提交为此前准备状态。

2026-10-14执行更新：预检204150已由用户提供COMPLETED/0:0、A6000、12分18秒的日志，进入包装器报告结束位置。短更新后的`gate False`是拟合指标，不是接口验收失败或历史效果阴性结论。按冻结入口，退出0表示包装器的初始重放、共同起点、前16控制损失和历史梯度等断言已执行通过；具体差值仍待读取预检JSON，尚未下载预测独立复算。下面的新命令先读取并打印关键预检字段，只在报告通过时提交正式一组配对训练；正式结果尚不存在。

## 已完成输入审计

只读实核74条训练/开发演示、42个既有原始分片和168个固定窗口。原始文件/当前输入/状态身份核验一致；当前图像处理与原缓存在≤1e-6分量差范围内一致。同一记录的t-1图像和当前图像使用同一冻结预处理函数，没有构造未来观测或解码保留测试演示。

| 固定数据 | 窗口数 | 有真实t-1的窗口 | 演示起点，重复当前帧 |
|---|---:|---:|---:|
| 训练 | 128 | 64 | 64 |
| 开发 | 40 | 30 | 10 |

有真实历史的94个窗口，前后RGB均不完全相同。对应原始当前/过去观测的末端位置变化中位数：训练11.45mm、开发17.28mm。没有时间戳，不能将它们解读为固定时间间隔或速度。图像不同不等于CLIP已编码有效运动信息，更不等于能改善泛化。

参考：早先[16窗历史拟合](BRIDGE_HISTORY_PAIR_20261011.md)只证明历史分支在已见样本可用，未检验这个128/40划分的泛化。官方成功PushT检查点有两个观察时刻，是本轮动机之一；本项目使用共享Adapter的历史差分融合，不能称为官方完整多帧策略复现。

[审计结果](../reports/bridge_history_generalization_preparation_20261014/input_audit.json)、[事前冻结计划](../reports/bridge_history_generalization_preparation_20261014/protocol.json)、[只读审计脚本](../scripts/audit_bridge_history_inputs_20261014.py)。审计不加载CLIP/权重，不进行优化。早期审计的可选渲染因本地Torch环境缺Matplotlib而退出，核心审计已完成；已改为PIL输出小预览并重新全量执行成功，没有安装依赖或修改数据。

## 唯一变量与共同预算

两组都保留相同历史投影、真实当前末端状态端口、参数量和两次共享Adapter前向：

- A：过去图像槽重复当前图像。
- B：过去图像槽使用同一原始记录的t-1；t=0时重复当前。

融合均为`current + W(previous-current) + current_state_projection`，W与原202278一样为1024×1024无偏置零初始化。当前图像、语言、当前状态、夹爪测量及16个未来到达位姿目标相同；不向模型输入演示编号、起点、真实下一动作或未来状态。

初始化43、同batch8/混合批次表、同8192更新与噪声/时间步种子、4096次1e-4再4096次1e-5、Adam/裁剪1全部保留。冻结CLIP和夹爪；沿用多尺度x0去噪器、原生100步随机DDPM和原位置/四元数尺度。三次采样种子1101/1102/1103相同。只评价位姿，不报告抓取成功率。

实现复用冻结202278训练入口。静态AST核查确认优化更新`advance`、评估`check`、梯度/状态复制与哈希函数均未改；变化仅为读取历史图像、提供过去视觉token及元数据。实际新历史加载器已在本地全168窗运行，所有过去图像身份和74个边界重复验收通过；Python入口导入与参数解析、两份Bash语法均通过。本地尚未运行完整模型/GPU预检，不将源码核查当作数值结果。

GPU预检每组共64次短优化更新（16次共同前段、三个16次恢复/学习率检查分支），验收初始化预测/条件与原控制重放、前16次控制损失、梯度、相同起点、参数范围和恢复路径。预检的拟合误差不是效果判定，也不代替8192次正式对照。

## 事前验收与停止条件

正式两组都必须满足原训练拟合门槛：每个采样平均位置≤5mm、旋转≤5°，≥90%训练窗在三个采样中均达标，平均终点位置≤1cm、旋转≤10°。重复当前组须复现202278控制（开发位置差≤0.1cm、旋转差≤0.5°）。预检重放/输入/梯度错误先修复，不能把接口失败当历史无效。

“值得复核的明确改善”必须同时满足：

1. 全40开发窗最终路径位置误差比配对控制降低至少10%。
2. 全40窗最终路径旋转平均误差不恶化。
3. 至少6/10条开发演示的位置误差降低。
4. 两组训练拟合和控制重放均通过。

10%和6/10是预先设置的实际筛选标准，不是统计显著性。同时报告第16目标、逐任务/演示、存在历史的30开发窗与10个起点窗，不能挑子组/检查点代替全40主结果。若要称开发协议通过，还须整体位置和旋转均优于静止；这些反复使用的开发数据仍不是未见最终测试。

**不满足以上候选条件，就结束当前设置下的追加诊断。**不追加轮数、帧间隔、融合宽度、种子或数据源。若训练拟合门槛失败，则结论为该固定预算下尚不能评价历史收益，同样不自动追加训练。转向另行冻结的“单任务、同场景、独立演示优于静止”目标。

只有明确候选成立才允许最多一次匹配初始化42复核；复核未复现则结束，不挑更多种子。这不是继续无期限试验的入口。

## 小包与资源

冻结包`bridge_history_generalization_v1.zip`：50,644字节，10个文件，SHA256为`f126f364fca1f330160364c9f2ad4a32add7d4804bc722a72ca8542511346154`。复用集群既有数据、CLIP、环境与源模块，不下载数据/模型或安装包。预检不保存checkpoint，25分钟运行上限；正式配对上限2小时30分，不是运行/排队时间预告。

正式最多两份训练参数检查点，合计约1.01GiB，只保留在集群；脚本保守预留2GiB新增产物和15GiB余量，以200GiB作为规划阈值。`du`检查不等于实时配额查询。本地只新增输入元数据、少量原始帧预览和小包，不复制数据集、图像tensor缓存或权重。

预检204150已完成。下文预检提交步骤保留作追溯；**正式训练尚未提交**。

本地PowerShell：

```powershell
scp "D:/ntu_related/dissertation/Zero_shot/cluster_packages/bridge_history_generalization_v1.zip" "D:/ntu_related/dissertation/Zero_shot/cluster_packages/bridge_history_generalization_v1.sha256" zixiao005@10.97.216.128:/projects/Zeroshot/
```

集群Bash：

```bash
cd /projects/Zeroshot
mkdir -p logs
sha256sum -c bridge_history_generalization_v1.sha256 && unzip -n bridge_history_generalization_v1.zip && HISTORY_PREFLIGHT=$(sbatch --parsable bridge_history_generalization_v1/preflight.sh)
echo "历史对照预检：$HISTORY_PREFLIGHT"
squeue --start -j "$HISTORY_PREFLIGHT"
```

查看预检：

```bash
sacct -j "$HISTORY_PREFLIGHT" --format=JobID,State,ExitCode,Elapsed,NodeList
tail -n 20 "/projects/Zeroshot/logs/bridge-history-preflight-$HISTORY_PREFLIGHT.out"
```

通过后读取小报告确认门槛，再给出正式提交命令；不同时排入正式训练，以免预检失败后继续排队。SHA与包内所有文件已复读核验；尚无本轮真实模型效果结果。

## 204150完成后的正式提交

下面命令先读预检报告、检查通过状态并打印原控制重放/损失差值，再提交一组正式对照。没有依赖已结束作业的调度记录，使用保存报告验收，避免历史作业被Slurm清理后出现dependency错误。

```bash
if /projects/Zeroshot/envs/multitask-preflight-v1/bin/python - <<'PY'
import json
r = json.load(open('/projects/Zeroshot/runs/bridge-history-preflight-204150/report.json'))
assert r['passed'] and r['stage'] == 'preflight' and r['error'] is None
assert r['initial_conditions_and_predictions_exact'] and r['real_history_projection_gradient_exists']
print('执行预检通过：', r['passed'])
print('两组初始条件和预测完全一致：', r['initial_conditions_and_predictions_exact'])
print('原控制重放差异：', r['original_initial_replay'])
print('前16次控制损失最大差：', r['first16_control_loss_max_difference'])
print('真实历史分支有梯度：', r['real_history_projection_gradient_exists'])
PY
then
    HISTORY_FIT=$(sbatch --parsable --export=ALL,HISTORY_PREFLIGHT_JOB=204150 /projects/Zeroshot/bridge_history_generalization_v1/fit.sh)
    echo "历史正式对照：$HISTORY_FIT"
    squeue --start -j "$HISTORY_FIT"
fi
```

两组各8192次更新，使用同一预定预算与已有数据，开发仅在最终权重评估；作业上限2小时30分，不是预计耗时。若没有预先定义的明确改善，本设置诊断到此结束，不再追加训练。

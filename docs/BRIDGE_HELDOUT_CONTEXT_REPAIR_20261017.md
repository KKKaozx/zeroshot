# 208034定位新增入口的上下文执行差异，恢复原方式

**208056完整结果已本地复核。** 11文件齐全，CPU独立复算全部总体/演示/任务/目标指标，与报告最大差2.54e-13；旧4窗上下文/预测均0差异。coverage位置13/13演示、5/5任务优于同批不动，旋转6/13演示、3/5任务；旋转加权位置12/13、旋转10/13演示更好，任务分别5/5与4/5。两模型的演示等权与窗口等权平均位置/旋转均优于同批不动，但西兰花任务旋转加权整体接近不动、coverage旋转更差。范围仅同任务新演示离线泛化，无抓取成功结论。详见[完整核验](BRIDGE_HELDOUT_RESULT_208056_20261017.md)。本次评估收尾，不重跑、不用已消耗测试集调参。

以下为此前记录。

**208056同批基线已收到（用户从报告提取，演示等权）。** 静止7.779cm/12.742°；coverage4.766cm/12.045°，位置/旋转平均误差约降低38.7%/5.5%；rotation_weight6.094cm/10.264°，约降低21.7%/19.4%。两组均有新13条独立演示上的平均优势，范围为五个已见任务的离线预测。完整报告/数组尚待本地复核，尚不声称逐任务全面改善、统计显著或抓取成功。下一步只核验已有证据，不重跑或据此调参。详见[208056结果](BRIDGE_HELDOUT_RESULT_208056_20261017.md)。

以下为此前记录。

**208056已完成（用户终端证据，COMPLETED/0:0，3分15秒，A6000）。** 两模型旧4窗上下文及预测重放差均为0。新13条演示/60窗已评分：按演示等权，coverage路径4.766cm/12.045°，rotation_weight路径6.094cm/10.264°。同批静止基线尚未贴出，完整报告和数组尚未本地复核，因此暂不判断是否优于静止基线。新集合已使用，禁止重跑或据此调参；下一步仅读取现有报告并核验保存证据。下方提交和诊断状态为历史记录。

**修复后独立留出评估已提交208056（用户编号）。** 状态与成绩尚未收到；只跟踪这一作业。先CPU核验208034实际保存证据，再运行原入口和原1e-4重放门槛，之后才评分新13演示/60窗。两检查点、数据、协议和采样不变，无训练。日志`/projects/Zeroshot/logs/bridge-heldout-fixed-208056.out`，结果目录`/projects/Zeroshot/runs/bridge-heldout-fixed-208056`。

状态：用户报告208034 COMPLETED/0:0、2分钟、A6000。终端配对结果支持恢复原上下文执行方式。完整报告/保存数组尚未下载本地；修复入口将在集群独立核对全部16份数组和报告后，才允许执行原评估。没有新13演示评分或新训练，修复作业尚未提交。

## 已得到的证据

| 旧4窗计算方式 | coverage上下文最大差 | coverage预测最大差 | rotation_weight上下文最大差 | rotation_weight预测最大差 |
|---|---:|---:|---:|---:|
| 原no_grad和参数标志，旧缓存或两种重新编码 | 0 | 0 | 0 | 0 |
| 失败入口的inference_mode和全部requires_grad=False，旧缓存或重新编码 | 9.536743e-7 | 0.0001168251 | 1.430511e-6 | 0.0001377165 |

原168行/失败64行状态计算在测试的原执行方式下也都重放为0差异。这将当前失败定位到**新增入口的上下文执行方式组合**；并未进一步分离inference_mode与参数标志哪一项触发具体底层数值路径，不声称已确定某个PyTorch内核故障。CLIP重新编码、测试的状态数组行数没有解释本次预测差异。

这是新入口引入的数值重放兼容问题；恢复后的0差异只验证旧4窗一致，不能说明历史厘米级泛化误差已解决。原1e-4门槛没有放宽，原模型值、数据、噪声、采样和测试划分没有改变。

## 最小修复与入口证明

原`bridge_heldout_evaluation_v1`代码/56MB归档、冻结协议、原始数据、两个checkpoint保持原样。新增约7KB包装入口动态加载原评估器，仅覆盖`contexts`：在原调用方的外层inference_mode中显式进入`inference_mode(False)`，用`no_grad`计算，并恢复原Adapter/decoder/history/state的requires_grad标志。没有优化器、反向传播或参数更新；CLIP和夹爪仍保持原标志。上下文计算完返回调用方原作用域，原采样器本来使用的inference_mode不变。

本地使用微型合成CPU线性运算验证：调用方外层模式恢复、上下文内部关闭梯度并离开inference_mode、原参数标志恢复、输出和原no_grad逐元素相等、参数值不变及无梯度。实际模型/权重未加载，无GPU执行。原上下文helper可直接取得不带装饰器的原函数体，原入口及其所有其他函数未改。CPU语法/包装作用域检查、Bash语法和ZIP内容检查通过；不是实际GPU完整评估通过。

集群包装入口先核对208034实际报告、诊断源码和合同快照、208005未评分记录，逐份复算16个旧预测/上下文数组差异。原方式各组合须真实等于0，失败方式须重现原阈值以上差异。证据不符就停止。之后继续原始主程序，两个旧4窗严格重放都通过后才会执行新13演示/60窗；原共享原子claim仍阻止重复评分。

额外保存`repair_entry.json`、`repair_runner_snapshot.py`、`repair_contract_snapshot.json`，并在最终报告标明实际包装入口身份。原`runner_snapshot.py`仍记录未改的底层入口，不将它误认为没有运行包装修复。

证据：[修复合同](../cluster_packages/bridge_heldout_context_repair_v1/repair_contract.json)、[本地检查](../cluster_packages/bridge_heldout_context_repair_v1/local_validation.json)、[旧诊断协议](BRIDGE_HELDOUT_REPLAY_FAILURE_208005_20261017.md)。

## 只上传代码补充包并提交一次

包6,827字节，SHA256 `9cd3582fff189afc5e4b72d8d8aeb88a13a79e90602c566b35530c5e2aa4989b`。不重传原数据、模型、环境或缓存，不重跑208034。

本地PowerShell：

```powershell
scp "D:/ntu_related/dissertation/Zero_shot/cluster_packages/bridge_heldout_context_repair_v1.zip" "D:/ntu_related/dissertation/Zero_shot/cluster_packages/bridge_heldout_context_repair_v1.sha256" zixiao005@10.97.216.128:/projects/Zeroshot/
```

集群终端：

```bash
cd /projects/Zeroshot
sha256sum -c bridge_heldout_context_repair_v1.sha256 &&
unzip -n bridge_heldout_context_repair_v1.zip &&
bash -n bridge_heldout_context_repair_v1/evaluate_fixed.sh &&
HELDOUT_FIXED=$(sbatch --parsable bridge_heldout_context_repair_v1/evaluate_fixed.sh) &&
echo "修复后独立留出评估：$HELDOUT_FIXED"
```

一个A6000作业，原20分钟上限。实际数据检查、两模型严格重放、新60窗评分都在同一作业；不再另排预检。输出`/projects/Zeroshot/runs/bridge-heldout-fixed-JOBID`，日志`/projects/Zeroshot/logs/bridge-heldout-fixed-JOBID.out`。

评分对象仍是五个已见任务的新演示；两个固定模型都须报告同批静止、按窗/演示等权路径误差、终点、全部任务和演示。评分后这批数据不再用于调参，不宣称物理抓取或未见任务成功。

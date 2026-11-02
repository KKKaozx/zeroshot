# 208005：新评估入口的旧窗口重放检查未通过

**208034完成并定位新入口的上下文执行方式差异（终端配对证据）。** 两模型在原no_grad/参数标志下，缓存或重新编码均上下文/预测差0；失败方式分别产生0.000116825与0.000137717预测差。原门槛未变，这不解释历史泛化误差。约7KB代码包装已准备：先在集群复核208034全部16数组，再恢复原上下文作用域，保留原入口/协议/两个检查点/严格重放和原子claim，之后才评分新13演示。尚未提交修复作业，也无新泛化成绩。见[定位和修复命令](BRIDGE_HELDOUT_CONTEXT_REPAIR_20261017.md)。

**旧窗口一致性诊断已提交208034（用户编号）。** 当前未收到Slurm状态或诊断报告；只跟踪该作业。仅旧4窗八预定组合，分离保存上下文/旧缓存/重新编码、计算模式与状态数组算术，精确相同输入复用采样。原门槛和两个模型不变，不训练、不评价新13演示。日志`/projects/Zeroshot/logs/bridge-replay-diag-208034.out`。

状态：用户报告208005为FAILED/1:0、1分39秒、A6000。失败发生在coverage模型旧4窗预测比较：动作张量最大绝对差`0.00011682510375976562`，原门槛`0.0001`。没有修改门槛或原13演示验收协议。该差异不是以cm为单位的泛化误差，需保存分量后才知道最大差来自xyz、四元数或夹爪。

原始源码在此断言后才创建新测试评分claim并执行13演示推理，因此依据报错位置可判定尚未进入新测试评分；实际失败report及claim状态将在只读诊断中再次核对。完整失败report尚未在本地取得。前面的cuDNN/cuBLAS注册警告不是直接退出点，不据此安装或改动环境。

## 已确认的入口差异，原因仍待分离

原207558/207746的上下文计算位于`observation_guard`的`torch.no_grad()`中，保留Adapter/decoder/history/state参数的原requires_grad标志；其输入来自已经核验保存的CLIP token缓存。新入口重新编码图像/文本，使用`torch.inference_mode()`计算上下文，并把所有参数的requires_grad设为False。新旧执行方式的数学目标相同，但是否因此触发不同数值路径、造成当前差异，需要实际配对结果，不能凭差值小就视为通过。

检查点SHA和参数初始化、原始输入检查已在报错前通过，且上下文最大差未超过原1e-4门槛；源码并未在失败前保存这次上下文/预测数组，所以已有日志不足以定位分量。新增诊断会先保存每个结果，再报告比较值，避免再次丢失失败证据。

## 新增一个仅旧4窗的有界诊断

新目录`bridge_heldout_replay_probe_v1`不修改原评估包、协议、两个模型或门槛，也不会自动继续新测试。保持原checkpoint、旧4窗、batch2、三采样种子和DDPM100：

1. 精确使用原保存上下文，检查采样器本身的重放。
2. 读取已有168-token缓存，核对原输入身份、CLIP哈希、分词及tensor哈希，用旧4条精确token重新计算上下文。
3. 分别在原no_grad/参数标志和失败inference_mode/参数标志下计算同一条件。
4. 重新编码同四张旧图像，分离编码方式和上下文方式；图像批2、文本批16，与原缓存一致。
5. 比较原168行状态特征与失败64行当前观察算术的前4行。只读取当前观察，不读取或评分新留出目标。

八个预定组合只用于旧4窗。精确上下文哈希和参数标志都相同的组合复用已有采样，不重复GPU采样。保存每个组合的旧预测/上下文以及xyz原单位与cm、物理位置差、四元数、旋转角差和夹爪分量。比较不通过也保存结果，并且始终不评价新13演示。

实际CPU语法/无训练调用检查、分量单位与四元数符号等价检查、Bash语法和归档内容检查通过；GPU与缓存实际检查仍待集群执行。不宣称修复已证实，也不将这件事升级为历史厘米级泛化误差的根因。

## 上传小诊断包

包9,068字节，SHA256 `70faa2ee8a8e2c48eefbaa749efb8d4b25ffc4aa1656a4f66bb01f9d9c63048f`；无数据、参考数组或权重，只复用集群已有文件。A6000单卡、4CPU、22GiB内存、10分钟上限，尚未提交。

本地PowerShell：

```powershell
scp "D:/ntu_related/dissertation/Zero_shot/cluster_packages/bridge_heldout_replay_probe_v1.zip" "D:/ntu_related/dissertation/Zero_shot/cluster_packages/bridge_heldout_replay_probe_v1.sha256" zixiao005@10.97.216.128:/projects/Zeroshot/
```

集群终端：

```bash
cd /projects/Zeroshot
sha256sum -c bridge_heldout_replay_probe_v1.sha256 &&
unzip -n bridge_heldout_replay_probe_v1.zip &&
bash -n bridge_heldout_replay_probe_v1/probe.sh &&
REPLAY_DIAG=$(sbatch --parsable bridge_heldout_replay_probe_v1/probe.sh) &&
echo "旧窗口一致性诊断：$REPLAY_DIAG"
```

只跟踪这个诊断，不重新提交原208005。结果出来后再按证据修正入口，不放宽原门槛或删除新演示。

证据：[诊断合同](../cluster_packages/bridge_heldout_replay_probe_v1/probe_contract.json)、[局部CPU验证](../cluster_packages/bridge_heldout_replay_probe_v1/local_validation.json)、[原冻结评估协议](BRIDGE_HELDOUT_EVALUATION_20261016.md)。

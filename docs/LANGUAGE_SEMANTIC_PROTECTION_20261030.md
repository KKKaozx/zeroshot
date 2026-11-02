# 动作、对象与关系保护检查接入候选流程

2026-10-30。完成8条训练原指令的明确字段绑定和保守形式检查，并接入可选候选导入入口。**本轮只使用已有输出，没有新增模型调用或训练；尚未接入原生约束解码，也没有覆盖全部3686条指令。**

## 检查保护什么

原指令绑定数据身份、文本ID和手工声明的角色字段。检查完整句式中的角色位置，不仅检查关键词是否都出现。

| 原始含义 | 保护规则 |
|---|---|
| 银锅到炉具右下方、蓝三角到蓝方块左侧 | 固定移动对象与参照物的角色；换位即不匹配 |
| 黄物体从桌上到抽屉顶部 | 保留来源、目的地和top-of表达，不替原句解释具体物理表面 |
| 推绿圆靠近黄六边形及蓝三角 | 保留push、比较级closer/nearer与两个参照物的and关系；不接受move、slide、nudge、or或漏掉对象 |
| 葡萄进入紫碗、RXBAR进入顶层抽屉 | 保留内部关系与目标；to不能替代in/into，upper不能替代top |
| 用毛巾擦紫碗 | 保留wipe及工具角色；不接受clean、取毛巾的额外步骤或工具/对象交换 |
| 苹果靠近Red Bull罐 | 保留near而非on/next-to；Red Bull空格形式是明确声明的品牌别名，不当作新增红色属性 |

所有实体别名均显式列入profile并至少锚定原句中的一个名称。受支持的词序可组合识别，例如把目的地方位移到句首、将工具短语前置、或交换“双参照and”的列举顺序。新增动作、否定、未声明对象/数量和不受支持句式无法通过完整匹配。

“未匹配”涵盖语义变化与合法但尚未支持的表达，不能自动判为语义错误；“匹配”只说明符合已声明的受控形式，仍不能替代文本审阅、原始标注正确性或实际场景核验。重复原句另有`original_reuse`标记。所有输出均保持`semantic_review_status=pending`、`usable_for_training=false`。

## 已完成的检查

在已有7B输出的40条开发样例上：25条先前文本接受候选全部匹配，15条先前拒绝候选全部未匹配。**这些例子参与规则设计，不能称为独立测试准确率或全池泛化结果。**

5项保护器检查通过，包含8个对象/参照/工具角色交换反例、19个否定/额外动作/关系漂移反例、8个组合词序与别名正例，并检查未知表达/未建profile的训练指令继续待审阅，以及错数据身份/源文本/实体绑定拒绝。另有3项候选工作流检查通过，验证可选接线、原始失败保留和禁止自动批准。没有重跑旧训练、生成或完整训练器套件。

## 文件与入口

- 检查器：`code/protected_language_frames.py`，仅标准库。
- 8条手工profile：`configs/protected_training_language_frames_20261030.json`。
- CPU复核入口：`tools/audit_language_semantic_protection_20261030.py`。
- 40条逐候选结果：`reports/language_semantic_protection_20261030/audit.json`。
- 实际可选导入结果：同目录`ingested_candidate_review/`，复用原7B原始输出，未重写旧审阅。

可选导入增加`--protected-frames`，默认行为不变。启用时核对当前训练目录SHA、数据身份和源文本，再附加逐候选保护结果；不会产生训练库或将匹配结果改成语义批准。下面命令已经执行，输出目录现已存在，**不要原样重跑覆盖**：

```powershell
training_cache\libero_native_env_v1\Scripts\python.exe tools\prepare_language_candidate_review_20261030.py ingest --requests reports\language_candidate_requests_20261030 --raw-outputs reports\local_7b_qualification_20261030_loopback_repair\raw_outputs.json --protected-frames configs\protected_training_language_frames_20261030.json --output reports\language_semantic_protection_20261030\ingested_candidate_review
```

当前仅8条有手工profile，另3678条没有，不能自动解析或批准。原先25条文本接受候选和1条完整5改写的进展保留，本轮没有新训练标签。

下一步是把这些已检查的受控形式用于新的有界候选生成机制，固定源字段、输入与预算后，再验证实际解码是否遵守约束；当前不声称这一步已经实现或获得新生成质量证据。主实验的反事实场景依据、执行合同及尺度要求仍未解除。

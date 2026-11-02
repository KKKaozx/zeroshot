# 语言增强资格回退与候选审阅入口

2026-10-30。已完成本地实现与8项CPU检查；真实增强库仍未就绪，未生成文本、训练模型或提交集群。此项直接补四增强条件的启动缺项，不新增Bridge诊断。

## 实际行为

新增 `reviewed-training-language-bank-v2`，保留v1原行为。新库仍须精确覆盖同一训练分区，绑定数据身份、演示group key及原文；转述条件仍要求每条5个互异、已审阅的等价候选。没有降低语义准入标准。

| 窗口状态 | v2处理 |
|---|---|
| 同组不同目标的donor及实际场景不匹配证据齐全；所需当前夹爪命令已知 | 按既定比例抽取反事实，监督维度使用零位移、单位相对四元数及保持命令 |
| 无已审阅的场景不匹配 | 记录资格不足，回退正样本，不删窗 |
| 有场景依据，但夹爪有监督且当前保持命令未知 | 同样回退，不从实测开度或未来动作猜命令 |
| Language Table仅XY监督 | 不要求不存在的夹爪命令；不改被屏蔽的Z/旋转/夹爪目标 |
| 尾窗或部分维度无监督 | 保留原mask；仅替换有效元素，不改被屏蔽的目标元素 |
| 声称有资格但donor、证据或命令格式不合法 | 拒绝，不将坏记录悄悄转为合格或忽略 |

新库中每窗需有 `counterfactual_review`：`status`取`available`或`unavailable`，均需非空`evidence`；`unavailable`还需非空`reason`且无donor。`available`必须有符合原场景审阅规则的donor。`hold_gripper_command=null`明确表示未知，实际mask若监督夹爪就回退。代码验证的是记录完整性，不会自动证明场景和指令语义正确。

`counterfactual_only`与`combined`保持相同负样本请求、资格与donor抽样；后者回退时继续沿用与`paraphrase_only`配对的正样本改写。两者都不改原图像、观测输入、batch数或源窗口顺序。0.25只针对合格窗口的请求概率，不能宣称全池有25%负样本。

两个真实训练入口均保存实际动作种类和请求、合格、不合格、回退次数及原因。实际负样本率应以`counterfactual / counterfactual_opportunities`计算，合格曝光率以`counterfactual_eligible / counterfactual_opportunities`计算；这些附加计数不能与动作种类一起相加当样本总数。v2规则进入恢复合同，旧v1检查点不能静默切换。

## 候选准备与导入

`tools/prepare_language_candidate_review_20261030.py`已从本次核对的3686条训练专用任务中，为四来源各选2条此前未生成过的指令，生成首批8个请求。选择按指令SHA排序，不看验证或测试结果。请求绑定当前组合数据身份、训练代表窗、原文和提示SHA；没有复用已失败预检来反复测试，也没有启动全池生成。

提示v3针对既有实际错误明确保持in/inside、on、near/contact及动作顺序，不猜含糊动词含义；无法忠实产生5条时允许弃答。这只是候选提示，**尚无新生成结果，不能宣称质量已改善**。

实际请求：`reports/language_candidate_requests_20261030/generation_requests.jsonl`，清单与身份：同目录`bundle.json`。工具只支持准备和导入，不调用模型。后续本地生成器应输出`local-paraphrase-batch-outputs-v1`，包含相同`dataset_identity`、清单中的`requests_sha256`作为`request_bundle_sha256`，以及`model.repository/revision/local_only=true`；`outputs`需精确覆盖这8条，每条有`instruction_id/request_sha256/raw_generation`。

候选导入入口（以下原始输出路径是待产生的文件，不是现成结果）：

```powershell
training_cache\libero_native_env_v1\Scripts\python.exe tools\prepare_language_candidate_review_20261030.py ingest --requests reports\language_candidate_requests_20261030 --raw-outputs <实际本地生成输出.json> --output <新的审阅目录>
```

导入保留格式失败、重复和弃答原文，不自动重试填数。格式通过仅获得`semantic_review_status=pending`，全部`usable_for_training=false`，不输出训练库。缺失、重复、额外ID或不同提示/数据身份直接拒绝。

## 验证与剩余边界

8项检查涵盖：配对抽样及正样本回退、未知夹爪/XY/尾窗、全局RNG及输入不变、重建控制器重放及合同拒绝、坏证据/命令拒绝、v1与原指令控制兼容；还直接抽取两个实际训练代码块执行到文本干预边界，验证标签与曝光计数接线。候选准备检查覆盖不重复既有指令、身份篡改拒绝、原始失败保留及禁止自动批准。使用合成场景声明，不是实际场景审核、模型更新或优化器恢复试验。

首次测试因系统临时目录权限在写fixture前失败，改为项目内临时目录后8项通过；没有申请额外权限或改动系统目录。

真实语义库、任务/目标及场景不匹配依据、真实资格覆盖仍待完成；源到LIBERO执行合同和最终训练尺度仍独立阻塞。原数据与已提交部署包未重建。实现状态见`configs/augmentation_implementation_status_20261030.json`，检查记录见`reports/language_eligibility_validation_20261030.json`。

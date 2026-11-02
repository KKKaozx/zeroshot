# 首批8条训练指令候选生成与逐项审阅

2026-10-30。**生成已完成，但8组均未通过严格结构检查，当前方案不能扩展到3686条全池。** 本批收口，保留原始失败；没有放宽准入、修复格式后送入训练、重试补数或启动机器人训练。

## 实际执行与结果

使用既有本地冻结 `Qwen/Qwen2.5-1.5B-Instruct`，revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`。既有权重和配置SHA逐项核对，所有加载设为local-only、无远程代码；本机RTX 4060完成8次贪心生成，含检查和加载共32.97秒，峰值reserved3.088GiB。没有下载模型、安装依赖、访问集群或更新参数。

请求来自已核对的当前训练指令目录，四来源各2条，未用验证/测试指令；这8条与旧预检保留的3条当前池指令不同。提示使用预先保存的v3，不在看完输出后改提示重试。提示和输入均不同于旧预检，不能将结果当作同指令配对的提示效果比较。

| 来源与原指令 | 严格结果 | 原始输出的主要问题 |
|---|---|---|
| Bridge：put the silver pot to the lower right of the burner | 拒绝 | 5个字符串被包在无效JSON花括号内；多条把右下方改成炉内、炉上、附近 |
| Bridge：move the yellow object from the table to the top of the drawer | 拒绝 | 无效JSON；5条均照抄原句 |
| Language Table：push the green circle closer to the yellow hexagon and blue triangle | 拒绝 | 无效JSON；把靠近改为前方/相邻等，并丢失明确推动动作 |
| Language Table：move the blue triangle left side of the blue cube | 拒绝 | 无效JSON；多条添加then place/position/align/set的额外或不明确后续命令 |
| BC-Z：place grapes in purple bowl | 拒绝 | 输出5个结构化对象而非句子，增加quantity=one，有一条把in改成on |
| BC-Z：wipe purple bowl with towel | 拒绝 | 无效JSON；照抄，或将wipe改成更泛的clean/不同的scrub，将towel泛化为cloth |
| Fractal：move apple near redbull can | 拒绝 | 无效JSON；把near改为on/beside/adjacent，另有照抄 |
| Fractal：place rxbar blueberry into top drawer | 拒绝 | 无效JSON；把RXBAR Blueberry替换为blueberry red apple，多条改成shelf或on |

7组是JSON语法错误；葡萄组虽然是合法JSON，元素类型不是字符串。严格解析器均返回不合格，没有候选训练准入。允许弃答的提示也没有得到弃答输出。

## 40个输出位置的人工式文本核对

直接阅读原始输出并逐位置记录Codex判断；这不是人工签字或场景核验。对无效JSON中的可读字符串仅作手工转录分析，并逐字重建原始输出核对；未将转录内容写成可训练候选。

| 文本判断 | 数量 |
|---|---:|
| 文字有潜力，仍属于结构失败组 | 3 |
| 照抄原指令，不是新改写 | 7 |
| 含义变化或等价性不足，保守拒绝 | 25 |
| 非字符串结构对象 | 5 |
| 总输出位置 | 40 |

3条文字参考分别保留了右下方关系、蓝三角相对蓝方块的左侧关系、苹果靠近Red Bull罐的关系。它们都没有通过完整组的结构准入，不构成3条训练批准；完整5条合格的指令数仍为0，真实训练库仍未产生。

同时暴露格式与语义问题，故单独修JSON解析器不能解决。此结果只约束这一本地模型、提示和8条输入组合，不是否定语言增强研究假设，也未验证所有本地语言模型都不适用。下一项生成方案应改变产生忠实候选的机制或模型能力，并用已有失败案例作为固定验收内容；不再围绕这8条自动重试、凑足5条或扩大当前方案。目标动作执行合同仍是独立阻塞。

## 防止重复与证据

已保存 `reports/language_candidate_attempts_20261030.json`，将8条登记为`closed_rejected_group`。后续准备工具默认排除登记的已尝试指令；生成入口也会在加载模型前拒绝已登记的批次。纯CPU的2项候选工作流检查在增加排除逻辑后通过，没有再运行模型。当前工具与实际运行时的快照分别保留，未追改当时证据。

- 原始输出：`reports/language_candidate_generation_20261030/raw_outputs.json`，SHA `d4044fcadf1c22dbd12c746b3a6e2b5b54a8d3d0a595ab3b9457b46cc37e1c26`。
- 实际生成参数、版本、计时：同目录`report.json`；实际运行代码`runner_snapshot.py`，原始请求同目录保留。
- 严格导入与完整原文：`reports/language_candidate_review_20261030/candidate_review_tasks.jsonl`及`report.json`。
- 40位置判断：同目录`manual_text_review.json`；可复核记录脚本`tools/review_language_candidate_batch_20261030.py`。

所有40位置及8个完整组均`usable_for_training=false`。本次0机器人模型调用、0训练更新、0新增集群作业，8次冻结本地语言模型生成；没有修改原始数据或已有训练权重。

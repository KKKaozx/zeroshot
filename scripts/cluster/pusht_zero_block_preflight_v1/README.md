# Push-T无注意力块控制：仅预检

目标是让后续0块与既有8块对照可比较，不先改变现有模型。CLIP冻结，两组沿用同一CLS/patch等权汇总、输出LayerNorm、1024→64投影及agent_pos2列。这里“无Adapter”精确定义为去掉全部8个跨注意力/FFN残差块；保留旧末端LayerNorm。LayerNorm和投影仍可训练，共67648个前端参数。文本缓存保留但零块分支不再使用语言；Push-T仅一句固定指令，这只是同任务机制对照，不是替代正式语言策略。

先用旧seed42创建作者后端，核对初始化SHA；seed43按原构造器创建完整8块前端，随后只删除块，避免直接构造零层改变随机数消耗并连带改变投影初始化。保留LayerNorm/投影对象和值以及CPU随机状态，生成记录。无预训练原生/项目/Panda权重加载，不从215999截断后继续训练，也不拿截断旧模型直接评分。

复用原90演示/10726窗/11356帧、训练统计、完整CLIP缓存、batch64/micro8、epsilon目标/余弦DDPM100/AdamW/EMA、原512400学习率时域与warmup500。重用核心代码逐字节锁定；layer数改变会改变参数量和块内dropout，属于整个8块堆叠的消融，不能唯一归因注意力某个算子。

预检只做8个独立更新，加一次完整raw/EMA/Adam/LR/EMA计数/所有RNG保存恢复重算，再用真实batch38做一次隔离探针。共10次更新函数执行，其中1次是重算、1次是一次性partial探针。最后仅两个训练窗做DDPM100输出形状/1:9切片检查，无开发分数、无闭环/最终50、无性能结论。所有预检权重均不作正式训练起点，只有后续重新初始化才可比较。

A40×1/4CPU/40GiB/15分钟硬上限，内部800秒，no-requeue。近期A6000环境CUDA失败而原诊断在A40成功，因此默认A40。只测零块耗时，不重跑8块训练或资源预检；按84168相同更新数估算零块正式预算，排队/完整检查点I/O/闭环另计。仍需训练原扩散后端，不能保证因为去掉Adapter就几分钟完成。正式预算与结果采用口径需在真实耗时报告后明确；本包没有正式训练入口和自动提交。

源八块报告固定为215999，主模型与CLIP不改；同初始化种子不意味着两个不同结构功能相同。历史八块训练使用A6000，硬件及数值差异须在正式解释中保留，不能把单seed差别当成已完成严格多seed因果证明。

提交：bash /projects/Zeroshot/pusht_zero_block_preflight_v1/submit_once.sh

完成验证：/projects/Zeroshot/envs/official-pusht-v1/bin/python /projects/Zeroshot/pusht_zero_block_preflight_v1/verify_preflight.py /projects/Zeroshot/runs/pusht-zero-preflight-实际编号

replay.pt是预检2更新时的完整恢复测试文件，留服务器；验证器核对SHA，无需下载。输出另有report.json、training_data.json和小smoke_actions.npz。需要至少8GiB可用盘，不删除旧检查点，不安装/下载依赖。

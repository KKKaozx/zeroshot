# 冻结DINO-L/14同预算视觉骨干对照

依据老师第六封邮件，在项目CLIP前端明显落后原生后，只替换视觉骨干。原CLIP8组215999和原生212158直接复用，不重新训练；冻结CLIP主项目不修改，不扩大/重抽数据，不追加零块/旁路/R3M分支。新DINO-L/14仍完全冻结，原8层Adapter、池化、1024→64投影、固定CLIP文本、agent_pos、作者U-Net各层FiLM保持。

数据90训练演示/11356帧/10726窗口、帧ID/原始像素及normalizer精确沿用。只用221055已完成的11.13GiB DINO缓存，引用220882实际资源测量，生成的checkpoint使用独立DINO格式并嵌入完整cache/backbone绑定SHA；旧CLIP、原生、零块和更换缓存的状态拒绝恢复。无需重复资源更新或缓存。Worker/client与221055只有protocol文件改名为visual_protocol.json，推理/通信操作不变。视觉协议原字节保留，sources条目属于历史缓存包；正式完整文件散列由project_contract.json独立检查。

正式新建共同seed42作者后端与seed43原8层前端，禁止任何已有学习端点初始化。AdamW/lr1e-4/betas.95,.999/weight_decay1e-6、原cosine horizon512400+warmup500、原epsilon-DDPM100、batch64/micro8含尾批38、独立shuffle/扩散/前端RNG、EMA power.75均原样保留。不是把25小时训练压缩到短schedule。

总有效84168更新，分42084/84168两段连续保存完整raw/EMA/Adam/LR/EMA计数/Python/NumPy/CPU/CUDA RNG。第一段内原首两步磁盘恢复audit必须逐状态/损失精确，额外重算一次不计入预算；每2000步覆盖本段latest.pt并用原子writing保存。第一段不评价、不挑checkpoint；核验后手工提交第二段。恢复带原合同与完整DINO绑定，不允许重置优化器、改变余弦进度或自动重投。

两段各A40/4CPU/40GiB/20小时硬上限/no-requeue。220882纯训练估算共25.46小时、每段约12.73小时，另有I/O/保存/第二段固定闭环，不是完成时间承诺。原CLIP使用A6000，新DINO使用A40，保留硬件和单训练seed的归因限制。

提交前与作业内检查du项目占用/文件系统空间，保守200十进制GB项目预算内至少18GiB余量；不是权威账户quota查询，不自动删任何文件。221055缓存、原数据、两个基准端点都保留。新正式完整checkpoint磁盘实际大小尚未实测。

最终仅84168 EMA在原dev20种子5000000..19、原DDPM采样计划和8步执行、max300步上评价一次。固定前两场视频，原max overlap reward含义不改。报告DINO/CLIP/原生均值和满奖励比例、DINO减CLIP配对增益与场景bootstrap区间；复用开发场景/一个训练seed，区间不含训练seed不确定性，不是final50确认或机器人抓取成功。final50不访问，不按dev中途调参、加预算或换候选，也不自动替换主CLIP模型。

DINO原生ImageNet归一化与最终LayerNorm随骨干变化；为保留画面，沿用旧224resize/224crop而非DINO默认256resize/224crop。结果属于该明确表示/预处理边界下的冻结视觉替换对照，不把任何差距归为Adapter单独因素。

第一段提交：bash /projects/Zeroshot/pusht_dino_training_v1/submit_part1_once.sh
第一段完成核验：/projects/Zeroshot/envs/official-pusht-v1/bin/python /projects/Zeroshot/pusht_dino_training_v1/verify_dino.py /projects/Zeroshot/runs/pusht-dino-part1-实际编号
第二段需核验后手工：bash /projects/Zeroshot/pusht_dino_training_v1/submit_part2_once.sh /projects/Zeroshot/runs/pusht-dino-part1-实际编号/latest.pt

若第一/二段中断，保留latest及日志，先检查错误和完整侧车，再另行审阅显式恢复。正常提交防重复；不会自行重新开始或更改预算。

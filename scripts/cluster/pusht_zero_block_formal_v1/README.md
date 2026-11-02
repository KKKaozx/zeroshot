# Push-T零块正式控制：只新增一组

承接219260成功预检。只训练一个零块控制，现有8块215999固定结果直接复用，不重训8层。CLIP冻结、保留原CLS/patch汇总/输出LayerNorm/1024→64投影与agent_pos；零块不利用语言，只作Push-T固定指令下的机制消融，不替代正式多指令模型。

从头seed42相同后端、seed43原8层完整构造后移除8个注意力/FFN块，逐项核对219260的共同初始化，绝不截断已训练215999，不加载预检权重。数据90演示/10726窗/11356帧、完整缓存、batch64/micro8/末批38、样本与噪声计划、epsilon/DDPM100/AdamW/EMA及原512400学习率时域不变。新控制完整84168更新，与旧8层同预算；没有更少步数的成绩替代比较。

原训练入口通过列明的有限替换派生，真实训练循环与评估数学不变。初始2步完整恢复检查和每2000/终点原子检查点保留，raw/EMA/Adam/LR/EMA计数/全部RNG独立零块协议绑定。允许显式--resume自己的完整检查点，禁止8层/原生/预检权重；没有自动重投。若中断，先检查已有计数与输出，不从头重跑。

预检每步0.547230543秒，84168纯训练估算12.794小时，另有载入、保存/哈希、闭环及排队；申请A40/4CPU/40GiB/20小时硬上限，不保证结束时间。8层原训练A6000、此控制A40，硬件/数值和单训练seed限制必须报告；不能宣称唯一注意力算子因果。

只在84168固定EMA评价一次原同20开发场景/同噪声流/两帧16→8/300步，固定前2视频，无中间选模，不访问final50。旧8层闭环不重跑。比较零减八奖励：均差>=0.05、场景配对bootstrap95%下界>0且满奖励比例不低，才称此设置下证据支持移除8块；对称负差/上界<0且满比例不高才支持保留8块；其余为未定。开发集已复用，三个分支都不自动追加预算/seed/变体，也不替换主语言模型或开放最终集。

提交时先调用原219260输出验证器一次，绑定原报告/初始化与新协议SHA，不重跑预检。请保留原pusht_zero_block_preflight_v1及runs/pusht-zero-preflight-219260目录。

提交：bash /projects/Zeroshot/pusht_zero_block_formal_v1/submit_once.sh

完成验证：/projects/Zeroshot/envs/official-pusht-v1/bin/python /projects/Zeroshot/pusht_zero_block_formal_v1/verify_formal.py /projects/Zeroshot/runs/pusht-zero-formal-实际编号

输出report/comparison/development/latest.json、latest.pt、updates.jsonl、绑定、startup_checks和前2固定视频。大权重留服务器。只做这一组控制，不重开空间旁路、不补已存在FiLM，不改原模型或依赖。

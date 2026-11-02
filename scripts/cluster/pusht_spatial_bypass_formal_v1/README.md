# Push-T空间旁路正式配对对照

承接已通过服务器验收的217778。复用预检模块、归一化、原作者训练核心和在线CLIP接线，均逐字节锁定。两组从原215999完整端点重新恢复，不接续预检的8次更新。CLIP始终冻结；原Adapter/投影/U-Net和新旁路头在两组中均联合接续，唯一比较是旁路读取全局均值还是带坐标的空间矩。

每组8192更新，84168→92360，同样数据90演示/10726窗口、批次/噪声流、原Adam/LR/EMA计数。原视觉64维接口不变，新头均262144参数。复用217778训练专用统计，不再拟合统计或诊断标签。

单作业顺序完成两组，申请12小时，短测纯训练投影合计7.72小时，另外有保存/恢复/闭环评价及排队时间。12小时是资源上限，不是保证的结束时间。每1024步及固定终点写完整raw/EMA/Adam/LR/RNG检查点，不自动重投或增加步数；若中断应先检查保存状态，不能从头重复提交。

每组只评价固定终点EMA的同一开发20场景及采样流，并各保存预先指定前2个视频。最终50继续封存。采用条件沿用预检前protocol：候选平均奖励同时比控制与原215999高至少0.05、配对场景bootstrap95%均值差下界>0、满奖励比例不低于两者。开发集已用过且只有一个源训练种子，采用也不是最终泛化/抓取确认；失败关闭本有界分支，不追加变体/预算/种子。

提交入口先运行原预检输出验证器一次，再把报告/统计SHA和本正式协议写入提交绑定。正式作业读取相同绑定并再次校验身份，不重跑预检更新。请保留旧预检目录，大replay.pt无需下载。

提交：bash /projects/Zeroshot/pusht_spatial_bypass_formal_v1/submit_once.sh

完成后：/projects/Zeroshot/envs/official-pusht-v1/bin/python /projects/Zeroshot/pusht_spatial_bypass_formal_v1/verify_formal.py /projects/Zeroshot/runs/pusht-bypass-formal-实际编号

需要18GiB可用空间保存两份完整原子检查点及临时写入。输出report.json、admission.json、统计与预检绑定、两组report/development/latest.json/完整latest.pt/updates.jsonl/前2视频。执行完成不等于性能改善。

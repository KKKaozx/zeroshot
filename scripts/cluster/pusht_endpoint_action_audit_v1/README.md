# Push-T冻结端点动作诊断

只读212158原生及215999项目固定84168更新EMA，CLIP冻结，0次优化、0次环境步。不读取失败旁路权重，不重复闭环评分、不访问final50。

固定16训练/16未参与策略训练的演示，各取作者窗口排名floor(n/3)、floor(2n/3)，合64窗；不看标签/奖励选样、不替换短演示。复用既有90/40角色，不称为新封存测试。服务器生成真实帧清单；不足3窗即停止。

原DDPM100生成，采样种子1001/1002逐一报告并平均，不选最好。每批4窗，两模型批次相同；钩子记录初始噪声、RNG后状态和辅助噪声SHA并核对匹配。报告动作坐标L2模拟像素误差（16步及真正执行1:9切片），按演示等权、排除padding；静止输入取第二观测当前agent位置，训练动作均值只用原90演示。辅助t=0/49/99epsilon误差含原目标全部16槽，不能替代自由生成/闭环成功。

沿用作者损失/采样与normalizer；加载后才移CUDA，模型冻结eval前后状态SHA一致。训练CLIP缓存复用合格report、长度/帧ID哈希及固定4个新选观测的在线对应，不重新全读11GiB缓存；未见观测只编码一次，不缓存可训练模块。CLIP worker结束后才生成动作。

资源A6000×1、4CPU、40GiB、30分钟；内部1700秒截止、Slurm不requeue。未实测实际服务器耗时，不保证完成；超时保留部分数组/报告并标未完成，不当作模型失败、不自动扩时/重跑。无自动训练或下一作业。

输出report.json、inputs.npz（原始观测/目标/帧ID/mask）、window_manifest.json、两个小预测NPZ、progress.json、clip_worker.log。大模型不复制。完成后用verify_output.py核对小输出，不需要重新读取大权重；权重身份在运行加载时已验证。

提交：bash /projects/Zeroshot/pusht_endpoint_action_audit_v1/submit_once.sh

完成验证：/projects/Zeroshot/envs/official-pusht-v1/bin/python /projects/Zeroshot/pusht_endpoint_action_audit_v1/verify_output.py /projects/Zeroshot/runs/pusht-action-audit-实际编号

本诊断帮助区分训练动作学习、未见演示差距及自由生成问题。专家动作可以有多解，离线误差不是控制成功率；不能凭一次短检查锁定唯一根因或自动启动长训练。

# 冻结CLIP的Adapter前空间旁路：仅预检

承接217660，不重跑已关闭210334。两组都完整恢复215999端点的raw/EMA/Adam/LR/EMA计数/RNG，保留原语义分支，增加相同262144参数的4096→64零初始化残差旁路。控制组使用原始CLIP patch全局均值重复4次，候选组使用原始patch的均值/x/y/xy空间矩。视觉64维+agent2维接口和作者U-Net/损失/调度不变。

旁路只使用原训练90演示11356帧的缓存，按训练通道均值方差固定标准化。无CLIP更新、无诊断状态标签输入、无留出统计。两组均接续训练旧Adapter/投影/后端和新头，不混入不同冻结范围。保留原优化历史，仅新参数使用空Adam历史；原EMA计数继续，新头EMA同步从零开始。

此次各8次独立预检更新，额外各重算1次用于完整磁盘恢复验证，不作为正式训练续点。检查零初始化原始raw/EMA条件和DDPM100动作精确一致、梯度有效、原参数Adam历史保留、模式绑定、完整新状态恢复及资源。只用两条训练窗口检查动作有限，无奖励/成功率，无开发场景或最终50访问。

后续拟固定两组各8192更新，只看固定终点EMA的dev20闭环重叠奖励。预检包不含正式入口、不自动提交后续训练，资源实测后另制入口。采用判据详见protocol.json，失败不自动加预算或换变体。

提交：bash /projects/Zeroshot/pusht_spatial_bypass_preflight_v1/submit_once.sh

验收：/projects/Zeroshot/envs/official-pusht-v1/bin/python /projects/Zeroshot/pusht_spatial_bypass_preflight_v1/verify_preflight.py /projects/Zeroshot/runs/pusht-bypass-preflight-实际编号

需要至少12GiB可用空间保存两份完整重放检查点，不删除源文件。预检报告、旁路统计和两组报告为小产物；大replay.pt保留在服务器。最长30分钟是申请上限，实际耗时及未来正式预算由本次测量决定。

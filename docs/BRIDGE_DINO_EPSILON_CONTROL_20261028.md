# 有效DINO＋当前位姿配方的单臂epsilon对照（2026-10-28）

229116的22,520,575字节报告包已本地验收：19指定文件、归档SHA0b11d4c6…f12d2b3，两臂完整8192日志/学习率/初始化/输入身份一致，全部训练与开发指标从保存预测复算通过。权重未下载，仅远端字节SHA与保存报告一致；新提交前仍会对集群旧DINO权重完整校验。实际DINO训练+原完整生成评价3019.90秒，约50分钟；不再按主模型5–6小时估计此小对照。

泛用目标比较186009/193195/197845继续复用，不重跑。新问题限定在后来已经有效的DINO＋9D当前位姿输入、CLS+patch均值、Adam阶梯LR配方。只新增一个epsilon候选，直接复用229116的x0，不再训练x0。固定seed42、1836训练/40开发窗口、batch8、8192次更新、同初始权重/数据/噪声/状态输入/读出/夹爪冻结和相应目标采样解释。8192是原对照预算，不能缩成短训练后与旧8192基线冒称纯单变量。

原完整core/loader/optimizer/evaluator文件不改，以4个明确源码插入点增加：固定常量身份选择、原正对照重放、首批原x0数值检查、checkpoint干预身份。完整有效源码/入口/帮助模块/协议摘要进入候选记录。只允许两项推导常量使用已存精确候选值，四项其他表必须已一致；整个原初始化摘要必须逐值匹配才准入，不替换编码器/学习参数、不放宽守卫。

训练前临时加载旧DINO训练权重，对固定首4训练窗/3采样seed/同batch2作12条100步旧预测重放，context与预测必须bit exact。随后完整恢复fresh初始参数值，保留RNG与模式；没有优化旧模型。实际第一噪声前向还需按原x0目标计算旧loss/各模块与全局梯度范数并逐值重放，清梯度后才对同图执行epsilon更新。不同即0更新停止。没有另跑工程训练预检。

最终按原完整1836训练窗及40开发窗、原3个采样seed作真实纯噪声100步生成，保存全部预测，复算轨迹/终点/演示等权，并与旧x0、不动比较。只报告效果及10演示重采样区间，不冒称训练seed稳健性、主OXE提升或LIBERO成功；不追加seed/预算、选checkpoint、换状态/读出或放行大评估。历史相同GPU型号参考不是同allocation新训练对；旧前缀数值重放限制明示。

本地CPU合成集成夹具验证了仅两buffer可恢复、额外参数/表变化拒绝、旧参数临时重放后fresh恢复、首批参考反向不优化且清梯度后保留图供epsilon反向、源码哈希/插入点/编译和LF检查。复用旧目标公式及调度验证，不重新跑数学诊断。本地没有加载真实模型/调用CUDA或训练；实际原模型重放仍待集群运行。ZIP11成员21,634字节，逐成员/哈希一致。见`reports/bridge_dino_epsilon_control_validation_20261028.json`和`bridge_dino_epsilon_control_package_20261028.json`。

预计50–65分钟不含排队，墙钟上限2小时。无新缓存/编码/依赖/数据，仅新约0.53GB学习权重和小报告。源旧权重/两个缓存/实际输入及失败profile必须仍在，缺失在登录节点停止，不重建或下载替代。主混训监督对照230933保持收束，本任务不变更主权重。

上传（本地PowerShell）：

```powershell
scp "D:\ntu_related\dissertation\Zero_shot\cluster_packages\bridge_dino_epsilon_control_v1.zip" zixiao005@10.97.216.128:/projects/Zeroshot/
```

提交（集群）：

```bash
(
set -euo pipefail
cd /projects/Zeroshot
unzip -n bridge_dino_epsilon_control_v1.zip
cd bridge_dino_epsilon_control_v1
bash submit_once.sh
)
```

日志`/projects/Zeroshot/logs/bridge-dino-epsilon-JOB_ID.out`，结果`/projects/Zeroshot/runs/bridge-dino-epsilon-JOB_ID/comparison.json`。目前尚未提交。

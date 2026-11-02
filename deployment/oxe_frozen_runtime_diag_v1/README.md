# 正式入口身份守卫诊断

CPU226567已复现绑定摘要，但没有复现226378失败。本包不替换原正式代码，通过Python trace沿原run_stage.py seed42/stage1初始化，在原身份守卫求值前抓取其真实局部变量并无条件停止。即使身份匹配也不会启动训练。另阻止意外模型forward或optimizer.step。

需保留原正式包及SHA256SUMS、CPU226567输出preflight.json/report.json。原正式输出若非空则拒绝；不删除、不写正式输出、不改提交marker、不绕过守卫。申请A40/4CPU/40GB/10分钟；不是正式重提。仅原CPU诊断无法覆盖的实际CUDA正式入口，不重复拟合或泛化实验。

解压后bash submit_once.sh。完成读取runs/oxe-id-runtime-JOBID/report.json；actual_profile.json记录逐项冻结张量摘要，不保存权重。报告若身份全部匹配会明确标记旧失败未复现，不自动训练/重绑/降低检查。

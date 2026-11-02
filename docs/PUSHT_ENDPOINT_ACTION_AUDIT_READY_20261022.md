# 冻结端点动作短检查：提交包就绪

2026-10-22。本地43项新入口检查通过，未加载真实端点、未训练、未新增环境步或集群提交。原生212158和项目215999权重、旧训练包均不修改。

包：`cluster_packages/pusht_endpoint_action_audit_v1.zip`，18文件，33859字节。

ZIP SHA256：`7c1740c8c9834c10d2a57f09b1b902773850158bc61ece86b4b40a407d0e66da`。

protocol SHA256：`201eb0c22cdf266d83c845dc414f771f4df2ba96017fd2d0a8bc8c588beabffd`。

只读两个原84168更新固定端点EMA，在32训练/32未见演示窗上做DDPM100动作生成，固定两个采样seed；辅助t=0/49/99epsilon误差。生成输入只含两帧观测与当前agent位置，没有未来目标。既有训练CLIP缓存复用，未见观测只编码一次，CLIP/策略参数均冻结。执行切片明确为1:9，padding不计入主动作误差。详见[排查表与既定方案](PUSHT_FRONTEND_EVIDENCE_AND_SHORT_CHECK_20261022.md)。

资源A6000×1、4CPU、40GiB，Slurm上限30分钟、不requeue；Python内部1700秒截止。实际时间与显存尚未在服务器测量；超时应标未完成，不据此判模型失败，不自动扩时或重复。一次提交保护目录`.submission`保留，禁止未知状态下重复运行。

本地检查包括独立选窗式与作者采样AST对18种长度/掩码的逐行比对；两角色各16演示/32窗及不跨演示；动作L2、padding、1:9切片和seed平均；真实冻结批次辅助函数用轻量假策略验证钩子噪声/RNG、只读参数、错误模式和清理；完整小输出验证器正反例；Python3.9、两Bash入口和18文件ZIP/SHA。假策略不等于真实大模型，不能宣称运行兼容和预测质量已由本地测试证明。

PowerShell上传：

```powershell
scp "D:\ntu_related\dissertation\Zero_shot\cluster_packages\pusht_endpoint_action_audit_v1.zip" zixiao005@10.97.216.128:/projects/Zeroshot/
```

服务器提交：

```bash
cd /projects/Zeroshot
unzip -n pusht_endpoint_action_audit_v1.zip
bash /projects/Zeroshot/pusht_endpoint_action_audit_v1/submit_once.sh
```

日志`/projects/Zeroshot/logs/pusht-action-audit-编号.out`，输出`/projects/Zeroshot/runs/pusht-action-audit-编号`。完成验证：

```bash
/projects/Zeroshot/envs/official-pusht-v1/bin/python \
  /projects/Zeroshot/pusht_endpoint_action_audit_v1/verify_output.py \
  /projects/Zeroshot/runs/pusht-action-audit-实际编号
```

输出小文件含inputs、window manifest、两组预测及报告，大模型不复制。没有新闭环奖励/抓取成功率或策略提升结论，不自动提交任何后续训练，final50仍封存。

构建脚本：`scripts/build_pusht_endpoint_action_audit_20261022.py`。
检查脚本：`scripts/check_pusht_endpoint_action_audit_20261022.py`。
检查报告：`reports/pusht_endpoint_action_audit_checks_20261022.json`。

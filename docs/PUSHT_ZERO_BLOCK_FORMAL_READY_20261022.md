# Push-T零块正式控制提交说明（2026-10-22）

219260预检和原服务器验证器均通过，证据来自用户终端；未下载该次大检查点。正式包本地41项新增入口检查通过，未提交正式训练。

## 本次只回答一个问题

保持冻结CLIP、原池化/LayerNorm/1024→64投影、agent_pos和作者扩散后端，移除8个跨注意力/FFN块，从头训练零块控制。同旧8块215999的84168次更新、数据、采样和噪声计划、优化与EMA条件匹配，直接复用旧8块终点成绩，不重训参照。零块不使用语言，仅用于Push-T固定指令的机制消融，不替代论文多指令模型。

固定终点才评估同20个开发场景，不用中间分数选模型，不接触final50。判据预先写入协议，结果可能支持移除、支持保留或证据不足；没有自动追加预算/变体。原8块在A6000训练，新控制申请A40，且只有一个训练seed、开发场景已复用；不能把结果当作单个注意力算子的最终因果证明。

## 资源与恢复

预检每步0.547秒，估算纯训练12.79小时；加载、保存/哈希、闭环和排队另计。申请A40一张、4CPU、40GiB、20小时硬上限，不保证实际结束时间。每2000步及终点保存完整状态，支持检查后显式恢复，不自动重投或从头重跑。

本地41项检查覆盖实际派生训练循环的计数、恢复身份、结果配对、验证器正反例、Python3.9/Bash语法与包散列；15个旧数据/训练/评估/在线模块字节不变。未用这些合成检查声称实际策略效果。之前47项前端检查不重复运行。

## 上传与提交

本地PowerShell：

```powershell
scp "D:\ntu_related\dissertation\Zero_shot\cluster_packages\pusht_zero_block_formal_v1.zip" zixiao005@10.97.216.128:/projects/Zeroshot/
```

集群终端：

```bash
cd /projects/Zeroshot
unzip -n pusht_zero_block_formal_v1.zip
bash /projects/Zeroshot/pusht_zero_block_formal_v1/submit_once.sh
```

保留服务器原`pusht_zero_block_preflight_v1`和`runs/pusht-zero-preflight-219260`目录。提交脚本调用原预检验证器并绑定报告/初始化，不重新执行预检；一次提交保护会保留作业编号。若提交失败，检查报错，不自行删除保护目录反复提交。

包：`cluster_packages/pusht_zero_block_formal_v1.zip`，30文件，67105字节。

SHA256：`7f2a1373d7eab9491bb7c4278516cd5697da1af858bd6c5a3abd14caa1592373`。

日志：`/projects/Zeroshot/logs/pusht-zero-formal-编号.out`。

输出：`/projects/Zeroshot/runs/pusht-zero-formal-编号/`。

结束后在服务器运行（用真实编号替换占位符）：

```bash
/projects/Zeroshot/envs/official-pusht-v1/bin/python /projects/Zeroshot/pusht_zero_block_formal_v1/verify_formal.py /projects/Zeroshot/runs/pusht-zero-formal-实际编号
```

完整验证需要服务器的`latest.pt`。小报告、更新日志、比较结果与两段固定视频可随后下载，大检查点留服务器。

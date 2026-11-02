# 区域读出正式配对入口：210291验证通过后

2026-10-18。用户运行集群完整验证器输出`SPATIAL PREFLIGHT VERIFIED`：峰值保留显存3.77734375GiB、暖更新0.160323583秒、配对正式估计1956.053秒（约32.6分钟，不含排队）。完整原report尚未本地下载；正式入口会读取集群真实报告并重新核验，绝不以本页数字替代。

## 固定实验

两组均从207558原检查点开始，原405演示/1836完整窗口、同4096×8采样计划、相同新Adam与1e-5学习率，各接续4096次。原Adapter/扩散/当前状态/历史端口在两组均学习；候选额外学习1026个区域评分参数，控制固定这些参数为零。CLIP视觉/文本、夹爪权重保持冻结，不修改原检查点或旧源码，不使用关闭分支的K1请求端口。

训练每次`calculate`的AST与成功预检完全一致。正式每组前8次损失要与210291对应组相符，事先固定容差1e-6；不是重新运行预检、8次之外额外热身或放宽旧回退门槛。实际CLIP/token/训练记录/检查点身份仍严格核验。没有完整1836训练池或旧40/13演示重评分。

两组最终权重全部保存及严格重载后，才读取并编码原固定新10开发演示的50个输入。冻结编码SHA必须与209481对应记录一致；目标、当前位姿特征和窗口身份也必须一致。原207558预测直接复用209481保存结果，不重新评价整套原模型。每组用原三个种子执行原生DDPM100，只评价原完整16目标。近期指标为这条16目标输出中的第1目标，不是另训K1策略。

采用规则完全沿用预检前protocol：近期第一目标位置须胜控制/原207558/不动，旋转不退步；至少同6/10演示近期位置同时胜两个模型参照；完整16路径/终点位置和旋转不能劣于控制/原模型，窗口计权近期和全段也不能反向退步。若失败，保留207558并关闭该读出分支，不追加种子、预算或变体。最终25条不读、不自动验收。执行完成、验证器通过和泛化改善是不同结论。

这是同任务已使用开发演示的离线位姿对照，不是未见任务零样本、物体检测或闭环抓取。夹爪权重未学习，也不能将其输出变化当成新夹爪能力。

## 发布与本地验证

- 追加包：`cluster_packages/bridge_spatial_formal_v1.zip`，16276字节、10文件。
- SHA256：`f5487bdbd505068f4faff0eab4709a7f09b945f6e56601dddb80968df671e66f`。
- 无权重、数据或环境；原`bridge_spatial_readout_v1`不覆盖。
- 训练loop的实际源码AST核对、Python语法/实际CLI导入、Bash语法、所有依赖指纹、ZIP字节/路径检查通过。
- 用32维八层Adapter和小玩具降噪器执行实际正式loop各8更新，检查首损失、冻结标记、参数格式和最终严格重载；这是合成入口验证，不是再次研究拟合或207558训练。
- 合成指标案例验证：可接受的候选通过；原样候选被拒绝；全段改善但第一目标变差被拒绝；第一目标旋转变差被拒绝；零四元数被拒绝。
- 尚未执行真实正式训练或提交作业。

## 上传和提交

本地PowerShell只上传新追加包：

```powershell
scp "D:/ntu_related/dissertation/Zero_shot/cluster_packages/bridge_spatial_formal_v1.zip" zixiao005@10.97.216.128:/projects/Zeroshot/
```

集群Bash：

```bash
set -euo pipefail
cd /projects/Zeroshot
printf '%s  %s\n' 'f5487bdbd505068f4faff0eab4709a7f09b945f6e56601dddb80968df671e66f' 'bridge_spatial_formal_v1.zip' | sha256sum -c -
if [[ ! -d bridge_spatial_formal_v1 ]]; then
  /projects/Zeroshot/envs/multitask-preflight-v1/bin/python -m zipfile -e bridge_spatial_formal_v1.zip /projects/Zeroshot
fi
bash -n /projects/Zeroshot/bridge_spatial_formal_v1/formal.sh
bash -n /projects/Zeroshot/bridge_spatial_formal_v1/submit_formal_once.sh
bash /projects/Zeroshot/bridge_spatial_formal_v1/submit_formal_once.sh
SPATIAL_FORMAL=$(cat /projects/Zeroshot/bridge_spatial_formal_v1/submitted_formal.job)
echo "区域读出正式对照：$SPATIAL_FORMAL"
squeue --start -j "$SPATIAL_FORMAL"
```

提交入口有锁和已有作业号保护，并先运行210291已有报告验证器；无历史Slurm依赖号参数，避免旧作业退出活动记录后的dependency问题。此验证器只读现有报告，不重新优化。实际任务开始时还会核对预检源码、协议及全部依赖；缺失/变化直接停止，不临时补下载或重编码训练数据。

查看结果：

```bash
sacct -j "$SPATIAL_FORMAL" --format=JobID,State,ExitCode,Elapsed,NodeList
if [[ -f "/projects/Zeroshot/logs/bridge-spatial-formal-$SPATIAL_FORMAL.out" ]]; then
  tail -n 35 "/projects/Zeroshot/logs/bridge-spatial-formal-$SPATIAL_FORMAL.out"
fi
```

完成后在集群核验（包括最终权重SHA与保存数组全部指标）：

```bash
/projects/Zeroshot/envs/multitask-preflight-v1/bin/python /projects/Zeroshot/bridge_spatial_formal_v1/verify_formal.py "/projects/Zeroshot/runs/bridge-spatial-formal-$SPATIAL_FORMAL"
```

每组只保存一个约529MB的最终非CLIP权重，无中途/Adam权重。其他小文件包含更新日志、50窗输入身份、3种子预测及完整报告；后续若下载小结果可排除`.pt`，但集群完整验证需要最终权重保留。

证据：[正式合同](../cluster_packages/bridge_spatial_formal_v1/formal_contract.json)、[执行入口](../cluster_packages/bridge_spatial_formal_v1/run_formal.py)、[采用规则实现](../cluster_packages/bridge_spatial_formal_v1/bridge_spatial_pair_metrics_v1.py)、[本地合成入口检查](../reports/bridge_spatial_formal_cpu_20261018/report.json)、[发布身份](../reports/bridge_spatial_formal_release_20261018.json)。

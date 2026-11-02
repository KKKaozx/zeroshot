# Bridge内部选模对照：操作与已完成检查

状态：205997完整小工件已下载并独立复核，32文件/22快照/6144更新通过。实际选中768步，最终6144步训练96/96拟合（1.660mm/0.531°）；原40外部窗最终8.135cm/15.422°，选中8.236cm/15.645°，固定主要判据失败。选模实现与身份/评分核对未发现错误，唯一根因未定。按计划结束，不扩展预算或另挑规则；见[独立结果与边界](BRIDGE_INNER_SELECTION_RESULT_205997_20261015.md)。下方上传、提交、查看命令均为历史追溯，不再执行。

正式作业查看：

```bash
sacct -j 205997 --format=JobID,State,ExitCode,Elapsed,NodeList
if [ -f /projects/Zeroshot/logs/bridge-selection-fit-205997.out ]; then
  tail -n 12 /projects/Zeroshot/logs/bridge-selection-fit-205997.out
else
  squeue --start -j 205997
fi
```

当前查看：

```bash
sacct -j 205825 --format=JobID,State,ExitCode,Elapsed,NodeList
if [ -f /projects/Zeroshot/logs/bridge-selection-preflight-205825.out ]; then
  tail -n 15 /projects/Zeroshot/logs/bridge-selection-preflight-205825.out
else
  squeue --start -j 205825
fi
```

## 本次实际完成什么

固定48条演示/96窗训练，16条演示/32窗内部选模，原10条演示/40窗最后评分。划分按整条演示及事前身份哈希，内部32窗不参与梯度。所有拟合的当前XYZ统计仅来自96个实际优化窗口，原固定0.1m动作单位保持不变。

一次6144更新，batch8，每窗512曝光；前3072学习率1e-4，后3072为1e-5。每384更新在内部32窗生成原生DDPM100动作，固定3采样种子；以演示等权的位置/静止、旋转/静止比值中较大者最小选模，平局选较早时刻。锁定之后才评价原40窗上的选中和最终权重。比较只在同一96窗训练过程内成立，旧128窗结果只是背景。

保持原CLIP、8层Adapter、U-Net、当前状态/重复当前历史端口和初始化43，不使用残差基础预测。训练误差仍报告，但不为达到毫米级追加训练。最多保留两份非CLIP权重，共不足1200MiB；原CLIP token缓存复用，上传包不含缓存、模型、图像或数据，不安装或下载新依赖。

本地检查：

- 错误跨集合取窗会被拒绝；任意改动全部留出观察都不能改变训练统计；完整6144更新抽样不含任何内部/外部窗口，96窗各512次。
- 选模规则经过演示等权、位置/旋转共同约束及平局保留较早权重的构造检查。
- 真正GPU预检：8次无监测更新，与8次带内部选模/保存操作的更新比较。参数、Adam、每步损失精确一致；保存的选中权重重载后对32窗的原生采样逐元素差0。
- 共16次操作更新，约170.83秒，RTX4060 Laptop峰值保留显存3.924GiB；原40窗无评分。临时本地权重验证后已移除。未重编码，复用约172MiB完整冻结token。
- 保存数组另用float64重新计算分数，最大分数差约2.71e-5，选中时间点一致；两份Bash脚本的只读语法检查通过。

第一版本地入口在0更新时停止：把Torch2.11本地初始化与Torch2.5集群指纹直接比较。现分别固定此前已保存过的两平台原始初始化指纹；本地与集群各自必须精确匹配，正式运行还必须与自己的集群预检版本/初始化相同。没有放宽成任意初始化，也未更改模型或训练规则。失败报告保留，不作为通过依据。

证据：[冻结协议](../cluster_packages/bridge_inner_selection_v1/protocol.json)、[GPU预检报告](../reports/bridge_inner_selection_preflight_v2_20261015/report.json)、[独立数组复核](../reports/bridge_inner_selection_preflight_v2_20261015/independent_audit.json)、[隔离与选模规则检查](../reports/bridge_inner_selection_contract_20261015/report.json)、[最初事前计划](BRIDGE_INNER_SELECTION_PLAN_20261015.md)。

## 第一步：本地PowerShell上传小包

ZIP共15,796字节，SHA256为`1229e9df1d79a94b5eab41bc2fa02955462538390d33665e57f179e4ebdb24d6`。

```powershell
scp "D:/ntu_related/dissertation/Zero_shot/cluster_packages/bridge_inner_selection_v1.zip" "D:/ntu_related/dissertation/Zero_shot/cluster_packages/bridge_inner_selection_v1.sha256" zixiao005@10.97.216.128:/projects/Zeroshot/
```

## 第二步：集群Bash只提交一次预检

```bash
cd /projects/Zeroshot
mkdir -p logs
sha256sum -c bridge_inner_selection_v1.sha256 && unzip -n bridge_inner_selection_v1.zip && SELECTION_CHECK=$(sbatch --parsable bridge_inner_selection_v1/preflight.sh) && echo "内部选模预检：$SELECTION_CHECK"
```

默认A6000；未读取当前集群显卡空闲情况，不能保证立即开始。只提交一次，保存编号。预检继续使用集群自己的缓存与软件环境；首次缓存不存在时允许只编码冻结的CLIP观察，不下载权重。

查看状态（新终端需把变量设为实际编号）：

```bash
squeue --start -j "$SELECTION_CHECK"
sacct -j "$SELECTION_CHECK" --format=JobID,State,ExitCode,Elapsed,NodeList
if [ -f "/projects/Zeroshot/logs/bridge-selection-preflight-$SELECTION_CHECK.out" ]; then
  tail -n 12 "/projects/Zeroshot/logs/bridge-selection-preflight-$SELECTION_CHECK.out"
fi
```

排队时尚无日志属正常；结束后可能不再出现在squeue，以sacct和report.json判断。短预检的动作误差不作泛化结论。

## 之后：预检结束且报告校验通过再提交正式

历史提交步骤：用户已执行报告校验及提交，正式编号205997，下面命令不再执行。原流程检查报告与当前源码身份，记录已提交编号避免重复。

```bash
/projects/Zeroshot/envs/multitask-preflight-v1/bin/python - "$SELECTION_CHECK" <<'PY'
import hashlib, json, pathlib, subprocess, sys
root = pathlib.Path('/projects/Zeroshot')
job = sys.argv[1]
assert job.isdigit()
pack = root/'bridge_inner_selection_v1'
report = json.loads((root/f'runs/bridge-selection-preflight-{job}/report.json').read_text())
assert report['stage'] == 'preflight'
assert all(report[k] for k in ['execution_checks_passed','input_contract_passed','observer_replay_exact','selected_checkpoint_replay_exact','clip_unchanged','other_frozen_unchanged'])
assert report['inner_and_outer_gradient_draws'] == 0
assert not report['outer_scored_during_selection']
assert report['actual_operational_updates_including_control'] == 16
for name, expected in report['entry_sha256'].items():
    assert hashlib.sha256((pack/name).read_bytes()).hexdigest() == expected
record = root/f'runs/bridge-selection-submission-{job}.txt'
with record.open('x') as stream:
    result = subprocess.run(['sbatch','--parsable',f'--export=ALL,SELECTION_PREFLIGHT_JOB={job}',str(pack/'fit.sh')], text=True, capture_output=True)
    if result.returncode:
        stream.write('SUBMISSION_FAILED\n'+result.stderr)
        raise SystemExit(result.stderr)
    new_job = result.stdout.strip()
    assert new_job.isdigit(), new_job
    stream.write(new_job+'\n')
    print('报告校验通过，正式编号：',new_job)
PY
```

正式入口在运行前仍会逐项检查预检代码、原始输入、统计与初始化指纹。因预检报告已确认成功，提交不依赖Slurm是否仍保留旧作业记录；若未结束、失败或报告缺失，该命令不能提交。若提交记录已存在，先查已有编号，不重跑提交。

## 解释与停止边界

主要问题是同一训练过程的选中模型是否同时改善原40窗的位置和旋转。即使改善，仍需报告相对静止及逐演示结果，不能认定整个问题已修复；若无改善，就结束这一固定选模规则，不扩展轮数、种子或新正则方案。原40开发窗已反复诊断，不能称未触及测试；夹爪冻结，无实际抓取或新任务零样本成功结论。

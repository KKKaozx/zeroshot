# 下一项泛化对照：视觉条件的变化是否带来负作用

状态：206274已完成27文件/13快照/8192更新的独立核验。原40窗位置6.966cm、8/10优于静止、6/10优于原图像，但旋转17.396°退步，主要集中关抽屉两演示；完整候选按原判据结束。训练8.437mm/3.160°、51/128窗门槛通过，不作为追加拟合理由。未做新训练/编码/推理/仿真，未下载PT或token缓存；见[独立结果与解释边界](BRIDGE_VISUAL_VARIATION_RESULT_206274_20261016.md)。下方上传、提交和下载步骤均为历史，不重复执行。

正式作业查看：

```bash
sacct -j 206274 --format=JobID,State,ExitCode,Elapsed,NodeList
if [ -f /projects/Zeroshot/logs/bridge-visual-fit-206274.out ]; then
  tail -n 12 /projects/Zeroshot/logs/bridge-visual-fit-206274.out
else
  squeue --start -j 206274
fi
```

当前查看：

```bash
sacct -j 206269 --format=JobID,State,ExitCode,Elapsed,NodeList
if [ -f /projects/Zeroshot/logs/bridge-visual-preflight-206269.out ]; then
  tail -n 15 /projects/Zeroshot/logs/bridge-visual-preflight-206269.out
else
  squeue --start -j 206269
fi
```

## 现有证据与真正缺口

动作范围、阶段取窗、视觉/状态近邻已在原128/40数据上检查过，本次直接引用，不重复扫描或编码。内部选模205997也已完成，选中768步没有改善原40窗的两项主要误差；该路线保持结束。

已有辅助状态岭回归用任务和当前末端状态，原40窗位置6.280cm，优于静止8.212cm；CLIP加状态6.239cm，原完整扩散202278真实状态组8.349cm/15.256°。但不同读出的参数量、正则和目标不同，这些结果无法把原因唯一归到视觉或扩散网络。之前固定旧权重换图像的干预也不等于训练时移除图像变化的对照。

尚未找到已完成的同结构、同128/40划分、同初始化与预算的“训练及评价均只有任务/当前状态变化”的扩散组。因此新增一组固定输入消融，检验视觉条件变化的净泛化作用。不是重做近邻、拟合、早停或状态基础＋残差候选。

## 只改变一个输入因素

完整图像控制直接复用202278真实状态组，不重训。新组保留同一冻结CLIP、8层Adapter、多尺度x0/DDPM100、重复当前历史分支、真实当前状态端口和可训练参数集合。

原本每个观察有一份257×1024视觉token。新组只从128个原训练窗口的token逐位置取均值，得到一份固定token；所有训练与评价观察都接收这一份。原语言token/掩码及当前状态仍按观察保留，目标动作、原0.1m动作单位和128训练窗状态统计不变；不增加基础预测、未来输入或目标依赖条件。

这是一项定位用的视觉输入消融，最终项目仍需要视觉定位与CLIP＋Adapter＋扩散。即使改善，也只能支持这个配方中视觉输入变化有负作用，不能单凭该结果断言CLIP错误、Adapter某层有bug、模型已学到伪语义关联，或无视觉系统已满足项目目标。若没有改善，也不能认证视觉模块完全正常。

## 固定预算与判据

- 原64条训练演示/128窗、10条独立开发演示/40窗，完整身份不变；不能拿205997实际96训练窗作为单因素因果控制。
- 初始化43，batch8，一次8192更新；前4096学习率1e-4、后4096为1e-5，每窗512次曝光；相同Adam、梯度裁剪及batch/noise种子。
- 最终权重评价，原生DDPM100及3固定采样种子。完整40窗、逐演示/任务与静止均报告；保留状态辅助读出6.280cm作为更强位置参照，不能只与旧扩散比较就宣布有效。
- 候选筛选：路径位置至少优于静止10%（≤7.3911cm），路径旋转不高于静止15.0985°＋0.02°数值容差，两者均低于复用的原控制，至少6/10演示位置优于原控制。训练误差完整报告，不设为继续训练到毫米级的条件。
- 无论结果如何，到固定预算结束；不改变固定视觉强度、均值方式、正则或选模时刻，不追加初始化/轮数。本轮只有一组新训练，旧图像模型直接复用。

原40窗已反复用于开发，不是未触及测试；3采样种子不是3次训练。夹爪冻结，只有离线位姿，不宣称抓取或未见任务零样本成功。

## 本地已完成的必要检查

原输入/标签、初始化和状态特征与此前本地原始配方精确一致。实际均值函数仅接受128个训练token序列，传入168序列会被拒绝；168个查询的视觉token完全相同，修改全部40留出token不影响训练均值。没有用目标动作拟合均值。

GPU预检为4共同＋4常学习率＋4恢复同学习率＋4降学习率，共16次入口操作更新。恢复的参数、损失、预测精确一致，CLIP与夹爪冻结，四个既定训练窗只做入口评分，无开发评分。约103.31秒、峰值保留显存3.785GiB；未保存本地新权重、未重新编码或下载模型。短预检误差/门槛不作泛化结论。

独立CPU复算5个保存快照，位置最大指标差1.37e-6cm，旋转5.54e-5°，已核对实际生成源码及各项指纹。Bash脚本只读语法检查通过。原训练器/模型文件不修改；新增入口只按出现次数断言包装原训练器，实际执行源码随run保存。

证据：[事前计划](../reports/bridge_visual_variation_plan_20261016/plan.json)、[实际均值函数检查](../reports/bridge_visual_variation_plan_20261016/runtime_helper_cpu_check.json)、[冻结协议](../cluster_packages/bridge_visual_variation_v1/protocol.json)、[本地预检](../reports/bridge_visual_variation_local_preflight_20261016/report.json)、[独立验收](../reports/bridge_visual_variation_local_preflight_20261016/independent_audit.json)。原计划JSON记录当时未实现状态，后续进展以本页为准。

## 上传并提交一次集群预检

包10,022字节，SHA256为`145d48cfad3f192db85ad0053817922c6947413e7369d7d1146cd5c23a18c1bd`；不包含图像、数据、token缓存或模型权重。正式只保留一份非CLIP权重（包含约1MiB固定视觉token）；复用原缓存，无安装下载。脚本按项目200GiB规划预算保留15GiB余量，这不是实时配额查询。

本地PowerShell：

```powershell
scp "D:/ntu_related/dissertation/Zero_shot/cluster_packages/bridge_visual_variation_v1.zip" "D:/ntu_related/dissertation/Zero_shot/cluster_packages/bridge_visual_variation_v1.sha256" zixiao005@10.97.216.128:/projects/Zeroshot/
```

集群Bash：

```bash
cd /projects/Zeroshot
mkdir -p logs
sha256sum -c bridge_visual_variation_v1.sha256 && unzip -n bridge_visual_variation_v1.zip && VISUAL_CHECK=$(sbatch --parsable bridge_visual_variation_v1/preflight.sh) && echo "视觉条件对照预检：$VISUAL_CHECK"
```

默认A6000，当前空闲情况未知；只提交一次并记录编号。预检完成后先校验其报告和当前源码指纹，再用fit.sh及VISUAL_PREFLIGHT_JOB启动一组正式8192更新。现在没有正式编号，不运行重复旧控制，也不把预检当成泛化提升。

## 206269完成后的条件提交（历史步骤，已提交206274）

集群终端执行以下整段，报告字段/固定协议/入口与缓存助手身份通过后才提交，并写一次提交记录。不存在报告、预检失败或源码不一致时停止；已有提交记录时只读编号，不重复提交。

```bash
/projects/Zeroshot/envs/multitask-preflight-v1/bin/python - <<'PY'
import hashlib, json, pathlib, subprocess
root = pathlib.Path('/projects/Zeroshot')
pack = root/'bridge_visual_variation_v1'
r = json.loads((root/'runs/bridge-visual-preflight-206269/report.json').read_text())
assert r['stage'] == 'preflight' and r['state_arm'] == 'real'
assert all(r[k] for k in ['execution_checks_passed','input_contract_passed','branch_restore_passed','clip_unchanged','other_frozen_unchanged','all_queries_same_visual_tokens'])
assert r['constant_visual_fit_rows'] == list(range(128))
assert not any(r[k] for k in ['language_inputs_changed','observed_state_inputs_changed','action_targets_changed','original_control_retrained'])
for key, file in [('visual_protocol_sha256',pack/'protocol.json'),('visual_entry_sha256',pack/'run_visual_control.py'),('encoder_cache_helper_sha256',root/'bridge_inner_selection_v1/selection_contract.py')]:
    assert hashlib.sha256(file.read_bytes()).hexdigest() == r[key]
assert r['visual_protocol_sha256'] == '71e8d16154288d557625bc817a5798d94f76855e33525ac3652dc4b37f46c120'
record = root/'runs/bridge-visual-submission-206269.txt'
if record.exists():
    print('已有提交记录，不重复提交：',record.read_text())
    raise SystemExit
with record.open('x') as f:
    result = subprocess.run(['sbatch','--parsable','--export=ALL,VISUAL_PREFLIGHT_JOB=206269',str(pack/'fit.sh')],text=True,capture_output=True)
    if result.returncode:
        f.write('SUBMISSION_FAILED\n'+result.stderr)
        raise SystemExit(result.stderr)
    job = result.stdout.strip()
    f.write(job+'\n')
    print('报告校验通过，视觉条件正式对照：',job)
PY
```

正式一次8192更新、最终权重评分，原控制复用。用户已执行条件提交并返回206274；入口仍会独立核对预检的原始输入、源码与固定视觉均值身份。尚无正式运行状态或改善结论，不重复提交。

## 历史证据引用

- [范围/阶段/近邻审计入口](../scripts/diagnose_bridge_growth128_generalization.py)及[已有结果](../reports/bridge_growth128_generalization_201663_20261013/analysis.json)
- [辅助状态/视觉/学习后条件读出](BRIDGE_GENERALIZATION_READOUT_20261015.md)
- [原真实状态128/40配对控制](BRIDGE_STATE_PAIR_RESULT_202278_20261013.md)
- [已结束的内部选模](BRIDGE_INNER_SELECTION_RESULT_205997_20261015.md)

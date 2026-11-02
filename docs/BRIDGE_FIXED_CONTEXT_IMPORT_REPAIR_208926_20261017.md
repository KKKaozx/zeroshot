# 208926正式入口导入冲突修复

用户已报告修复后正式作业208964提交，复用预检208897。尚未收到运行状态、绝对路径验证器检查输出或新泛化成绩；只跟踪208964，不重复提交。日志`/projects/Zeroshot/logs/bridge-fixed-fit-208964.out`，报告`/projects/Zeroshot/runs/bridge-fixed-fit-208964/report.json`。以下“尚未提交”为修复准备时的历史记录。

用户报告208926为FAILED 1:0，A6000，5分11秒。旧4窗context/预测重放仍为0差。异常在`FixedHooks.fit`首部导入验证器时发生，早于`advance('fixed_context',8193,16170)`，所以没有新增训练或泛化结果。尚未本地取得完整失败报告；修复入口会在集群另外检查失败报告、无更新日志和无PT，若发现已经训练则停止，避免盲目重复。

根因是新入口的模块名称冲突：继承入口把旧目录放到sys.path前端，导致`from validate_preflight import validate`加载了`bridge_demonstration_expansion_v1/validate_preflight.py`，其中没有`validate`。预检分支不调用这个fit内导入，因此先前预检没有触发。之前的本地AST/CPU参数屏蔽检查也没有覆盖这种正式入口导入顺序，这是本轮准备遗漏。

## 修复范围

新包`bridge_fixed_context_import_repair_v1`仅增加外层入口，不修改原包的任何源码、协议、预检验证器、检查门槛或旧输出。它按明确文件路径加载原fixed_hooks和原验证器，在原fit调用期间将该验证器绑定到对应模块名，结束/报错后恢复原模块缓存。原fit函数体不重新编译或修改，context/check/advance及全部训练、评价逻辑仍为原函数。

入口先用真实208897完整报告校验，再加载模型，减少此类错误导致的GPU等待。原8192模型/Adam、7978新增更新、数据、批次、Dropout、噪声、学习率、梯度裁剪与参数屏蔽均保持。复用已通过208897，不重做12步GPU预检，不重建8192前缀，不重做208586联合正式对照。

## 已做的回归检查

本地明确标注为合成数据的预检报告夹具，使用原未改验证器和文件哈希：

1. 在模块缓存放入错误的同名模块，执行实际原fit函数，复现完全相同的ImportError，尚未进入advance。
2. 应用修复后，执行同一个原fit函数，通过原完整验证器，准确到达`advance('fixed_context',8193,16170)`，测试故意在这里停止，0训练更新。
3. 复制文件被篡改时原验证器仍拒绝，门槛未绕过；调用结束或抛错后，模块缓存恢复。
4. 原全部文件和23,720字节旧ZIP哈希保持不变；启动训练参数逐项相同，仅入口路径、修复包完整性校验及作业名称变化。Python编译、Bash只读语法检查、ZIP逐文件核验通过。

未加载项目模型、执行GPU或重新训练。合成夹具不是实际集群预检报告；真实报告仍由启动器及修复入口使用原验证器检验。

## 上传并重新提交正式入口

修复包6,905字节，无权重/数据，SHA256为`bb297fd4c5b39971c57c1e48461e80486e2ee0c1a3d728f2dee889e096c26cf5`。

本地PowerShell：

```powershell
scp "D:/ntu_related/dissertation/Zero_shot/cluster_packages/bridge_fixed_context_import_repair_v1.zip" "D:/ntu_related/dissertation/Zero_shot/cluster_packages/bridge_fixed_context_import_repair_v1.sha256" zixiao005@10.97.216.128:/projects/Zeroshot/
```

集群：

```bash
cd /projects/Zeroshot
sha256sum -c bridge_fixed_context_import_repair_v1.sha256 &&
unzip -n bridge_fixed_context_import_repair_v1.zip &&
FIXED_TRAIN=$(sbatch --parsable --export=ALL,FIXED_PREFLIGHT_JOB=208897 bridge_fixed_context_import_repair_v1/fit_repaired.sh) &&
echo "固定上下文修复后正式对照：$FIXED_TRAIN"
```

日志和结果目录沿用`bridge-fixed-fit-JOBID`，不会覆盖208926。新输出额外保存修复入口、修复合同和`import_repair_entry.json`证据。当前修复后作业尚未提交，泛化结果尚未产生。

[修复准备记录](../reports/bridge_fixed_import_repair_preparation_20261017.json)、[导入回归结果](../reports/bridge_fixed_import_repair_validation_20261017/validation.json)、[原正式失败记录](../reports/bridge_fixed_context_formal_submission_208926.json)、[保持不变的科学对照](BRIDGE_FIXED_CONTEXT_GENERALIZATION_20261017.md)。

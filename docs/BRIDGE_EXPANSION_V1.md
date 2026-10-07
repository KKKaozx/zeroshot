# Bridge训练扩充 v1

最新状态：190188检查与190189训练/完整评价已COMPLETED/0:0。
latest三采样种子均值：训练10.299cm/16.748°，开发10.189cm/17.015°，均仍劣于各自静止位姿基线。
固定开发位置与夹爪平衡准确率改善，旋转及打开→关闭目标对下降；未通过位姿基线。
完整汇总见reports/bridge_expansion_190189_log_summary.json；新run逐任务JSON和history待取回。
以下“当前未执行新训练”等为准备阶段记录。

本轮只扩大Bridge五个已有指令的训练演示，不加入其他来源，不更换网络。
原pilot的45条训练演示全部保留；10条开发演示与13条保留测试演示逐字段不变。
新增演示按原始train分片、记录顺序选择，以origin路径和episode ID去重，
每任务训练演示最多100条，不根据数值动作或模型误差挑选。

扫描范围为原有0–31分片基础清单、已扫描32–63分片候选，以及新扫描64–255分片。
得到412条训练候选：开抽屉100、拨杆100、取西兰花55、关微波炉64、关抽屉93。
对应1848个16步完整窗口，开发集仍为40窗口。100条是上限，不是每任务都达到。
这仍是同任务演示留出，不证明新任务、独立场景或跨机器人泛化。

首次验收发现新增拨杆演示的末尾有效夹爪命令无法按既定阈值确定二值标签。
随后对全部422条训练/开发候选检查同一末尾条件，共7条新增训练演示不满足。
保留原候选清单及失败报告，另存eligible清单并逐条记录剔除原因；不补选替代。
完整验收通过405条训练演示、1836窗口，拨杆93条，其他任务数量不变。
这是现有标签合同的适用性问题，不代表这7条原始演示本身损坏。

训练/开发全部415条演示、13,517帧RGB完成验收，独立位姿还原与生产目标核对通过。
全部10条开发原始记录的序列化payload SHA256与原pilot审计完全一致。
训练目标中10个位移目标超出各坐标30cm的记录范围，审计只记录，不裁剪或排除。

已有原始文件直接从E盘读取，不复制完整Bridge。不导出保留测试目标或图像。
选中训练/开发需通过现有逐帧RGB、有限数值、轨迹边界、独立位姿还原、
夹爪命令时间对应与生产加载器全窗口核对，随后才能导出。

复现元数据选择时可重新扫描32–255分片，不依赖本地忽略的缓存文件：

```powershell
python code/prepare_bridge_expansion_plan.py --data E:/dataset/bridge_v2_0.0.1/0.0.1 --base-plan configs/bridge_multitask_pilot_plan.json --cache results/bridge_expansion_rescan.json --output results/bridge_expansion_reproduced_plan.json --start 32 --stop 256 --per-task 100
```

数值验收使用现有脚本，指定新报告路径：

```powershell
python code/audit_bridge_multitask_pilot.py --plan configs/bridge_expansion_eligible_plan_v1.json --data E:/dataset/bridge_v2_0.0.1/0.0.1 --output reports/bridge_expansion_source_audit_verified_v1.json
```

训练设置沿用冻结CLIP、8层512维Adapter、条件时间U-Net/DDPM100步、
独立夹爪头、batch 2、seed 42、AdamW 1e-4、cosine、20轮全训练窗口。
总更新为18360次，原pilot为2080次；数据量和优化曝光同时增加，不能称为
仅改变样本数量的因果对照。每轮评价原40开发窗口，结束评价best/latest的
全部训练/开发窗口及三个采样种子，采样种子不等于独立训练种子。

判断改善看实际采样动作的位置、旋转与各自静止基线，以及夹爪平衡准确率、
双向切换目标对和逐任务结果；降低加噪训练损失不代表成功。开发集只10条演示，
反复使用的比较属于开发，保留测试仍不使用。当前未执行新训练。

导出包已通过本地主入口prepare-only复读：405训练、10开发、1876窗口；
文件SHA256、ZIP CRC、两个Bash脚本语法及原有5项分区/指标测试通过。
ZIP约694MiB，解压约779MiB，本地两份合计约1.44GiB；未下载新数据或权重。
解压包位于training_cache/exports/bridge_expansion_training_v1，压缩包同目录上一级。

在本地PowerShell上传：

```powershell
scp "D:/ntu_related/dissertation/Zero_shot/training_cache/exports/bridge_expansion_training_v1.zip" "D:/ntu_related/dissertation/Zero_shot/training_cache/exports/bridge_expansion_training_v1.sha256" zixiao005@10.97.216.128:/projects/Zeroshot/
```

在集群终端校验、解压，先提交CPU环境/分区检查，训练依赖检查成功：

```bash
cd /projects/Zeroshot
sha256sum -c bridge_expansion_training_v1.sha256 && unzip -n bridge_expansion_training_v1.zip
```

只有上面成功后，运行：

```bash
BRIDGE_SETUP=$(sbatch --parsable /projects/Zeroshot/bridge_expansion_training_v1/setup_multitask_training.sh)
BRIDGE_EXPAND=$(sbatch --parsable --dependency=afterok:${BRIDGE_SETUP} /projects/Zeroshot/bridge_expansion_training_v1/multitask_training.sh)
echo "检查作业：$BRIDGE_SETUP；训练作业：$BRIDGE_EXPAND"
squeue --me
```

训练日志为/projects/Zeroshot/logs/multitask-training-作业编号.out，
输出为/projects/Zeroshot/runs/bridge_expansion_training_v1-作业编号。
复用multitask-preflight-v1环境和固定CLIP缓存，不覆盖旧实验输出。
环境检查显式安装Pillow，避免上次缺PIL导致的首批停止。

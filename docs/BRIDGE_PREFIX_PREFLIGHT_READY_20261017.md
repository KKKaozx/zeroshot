# 新输入已核验，单次集群预检包已准备

承接[固定配对设计](BRIDGE_PREFIX_PAIR_PREPARATION_20261017.md)。当前仍无新的泛化/抓取结果，207558权重和主参照不变。本轮完成数据资格与新运行入口检查，未向集群提交作业。

## 已完成

- 复用原405条训练演示，核验全部1558个新增后段起点：原始图像/指令、当前位姿/开度、真实未来长度、位姿目标及有效夹爪命令。没有重新解码旧1836个完整输入，没有安装依赖、加载策略、编码CLIP或训练。
- 新10条开发演示全部合格，对应50个完整16步窗口；只做数据资格，不计算模型分数。新增记录仅从E盘只读提取，未修改E盘数据。
- 25条原预留最终验收演示仍为元数据，未读取内容。导出文件中只有10条指定开发记录，其他索引为空占位；独立检查非空记录身份恰好等于这10条，未混入最终数据。
- 新的轻量记录解析器在全部10条新开发记录上，与TensorFlow官方Example解析结果逐项一致。训练405条的原始记录SHA与原已核验导出对应；新标签使用原冻结坐标/夹爪定义。
- 实际原始指令有88处大小写/空白等变体，本地CLIPTokenizer检查与同任务规范指令的token IDs都一致；没有静默重写原始指令。集群仍按实际token IDs/attention mask核对后才复用旧语言特征。
- 新集群数据入口已在本地CPU完整走通：1558新增训练/50开发窗口与冻结参考的有效目标、当前状态特征、当前开度及固定归一化像素网格差均为0。这是新运行入口集成检查，不是再次训练或评价旧模型。

完整RGB使用编码字节SHA、解码RGB SHA、形状和冻结预处理源码校验。归一化浮点数保存固定3×8×8像素网格作跨平台数值保护，容差1e-6；**没有声称比较过所有归一化像素**。标签和状态特征容差1e-6，当前开度及有效夹爪标签要求一致；未知未来仍保留NaN并由明确请求规则排除。

## 预检将执行什么

新包自包含冻结源码和新接口，没有通用同名validate模块依赖；使用现有环境、现有CLIP和原权重，无下载/安装。

1. 核对实际207558检查点SHA、原配置和全部可训练键，复用旧1876行CLIP缓存。仅1558个新增训练画面编码一次；语言仅在token IDs/mask完全匹配后复用。缓存放CPU，按batch传入GPU。
2. 用原训练位置0、32、64、96的已保存上下文/预测作兼容检查。维持原batch2和seed1101，跳过的旧batch仅消耗相同随机数，不重新生成1836条结果。原上下文/预测绝对容差均1e-4，不在失败后放宽。
3. 控制/候选各执行固定8次工程更新，核对同起点首个16步损失、冻结与有限梯度、参数确实变化。第4步的权重/Adam临时落盘后恢复，检查状态及下一次前向预测/损失一致；不增加额外optimizer.step做重放。临时约1.5GiB文件验完删除，不保存新最终权重。
4. 报告显存和配对正式训练的运行时间估计。预检资源门槛仍是7.5GiB和估计2小时；不打开发模型分数、不打开25最终记录、不要求8步拟合通过。

八次更新只供工程资格检查，不接续其权重开展正式训练。两组正式设计仍各4096次新增更新；本包没有正式训练入口，预检通过也不会自动提交正式作业。

## 本地检查与剩余限制

Python语法、CLI导入、Bash语法、完整CPU数据入口以及ZIP逐文件SHA已通过；ZIP无CLIP资产、权重或大型特征缓存。代码静态审查还明确将根部扩散表移到CUDA，与时间步索引同设备，同时将CLIP留在CPU/按需编码，不改变原表值。

真正的207558训练权重GPU重放、CUDA训练/恢复和资源门槛尚未执行，必须由这次集群预检确认。不能把本地随机完整模型的接口检查或数据校验当作上述GPU结果。

最终发布包：[bridge_prefix_pair_v1.zip](../cluster_packages/bridge_prefix_pair_v1.zip)，25,978,393字节，SHA256：`1196de26e783d6144b11c26fc2bcfe1f99567bd31d85848ad7e4eb20f52ae49c`。此前暂存包校验记录保留，以上发布清单为本次上传版本。

## 上传和唯一一次提交

上传ZIP到`/projects/Zeroshot/`。可使用现有文件传输方式，或在本地PowerShell运行：

```powershell
scp "D:\ntu_related\dissertation\Zero_shot\cluster_packages\bridge_prefix_pair_v1.zip" zixiao005@10.97.216.128:/projects/Zeroshot/
```

集群运行：

```bash
cd /projects/Zeroshot
unzip -qn bridge_prefix_pair_v1.zip
bash bridge_prefix_pair_v1/submit_preflight_once.sh
```

防重复提交入口加文件锁并记录作业号；再次调用只显示已有编号，不重复提交。暂不并发提交其他显卡备选。默认申请A6000一张、4CPU、40GiB内存、45分钟，仅提交一个作业。当前本地没有SSH操作或提交。

日志为`/projects/Zeroshot/logs/bridge-prefix-preflight-作业号.out`，报告为`/projects/Zeroshot/runs/bridge-prefix-preflight-作业号/report.json`。完成后可在集群用现有Python校验：

```bash
/projects/Zeroshot/envs/multitask-preflight-v1/bin/python /projects/Zeroshot/bridge_prefix_pair_v1/verify_prefix_preflight.py /projects/Zeroshot/runs/bridge-prefix-preflight-作业号
```

证据：[输入资格报告](../reports/bridge_prefix_input_audit_20261017/report.json)、[新入口CPU数据检查](../reports/bridge_prefix_packaged_input_smoke_20261017.json)、[独立数据/归档核对](../reports/bridge_prefix_preflight_independent_check_20261017.json)、[最终发布清单](../reports/bridge_prefix_preflight_release_20261017.json)。

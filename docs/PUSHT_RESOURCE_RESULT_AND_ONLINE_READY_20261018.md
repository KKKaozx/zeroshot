# 210763资源结果与在线接线预检

210763完整report已本地读取、校验关键条件并保存SHA，A6000/2:10/0:0。CLIP所有参数冻结且前后权重相同；两组初始后端SHA相同、仅训练90演示拟合的normalizer SHA相同；位置列实测[64,65]。项目Adapter/映射和共同后端均有有限非零梯度，EMA/DDPM100输出形状通过。这是接线/成本证据，没有任务得分或泛化结果。

| 项目 | 作者原生组 | 冻结CLIP+项目Adapter组 |
|---|---:|---:|
| 可训练参数 | 262,709,026 | 305,159,362 |
| 每次更新（有效batch64，含EMA） | 0.51075秒 | 1.03713秒 |
| 训练加采样峰值保留显存 | 5.36GiB | 7.65GiB |
| 20,000更新纯训练投影 | 2.84小时 | 5.76小时 |
| 84,168更新纯训练投影 | 11.94小时 | 24.25小时 |

长预算两组合计约36.19 GPU小时；不含12GB全训练特征缓存、检查点、闭环评价及其他I/O。估计只来自3次暖更新，不能当运行时保证。原0.9309不是这个新模型或新训练配方的成绩。

## 预算建议，尚无正式训练入口

先完成在线输入/执行接线，再锁定正式采样/噪声与恢复协议。候选分阶段预算：先原生参考20,000更新（约2.84小时纯训练），在预先锁定的开发环境检查其是否建立基本闭环能力。若达到预先约定门槛，项目组使用相同20,000更新；否则原生可从原状态接续到84,168的上限，达到门槛再给项目组匹配预算。不能把短阶段尚未成功说成项目模块失效，也不因为项目分数临时挑原生预算。最坏两组约36.19小时而非自动批准开跑。

建议的正式开发/最终边界及门槛草案独立记录在`configs/pusht_frontend_budget_draft_20261018.json`，当前status=draft，不是生产协议，不评分、不读取所列回合；训练90演示不增加、更换。正式前还须核验独立噪声计划与原生损失一致性、保存恢复与全数据缓存，不能把本次资源probe直接升格为训练器。

## 当前剩下的具体接线

原作者[图像环境](https://raw.githubusercontent.com/real-stanford/diffusion_policy/5ba07ac6661db573af695b419a7947ecb704690f/diffusion_policy/env/pusht/pusht_image_env.py)把渲染uint8图转换成float32并除255，而离线CLIP缓存接收原始float32像素。新wrapper在作者normalizer之前取观察，逆回原像素网格；只消除已知整数网格的浮点往返误差，不改离线数据。否则直接把[0,1]或[-1,1]当原像素送入CLIP会产生上下文偏差。这是待核验边界，不是已发现的Bridge根因。

新在线worker在既有项目环境中重算CLIP，作者环境通过继承的私有socket接收数值NPZ和JSON；没有监听端口、pickle消息或每步文件写盘。父进程有请求ID/大小/超时检查；退出时仅清理它启动的子进程。仍不安装、下载或合并环境。

## 一次短预检，0训练更新

- SHA绑定210763已成功报告及原inputs/tokens，不重跑5更新资源试验、不重算全训练集。
- 对照缓存与在线文本、视觉特征、短批次补到8、Adapter条件、相同采样seed下DDPM100完整动作。预先固定token/context atol1e-4、rtol1e-5，动作绝对差上限0.05像素；失败保存证据并停，不临时放宽。
- 原后端和normalizer重新构建指纹一致；使用新随机、**未训练**的Adapter/策略，不能当作210763五更新后的权重。
- 只用独立接线种子20261018，在真实PushTImageEnv最多两次决策/16环境步，更新两帧历史。丢弃奖励，不算覆盖率、成功率、开发或最终分数，不生成视频。目的是验证在线输入确实能经过模型送到环境，绝不是任务能力展示。
- 结束核对worker CLIP仍冻结、完整权重指纹未变，记录通信耗时。“在线”指不依赖训练图像缓存，不代表达到墙钟实时控制速率。

## 已完成本地检查与边界

所有256种8位像素的float32/255往返精确恢复；重复缩放/错误类型拒绝；分段内存socket表面的NPZ消息往返、对象/超大消息拒绝；wrapper成功及异常后清空缓存、原状态留给作者normalizer处理，通过。Python3.9语法、CLI/Bash和包SHA通过。本地没有运行真实跨Python子进程、CLIP或仿真；这些由本GPU预检核验，不冒称已通过。

## 上传与提交

包`cluster_packages/pusht_live_bridge_v1.zip`，资源为单A6000、CPU4、22GB主存、20分钟上限。只写`runs/pusht-online-JOBID/report.json`和worker日志，不启动正式训练。保留207558、所有旧包及数据，CLIP硬冻结，Bridge最终25未读。

Windows PowerShell：

```powershell
scp "D:\ntu_related\dissertation\Zero_shot\cluster_packages\pusht_live_bridge_v1.zip" zixiao005@10.97.216.128:/projects/Zeroshot/
```

集群：

```bash
cd /projects/Zeroshot
if [ ! -e pusht_live_bridge_v1 ]; then
  /projects/Zeroshot/envs/official-pusht-v1/bin/python -m zipfile -e pusht_live_bridge_v1.zip .
  PUSHT_ONLINE=$(sbatch --parsable /projects/Zeroshot/pusht_live_bridge_v1/preflight.sh)
  echo "Push-T在线接线预检：$PUSHT_ONLINE"
else
  echo '目录已存在，请先核对已有作业，避免重复覆盖/提交。'
fi
```

查看：

```bash
sacct -j "$PUSHT_ONLINE" --format=JobID,State,ExitCode,Elapsed,NodeList
tail -n 30 "/projects/Zeroshot/logs/pusht-online-$PUSHT_ONLINE.out"
```

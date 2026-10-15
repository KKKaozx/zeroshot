# Bridge EMA 单因素配对实验

只检验 EMA 参数平滑对当前 Bridge 配方的影响。复用 193761 的数据划分、初始化、随机训练顺序、20 epoch、18360 次更新、batch=2、AdamW、损失、动作尺度、冻结 CLIP 和多尺度 x0 U-Net。

同一次优化保存 final.pt 和 ema.pt；EMA 平滑全部可训练参数，不参与梯度计算，不包含冻结 CLIP。作者实际 checkpoint 的 EMA recipe 为 update_after_step=0、inv_gamma=1、power=0.75、min=0、max=0.9999；每次 optimizer.step 后更新。参考固定 commit：
https://github.com/real-stanford/diffusion_policy/blob/5ba07ac6661db573af695b419a7947ecb704690f/diffusion_policy/model/diffusion/ema_model.py

本实现只存 trainable shadow，保留原始固定 buffers；当前网络没有 BatchNorm，遇到 BatchNorm 会停止。普通模型与 193761 的最终张量摘要也会比较；跨 GPU 的非确定算子可能导致不同，届时仍以本次同一优化轨迹的 raw/EMA 配对为依据。

## 资源和已有依赖

无需 pip/conda 安装或下载。依赖现有 multitask-preflight-v1 环境、bridge_expansion_training_v1 数据与代码，以及 192651、191004、193711、193761 报告和 193761 checkpoint。启动时校验核心代码、数据包、划分与初始化；缺少文件或哈希变化则停止。

训练新增两个约 0.5 GB 的 checkpoint（合计约 1 GB），显存额外约 0.5 GB 的 EMA shadow；评估保存约 12 MB 的未压缩预测数组及小报告。训练时间上限 90 分钟，评估时间上限 3 小时，这是作业预算，不是运行时间承诺。以前 193761 训练约 37 分钟；EMA 有额外开销。

## 上传和训练

本地 PowerShell：
```powershell
scp "D:/ntu_related/dissertation/GPU cluster/bridge_ema_pair_v1.zip" "D:/ntu_related/dissertation/GPU cluster/bridge_ema_pair_v1.sha256" zixiao005@10.97.216.128:/projects/Zeroshot/
```

集群 Bash：
```bash
cd /projects/Zeroshot
mkdir -p logs
sha256sum -c bridge_ema_pair_v1.sha256 && unzip -n bridge_ema_pair_v1.zip
EMA_TRAIN=$(sbatch --parsable /projects/Zeroshot/bridge_ema_pair_v1/train_ema.sh)
echo "EMA训练：$EMA_TRAIN"
```

先确认训练 COMPLETED 0:0，记录真实编号，断连不会取消 sbatch 作业；变量可能在重连后消失。

```bash
sacct -j "$EMA_TRAIN" --format=JobID,State,ExitCode,Elapsed,NodeList
tail -n 12 "/projects/Zeroshot/logs/bridge-ema-train-$EMA_TRAIN.out"
```

## 训练完成后评估

```bash
EMA_EVAL=$(sbatch --parsable --export=ALL,EMA_TRAIN_ID="$EMA_TRAIN" /projects/Zeroshot/bridge_ema_pair_v1/evaluate_ema.sh)
echo "EMA评估：$EMA_EVAL"
sacct -j "$EMA_EVAL" --format=JobID,State,ExitCode,Elapsed,NodeList
tail -n 8 "/projects/Zeroshot/logs/bridge-ema-eval-$EMA_EVAL.out"
```

全部 1836 个训练窗口和 40 个开发窗口，各用 3 个相同采样种子。分别比较原生 100 步 DDPM 与一步 x0 读出，不能把两种协议混成一个结论。逐批验证 raw/EMA 的 CUDA 随机流相同；CLIP、checkpoint 在评估前后不可变化。报告包含整体、逐任务、逐演示的路径/终点位姿误差、夹爪平衡准确率和切换时刻命中率。切换时刻命中率仅指预测开合标签，不代表真实抓取成功。

## 如何判读

训练 PASSED 只证明执行和完整性检查通过；评估 execution_checks_passed 也不代表 EMA 有效。首先检查相同协议下的验证物理误差及逐演示结果，兼顾训练误差和开合切换。若只改善夹爪或训练集，不能宣称位姿泛化改善；若只改善一步读出，不能宣称 100 步扩散链改善。

若两种权重仍差，应降低 EMA 的排查优先级。若 EMA 改善，只能说明它对这份配方有帮助，不能认定它是唯一根因。一次训练和三个采样种子不是三次独立训练；重叠窗口也不是独立演示。未读取预留测试目标，未运行机器人仿真。

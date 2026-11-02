# 219059在模型加载前CUDA初始化失败

用户终端显示18文件校验通过，Torch随后报CUDA driver initialization failed，并在run_audit.py环境检查处退出。尚未调用source_root、读取数据或加载端点；0次训练、0个模型预测。本次没有动作诊断结果，不能判模型好坏。

核对audit.sh与已成功运行217820的formal.sh：GPU均请求a6000:1，使用同一official-pusht-v1 Python及其lib目录。没有发现新脚本漏请求GPU或改用CPU解释器的证据；但实际分配GPU、可见设备、节点驱动和库加载状态未收到，不能认定是节点故障，也不能据此安装/升级Torch或改模型。

先通过sacct读取219059实际节点与AllocTRES。若需GPU健康检查，只提交一份2分钟资源诊断：记录Slurm GPU环境、nvidia-smi，并分别调用既有official与multitask环境的torch.cuda.init，不加载模型/数据、不下载/安装。保留旧作业与.submission保护，不自动重投主诊断，不把环境失败当模型失败。

## 健康检查219094结果

同一gpu-a6000-1分配确有1张A6000、4CPU/22GiB；nvidia-smi能通过NVML列出设备和驱动595.71.05。official Torch1.12.1/CUDA11.6与project Torch2.5.1+cu118均在裸torch.cuda.init失败；没有模型或数据介入。当前应优先处理共享GPU访问环境，不能认定某个模型模块引起失败，也不能仅凭NVML成功认定CUDA正常。节点驱动、设备权限/作业隔离、共享库加载的具体原因尚未区分。

CUDA_VISIBLE_DEVICES=0与SLURM_JOB_GPUS=1可能涉及可见设备重编号，单凭这两个值不能判为错配，不手工覆盖Slurm可见设备设置。NVIDIA资料：https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html 及 https://docs.nvidia.com/deploy/nvml-api/nvml-api-reference.html 。

恢复建议只做一次换A40的原动作短检查，命令行覆盖GPU请求，保持原audit.sh字节、协议、30分钟上限、无训练、无requeue和加载前CUDA保护。两项旧作业均已失败，可用独立.retry_a40_after219094目录保护人工恢复，不删除原.submission。未提交该恢复；若另一节点仍在裸初始化失败，先保留日志并查共享环境/联系管理员，不继续循环换模型或升级依赖。

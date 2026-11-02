# A6000启动故障与可选路径

228978预检已在gpu-a6000-1完成；229016正式作业及229042健康检查均在同节点CUDA启动失败，均未进入原训练更新。该节点申报10张A6000，当前仅证明两次分配不可用，不证明其余设备可用或全部损坏。

229016的nvidia-smi显示PCI 00000000:00:1C.0，温度/风扇/功耗ERR；229042再次显示该PCI路径，UUID GPU-1331d351-1136-494f-8045-05df4cb4bc18，SLURM_JOB_GPUS=3、CUDA_VISIBLE_DEVICES=0，torch.cuda.init抛出No CUDA GPUs are available。预检日志是02:12.0。PCI地址相同不能独立证明所有虚拟化环境下始终为同一物理卡。

不手动把CUDA_VISIBLE_DEVICES改成未分配的GPU标号，不覆盖Slurm资源限制，不重复排同样作业碰运气。当前完整DINO缓存、原预检/实验包/权重和失败记录保留。

可选路线：等A6000运行环境恢复后复用旧CLIP结果，或转A40并重新配对训练CLIP和DINO以控制训练显卡差异。后一方案需要额外一个CLIP训练，预算两个8192步，而不是把旧A6000的CLIP与新A40的DINO称为严格单变量对照。按现有A6000测量预计合计2至3小时级，A40实际速度尚未测量；不自动启动，已请求用户选择。

## 可转交管理员的故障说明草稿（未发送）

On gpu-a6000-1, job228978 completed the CUDA/DINO preflight, but job229016 failed before training with CUDA unavailable and nvidia-smi ERR fields. A single retry, job229042, failed in a60-second CUDA startup check before any model construction or optimizer update.

The latter allocation reported SLURM_JOB_GPUS=3 and CUDA_VISIBLE_DEVICES=0. nvidia-smi identified GPU-1331d351-1136-494f-8045-05df4cb4bc18, PCI00000000:00:1C.0, NVIDIA RTX A6000. torch.cuda.init raised RuntimeError: No CUDA GPUs are available. The earlier successful preflight showed PCI00000000:02:12.0. The logged driver version was595.71.05; the project uses its existing PyTorch2.5.1+cu118 environment.

Could you inspect the allocated device and Slurm/device/driver mapping, and remove the unavailable device from allocation while it is being repaired? We have stopped retries and have not changed CUDA_VISIBLE_DEVICES or the training environment.

# 211279失败：断点恢复时归一化张量回到CPU

用户报告A6000/1:13/1:0，Conv2d输入CPU、权重CUDA。当前只有终端回溯，未本地读取完整report，实际completed_updates数不冒称已核验。

根因已通过源码和本地真实GPU复现：作者[DictOfTensorMixin加载逻辑](https://github.com/real-stanford/diffusion_policy/blob/5ba07ac6661db573af695b419a7947ecb704690f/diffusion_policy/model/common/dict_of_tensor_mixin.py)会使用检查点张量重建normalizer的ParameterDict。新入口将检查点加载到CPU，再调用load_state_dict时，归一化scale/offset也重建在CPU；原_normalize显式将输入移动到scale.device，所以已经搬到GPU的图像又回到CPU，与CUDA卷积不匹配。

这是助手新增入口的恢复错误。先前小模型CPU测试用的简化normalizer不具备上述重建行为，因此漏检；不是训练方法失败，也不构成Bridge泛化根因证据。旧资源/在线预检没有执行这条CPU检查点恢复路径，它们的通过结论不受此错误推翻。

## 唯一修复

新独立overlay只改变restore函数：加载前记录目标设备，加载后将原模型及EMA的全部张量搬回该设备，并检查全部注册参数/缓冲区。优化器、LR、EMA计数、RNG的原恢复逻辑和精确重放自检保留。

原训练包/协议/数据/2万预算/损失/随机计划/开发种子均未修改，失败输出保留。修复入口校验原9文件SHA，显式绑定修复模块后执行原训练runner，避免同名导入命中旧helper。不从失败的启动自检检查点恢复，重新开始相同种子的原生20k任务；不另交资源或接线预检。

## 验证

本地RTX4060 Laptop GPU、Torch2.11：小Conv2d策略采用真实作者normalizer及公开的字典加载行为，旧restore复现了逐字相同的CPU/CUDA报错；修复后raw与EMA归一化张量均在GPU，下一次更新的loss及raw/EMA/Adam/LR/计数/全部RNG摘要完全相同。真实原runner的导入绑定和原文件篡改拒绝也经过检查。并非已经在集群Torch1.12/A6000验证，集群必须继续通过原启动自检；未加载大策略/CLIP、训练数据或仿真，也未评分开发或最终集。

发布包：[pusht_native_device_repair_v1.zip](../cluster_packages/pusht_native_device_repair_v1.zip)。详见[GPU回归记录](../reports/pusht_native_device_repair_gpu_checks_20261018.json)及[发布记录](../reports/pusht_native_device_repair_release_20261018.json)。

PowerShell上传：

```powershell
scp "D:\ntu_related\dissertation\Zero_shot\cluster_packages\pusht_native_device_repair_v1.zip" zixiao005@10.97.216.128:/projects/Zeroshot/
```

集群提交见包内README；日志路径仍为logs/pusht-native20k-新作业号.out。先看到PUSHT_RESTORE_DEVICE PASSED，再看到PUSHT_NATIVE_STARTUP PASSED，才是本次恢复路径通过。

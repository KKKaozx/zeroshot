# 冻结DINOv2独立对照资源预检

按老师第六封邮件的“项目前端明显较差”分支准备，不解冻或覆盖CLIP主模型，不延续已关闭的层数和空间旁路试验。只选择一个新骨干DINOv2 ViT-L/14，不同时扩出R3M搜索。

## 已实际完成

- 从Meta官方`facebook/dinov2-large`下载固定提交`47b73eefe95e8d44ec3623f8890bd894b6ea2d6c`；权重1217522888字节，SHA256与上游公开LFS相同：`399fba97a95f22c36834418bc69373364a99af3a1153da1c0fb31db567c92e23`。配置和处理器也逐项固定SHA；无远程代码执行、无新包安装。
- 本机RTX4060 Laptop实际加载该权重，使用既有动作诊断中的两帧训练图像，不读新留出标签。冻结骨干前后state SHA一致，输出`[2,257,1024] float32`；原8层Adapter能接收并获得有效梯度，状态列不变，没有新增可训练宽度映射。梯度检查使用合成12-token文本，仅验证接口；服务器须改用原213439合格缓存的真实固定CLIP文本。
- 原始图像与原在线RGB往返得到的token相同，固定pad8执行相同。白色图像的解析归一化数值差约1.19e-6，来自FP32图像插值/运算，按3e-6检查，不用字节相等替代颜色数值检查。没有策略优化更新、任务仿真或效果结论。
- 小预检包12文件，5个旧核心文件字节原样复用；新包散列、Python3.9、Bash和验证器正反例共20项检查通过。未重复旧原生/CLIP资源探针、完整恢复或拟合实验。

## 受控边界

DINOv2-L/14宽度1024、patch14，在224方形输入下使用CLS+256patch token，保留原8层Adapter、池化、投影及CLIP文本。标准DINO最终归一化token与原CLIP视觉last_hidden_state的内部归一化不同，属于骨干表示差异，应记录。

DINO默认processor resize256后crop224，会改变原96×96画面的可见范围。本独立对照采用原CLIP的bicubic resize224/crop224几何，保持画面范围，改用DINO原生ImageNet均值/标准差；这是为配对实验显式修改默认几何，不能称原封不动的DINO默认评估。服务器执行时还会与原CLIP实际processor逐项核对几何。未同时改分辨率、token宽度、Adapter层数、语言或动作目标。

来源：[官方配置](https://huggingface.co/facebook/dinov2-large/blob/47b73eefe95e8d44ec3623f8890bd894b6ea2d6c/config.json)、[官方处理器](https://huggingface.co/facebook/dinov2-large/blob/47b73eefe95e8d44ec3623f8890bd894b6ea2d6c/preprocessor_config.json)、[原作者实现](https://github.com/facebookresearch/dinov2)。这些资料不证明DINO会改善本任务。

## 本次服务器作业

复用210763相同64训练窗口/128帧和训练专用归一化数据，读取213439的原固定文本；不重新选窗、不扫描全数据、不生成全部11356帧缓存。项目现有Py3.10环境编码DINO，作者Py3.9环境调用原样`resource_probe.measure`执行5次丢弃更新和两窗采样形状检查。原资源测量函数只是5步固定batch计时，不取代之后正式84168预算或正式学习率/恢复协议。没有闭环、任务评分、最终集访问、权重输出或自动正式提交。

申请A40一张、4CPU、40GiB、20分钟硬上限；真实服务器耗时尚未测定。这个上限不代表之后正式同预算训练只需20分钟，原8层和大U-Net仍需较长训练。

## 上传与提交

预训练权重约1.22GB，另上传一次，不夹入22KB代码包。PowerShell：

```powershell
scp -r "D:\ntu_related\dissertation\Zero_shot\baselines\dinov2_large_frozen_assets_20261023" zixiao005@10.97.216.128:/projects/Zeroshot/
scp "D:\ntu_related\dissertation\Zero_shot\cluster_packages\pusht_dino_resource_v1.zip" zixiao005@10.97.216.128:/projects/Zeroshot/
```

集群：

```bash
cd /projects/Zeroshot
unzip -n pusht_dino_resource_v1.zip
bash /projects/Zeroshot/pusht_dino_resource_v1/submit_once.sh
```

SHA256：`717640da566b8b603f8af0ddd031460e804d95f367985af39038a844f0868e88`。

保留原210763、213439和baseline_setup资产目录。提交前校验骨干大权重SHA，一次提交保护不自动删除。日志`logs/pusht-dino-resource-编号.out`、输出`runs/pusht-dino-resource-编号`。完成核验：

```bash
/projects/Zeroshot/envs/official-pusht-v1/bin/python /projects/Zeroshot/pusht_dino_resource_v1/run_probe.py verify --output /projects/Zeroshot/runs/pusht-dino-resource-实际编号
```

后续正式对照还需训练专用DINO全缓存、真实worker IPC一致性和checkpoint骨干身份绑定；当前没有自动通过这些项或提交84168更新。它们只覆盖更换骨干新引入的路径，不重做已有效的旧模块。

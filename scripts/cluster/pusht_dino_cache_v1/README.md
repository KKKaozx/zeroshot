# 冻结DINO训练帧缓存与真实在线接线

承接220882已通过的资源探针，不重复五步资源更新。只编码原90训练演示/11356帧，仍是原10726窗口来源；不抽新窗、不读开发/最终标签、不更新Adapter/U-Net/视觉模型、不计算奖励、不自动提交长训练。原CLIP模型和缓存不修改。大权重不随ZIP上传，继续用服务器已有冻结DINO资产。

DINO-L/14输出257×1024原生最终归一化token。保持旧CLIP resize224/crop224画面几何，使用DINO ImageNet均值/标准差；这是与DINO默认resize256/crop224的明确偏离。固定CLIP文本直接复制213439/text.npz原字节，不重算文本，不新增宽度映射。原8层Adapter、投影、作者U-Net/归一化均复用。

原official-pusht-v1 Python读取已验收训练数据；新DINO worker使用现有multitask-preflight-v1运行时。私有继承socket、numeric NPZ、32帧请求内pad8与原缓存一致。生成约11.13GiB visual.npy，frame_ids/text来自原训练缓存；不落盘另一份原始图像，不复制CLIP视觉缓存。

缓存对照220882原128训练帧。关闭缓存worker后，以新worker重新加载DINO，经过原LiveProjectPolicy回调检查1/2/4批量及补齐路径；原8层Adapter上下文与2窗DDPM100完整/执行动作均比较，固定采样seed1101。策略从原seed42后端+seed43前端新建，只推理不训练，结果不能解释为Push-T能力。token容差atol1e-4/rtol1e-5，context绝对1e-4，动作绝对1e-3仿真像素，阈值预先固定，失败不放宽。

空间检查在提交前和作业内各执行一次：du项目已占用量、共享文件系统可用量，按保守200十进制GB项目预算扣除完整11.13GiB缓存，再保留24GiB给后续完整检查点/原子保存和余量。不是权威账户quota查询；若不足直接停止，不自动删除任何既有文件或降精度。失败新输出保留，重试需先检查原因，不自动重投。

申请A40/4CPU/40GiB/30分钟硬上限，不代表实际必需30分钟。原220882新骨干吞吐估算11356帧纯编码约186秒，另有数据读取、IPC、散列和接线检查开销。A6000曾CUDA初始化失败，继续用已验证A40，不改变环境/依赖。

提交：bash /projects/Zeroshot/pusht_dino_cache_v1/submit_once.sh

完成验证：/projects/Zeroshot/envs/official-pusht-v1/bin/python /projects/Zeroshot/pusht_dino_cache_v1/run_cache.py verify --output /projects/Zeroshot/runs/pusht-dino-cache-实际编号

输出report.json、backbone_binding.json、training_data.json、live_action_identity.npz及两个worker日志。下载这些小结果即可，11.13GiB visual.npy留服务器。backbone_binding含报告/缓存/骨干/预处理/文本/原始像素身份，后续正式checkpoint必须嵌入并核对；本包无训练/checkpoint保存入口，故不声称正式恢复绑定已经执行。正式预算/恢复入口须下一步单独审阅后生成。

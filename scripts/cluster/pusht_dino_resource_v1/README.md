# 冻结DINOv2独立对照的短资源预检

承接老师第六封邮件。保留冻结CLIP主模型和所有原端点，仅新增视觉骨干对照资源探针。使用Meta官方facebook/dinov2-large固定提交47b73eefe95e8d44ec3623f8890bd894b6ea2d6c，safetensors与公开LFS SHA相同。禁止远程代码、联网自动下载、安装包或正式训练自动提交。

DINOv2-L/14在224方形图像下输出257x1024原生最终归一化token，不增加宽度映射。沿用原8层Adapter、池化、投影、CLIP固定文本与agent_pos。DINO的默认预处理是resize256/crop224；本对照显式采用旧CLIP的resize224/crop224以保留画面范围，使用DINO原生ImageNet均值标准差。这一预处理偏离和DINO最终LayerNorm属于公开的表示边界差异，不能隐瞒为所有数值完全相同。服务器会核对旧CLIP实际几何配置。

只读取旧210763资源探针的同64训练窗口/128帧及训练专用归一化数据，不再扫描数据/抽取新窗、不读最终集、不重新跑原生或CLIP资源组。仅编码128帧和原样resource_probe.measure中的5个丢弃更新，最后2窗采样只检查形状，无环境/奖励。旧measure字节不变，此五步不是84168正式训练的缩短替代，不用于性能比较；不产出供后续训练加载的权重。

项目现有Py3.10/Torch2.5.1/transformers5.17环境编码DINO，原Py3.9/Torch1.12.1环境原样训练探针，进程分离，不修改环境。只申请A40/4CPU/40GiB/20分钟/no-requeue。大权重不包含在小ZIP中，需单独上传dinov2_large_frozen_assets_20261023文件夹。原210763和213439目录保留。

提交：bash /projects/Zeroshot/pusht_dino_resource_v1/submit_once.sh

完成核验：/projects/Zeroshot/envs/official-pusht-v1/bin/python /projects/Zeroshot/pusht_dino_resource_v1/run_probe.py verify --output /projects/Zeroshot/runs/pusht-dino-resource-实际编号

报告明确未做全训练缓存、真实新worker IPC或正式checkpoint骨干绑定。先据资源结果评估既定同84168训练预算，不承诺短时间内完成正式实验或效果改善。旧CLIP主模型、219466/215999/212158及final50保持不变。

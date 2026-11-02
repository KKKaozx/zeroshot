# Push-T数据收尾与冻结前端资源预检

2026-10-18，210690完整report已本地读取并保存指纹。数据层收尾；不再重复210557盘点、210631下载、210690全量解码，不声称Bridge泛化问题已修复。

## 已完成的数据事实

- 206条独立演示、25,650帧，原始图像float32/96×96×3，实际值域[65,255]；除255后[0.25490196,1]。原始float保持不变，没有转uint8或修改数据。
- 图片、动作、状态各160块、演示边界1块，预期块存在、全部解码及有限值检查通过。数据仍在`runs/pusht-data-prepare-210631/extraction_staging/pusht/pusht_cchi_v7_replay.zarr`，须保留210631目录。
- 按作者原配方90条训练演示、10,726窗口。作者validation是训练下采样的补集，共116条/13,482窗口；不是最初val_ratio=0.02的4条。这个计数不是新实验划分，不能称已做独立泛化评价。
- 本地分析：`reports/pusht_data_analysis_210690.json`；完整报告：`reports/pusht_float_data_210690.json`。

## 选定接口及环境隔离

作者环境Python3.9/Torch1.12.1/torchvision/robomimic/diffusers/Zarr已工作；项目环境Python3.10/Torch2.5.1/transformers5.17已工作。**不尝试把依赖强行装到一起**：冻结CLIP只在项目环境编码，float32 NumPy特征缓存交给作者环境。Adapter本身只依赖Torch，其固定源码不变，仅在执行时添加future annotations头，解决Python3.9不支持求值Tensor|None的注解问题。

原生组保留作者CNN/原裁剪前处理；项目组为冻结CLIP图文特征→原八层Adapter/cls_patch_mean→Linear1024→64。两观察各加原2维位置，条件合计132维，后端保持作者完整[512,1024,2048]、时间128、epsilon/cosine100/DDPM100。原生位置在66维向量中可能在前或后：预检先对原编码器做位置干预，识别其真实列位置并保留，不预设拼接顺序。

这是**前端系统替换**（包含各自图像前处理、特征和映射）的检验，不是仅Adapter单层的因果实验。项目CLIP前处理固定；冻结特征可缓存，不使用Bridge权重，也不把随机前端插到旧作者已训练后端。

## 唯一有界GPU预检

新包`cluster_packages/pusht_frontend_resource_v1.zip`，8文件，具体大小/SHA见`reports/pusht_frontend_resource_release_20261018.json`。单A6000、CPU4、主存22GB、上限45分钟；未提交。

1. 从90条训练演示的固定池抽64个窗口，保留两个真实观察和16个动作；只从训练90条的全部帧取动作/位置归一化拟合输入。两组用相同统计及反归一化。这有意区别于原作者全replay统计。
2. 项目环境只编码这128次观察及固定任务文本，不缓存整个训练集。CLIP所有参数requires_grad=false，eval/inference；前后完整参数指纹相同且无梯度才通过。
3. 作者环境先测原生组，再在独立进程重建同一随机初始化后端测项目组；共同后端/normalizer SHA必须一致。有效batch64，microbatch8×累积8，FP32，同AdamW和EMA。每组5次更新，前2次暖机，后3次计时；只是重复同64窗的资源测量，不作拟合或泛化判定。
4. 两组检查前端/后端梯度，项目额外检查Adapter和映射梯度；各一次EMA的DDPM100采样（batch2）测形状、有限性和速度，不执行环境、不读开发或最终回合。
5. 资源报告给出逐步时间、显存、参数量及以84,168次更新作参考尺度的训练耗时投影。该尺度接近原检查点保存进度，**不是已批准正式预算**；投影不含完整缓存、保存日志、闭环评价和实时CLIP通信成本。

本资源探测没有配对两组的扩散噪声抽样，不能用其loss比较效果；正式对照仍须锁定匹配噪声/样本计划和公平预算。整个训练集FP32视觉缓存按窗口公式推得约11.13GiB，资源脚本还会用实际训练帧数报告，不能假装只有小缓存的内存/I/O成本。

## 停止与后续边界

任一接线、冻结、源指纹、梯度、显存/运行失败即停止，保存阶段错误，不自动安装、降规格、追加更新或提交正式训练。预检即使通过，也不说明新初始状态任务能成功。

本次完整闭环尚未实现实时CLIP跨环境通信，不能把训练缓存当成部署路径。资源可承受后，还需一次缓存/实时同输入一致性与执行接线检查，锁定训练/开发/最终回合与指标，然后才决定正式匹配预算。原0.9309仅为已完成的预训练参考，不能作短预算项目的公平比较成绩。若匹配配方下原生参考没有闭环能力，不作项目模块归因。CLIP永不解冻，207558保留，Bridge最终25封存。

## 本地验证与提交

Python3.9语法、CLI、Bash、ZIP/SHA检查通过。小型两层Adapter证实延迟注解前后state/output逐元素一致；三种位置列布局保持、Adapter/映射梯度及坏缓存拒绝通过。没有本地真实CLIP/作者完整U-Net或GPU运行；Torch1.12实际运行仍待本预检。

Windows PowerShell：

```powershell
scp "D:\ntu_related\dissertation\Zero_shot\cluster_packages\pusht_frontend_resource_v1.zip" zixiao005@10.97.216.128:/projects/Zeroshot/
```

集群首次解压与提交：

```bash
cd /projects/Zeroshot
if [ ! -e pusht_frontend_resource_v1 ]; then
  /projects/Zeroshot/envs/official-pusht-v1/bin/python -m zipfile -e pusht_frontend_resource_v1.zip .
  PUSHT_RESOURCE=$(sbatch --parsable /projects/Zeroshot/pusht_frontend_resource_v1/preflight.sh)
  echo "Push-T前端资源预检：$PUSHT_RESOURCE"
else
  echo '目录已存在，请先核对已有作业，避免重复覆盖/提交。'
fi
```

查看：

```bash
sacct -j "$PUSHT_RESOURCE" --format=JobID,State,ExitCode,Elapsed,NodeList
tail -n 25 "/projects/Zeroshot/logs/pusht-front-resource-$PUSHT_RESOURCE.out"
```

仅全部阶段通过才生成`runs/pusht-front-resource-JOBID/report.json`；失败看同目录`STAGE_error.json`。没有正式入口或自动后续作业。

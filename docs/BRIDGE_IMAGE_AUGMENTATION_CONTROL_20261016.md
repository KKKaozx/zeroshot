# 训练图像增强：单项泛化对照

**最新：完整工件独立验收已结束。** 206915的29文件/13预测/增强轨迹核验通过，开发8.499733cm/15.764689°，主判据失败。此候选收口，不重复下载/编码/训练/采样。见[独立结果](BRIDGE_IMAGE_AUGMENTATION_RESULT_206915_20261016.md)；后续只读条件检查与同配方数据覆盖准备见[下一步决定](BRIDGE_GENERALIZATION_NEXT_STEP_20261016.md)。下文待复核、下载和提交命令均为历史。

**206915固定预算已结束：没有改善（终端证据，待工件复核）。** COMPLETED/0:0、56分43秒、A6000，8192更新；原40开发8.499733cm/15.764585°，高于旧控制8.349127cm/15.256048°及静止8.212326cm/15.098500°，未过事前主要判据。训练7.296mm/2.548°、拟合门槛False不决定追加。候选结束，不扩展步数、强度或种子；后续仅小工件核验，不重跑编码/训练/采样。见[结果与下载命令](BRIDGE_IMAGE_AUGMENTATION_RESULT_206915_20261016.md)。下文排队、未完成或提交命令为历史，勿再执行。

**当前唯一正式作业：206915，预检206895。** 用户在给出的条件验证提交命令后返回正式编号，因此推断集群验证器已成功退出；完整预检报告未下载本地独立复核，验证器的具体显存/耗时输出尚未收到。正式排队/运行/完成状态未收到，无新泛化结论。不重提或因训练拟合不足延长，原202278控制复用。下文预检待验证、尚无正式编号或提交命令均为历史记录，勿重复执行。

```bash
sacct -j 206915 --format=JobID,State,ExitCode,Elapsed,NodeList
tail -n 15 /projects/Zeroshot/logs/bridge-aug-fit-206915.out
```

若PENDING且没有日志，等待即可；需要查看预计开始时间时运行`squeue --start -j 206915`，预计时间不是承诺。结束后按固定原40开发窗位置与旋转评价；在拿到完整报告/预测前不判定改善。

**最新状态：预检206895已完成。** 用户贴出的sacct为COMPLETED/0:0、4分23秒、gpu-a6000-1。常学习率与恢复同学习率的首损失及最终位置/旋转终端值一致；门槛False仅说明几步预检未达到拟合标准，不是泛化判定，也不要求据此加步数。完整报告未在本地取得，先运行包内只读验证器检查来源/原202278初始化、冻结、恢复和资源，成功才提交唯一正式组。没有新正式编号，不重复预检。下文“没有提交”及首次提交命令为历史记录。

一次执行以下命令，验证失败不会提交作业；不依赖已结束作业的Slurm依赖记录：

```bash
if /projects/Zeroshot/envs/multitask-preflight-v1/bin/python /projects/Zeroshot/bridge_image_augmentation_v1/validate_preflight.py /projects/Zeroshot/runs/bridge-aug-preflight-206895; then
  AUG_FIT=$(sbatch --parsable --export=ALL,AUG_PREFLIGHT_JOB=206895 /projects/Zeroshot/bridge_image_augmentation_v1/fit.sh) && echo "图像增强正式对照：$AUG_FIT"
fi
```

本轮按原8192预算结束，以原40开发窗位置和旋转为主要判据；预检训练误差不决定延长。正式脚本也会再次验证报告来源和配额规划，实际显存与保守耗时以验证器打印值为准。

```bash
sacct -j 206895 --format=JobID,State,ExitCode,Elapsed,NodeList
tail -n 15 /projects/Zeroshot/logs/bridge-aug-preflight-206895.out
```

取得COMPLETED/0:0后，再执行只读验证并贴输出：

```bash
/projects/Zeroshot/envs/multitask-preflight-v1/bin/python /projects/Zeroshot/bridge_image_augmentation_v1/validate_preflight.py /projects/Zeroshot/runs/bridge-aug-preflight-206895
```

用户已确认目标：改善独立留出演示泛化，训练拟合只作检查。此前固定视觉均值206274的完整候选失败并收口；本轮保留真实图像，不调整该候选强度，不重跑原控制。

## 论文依据与实际差异

已成功执行的PushT195527权重配置为96×96输入、84×84裁剪、eval_fixed_crop=true；不是通用示例配置里的76×76。[固定提交的CropRandomizer](https://github.com/real-stanford/diffusion_policy/blob/5ba07ac6661db573af695b419a7947ecb704690f/diffusion_policy/model/vision/crop_randomizer.py)训练随机裁剪、评估中心裁剪。项目保存的[实际权重配置](../reports/official_pusht_195527/checkpoint_config.yaml)可核查84设置。

迁到Bridge的[作者参照配置](../cluster_packages/bridge_author_reference_v1/training_config.yaml)显式crop_shape:null；项目原128窗模型则一直使用每张图唯一的固定CLIP token。现有索引未见同一Bridge模型的训练图像增强实验。这是待测试的训练配方差异，不能先认定为根因。

## 固定一项改动

在现有224×224、完成CLIP归一化的训练图像上，边缘复制填充8像素，各轴均匀抽取0…16整数偏移，裁回224×224。等价于每轴最多8像素的小平移；边缘内容可被截去最多8像素。没有缩放、插值、翻转、颜色变换或动作/状态标签变换。中心偏移8精确还原原图；复制与裁切也与逐通道CLIP归一化交换，因此不是另一套归一化。

这是适配CLIP的保守增强，借鉴作者训练随机/评估固定原则，**不等于84/96裁剪的精确复现**。增强可能帮助减轻固定画面细节的记忆，也可能移除有用位置线索；由留出结果决定。

只对128训练窗在线增强并用冻结CLIP编码；评估仍用原始固定图像token。两次重复当前观察的Adapter调用共用一次增强。增强使用独立CPU随机发生器510000+update，保留旧dropout和扩散噪声种子。真实状态、语言、标签、原128训练/40开发与模型参数集合不变。原202278真实状态控制直接复用。

## 预算和判据（模型更新前冻结）

初始化43、8192更新、批8，每窗512曝光；前4096步1e-4、后4096步1e-5；原x0/DDPM100和三个采样种子保持一致。仅一个增强强度、一次初始化、最终权重。不因训练误差不够低追加拟合。

主要结果为原40开发窗位置与旋转，保留全部10演示、5任务。报告终点、演示等权、逐任务/演示与训练误差，不能换主要指标。沿用先前完整候选筛选：位置≤7.391093cm，旋转≤15.118500°，两项低于原控制，至少6/10演示位置低于原控制。该规则是工程筛选，不是统计显著性认证。比较静止8.212326cm/15.098500°；辅助状态读出6.2798cm仅作不同模型的位置参照。

通过也只支持这一固定配方；不证明唯一根因、未见任务或抓取能力。失败则收口，不追加裁剪强度、颜色变换、种子或轮数。原40开发集多次用于诊断，不能称未触碰测试集。

## 实施与资源

原源码保留不动，薄包装记录实际生成源码、协议和继承源指纹。复用现有模型/数据/环境与原完整token缓存，增强仅在线生成，不存多视图缓存。冻结视觉编码器需留在GPU，先检验新增显存和耗时。预检只16次入口更新、四个训练窗，无开发评分或正式训练；检查中心编码、全局RNG隔离、恢复一致、冻结CLIP和原图评估路径。

正式要求预检峰值保留显存<7.5GiB、保守预计<9000秒；实际Slurm申请3小时。一份非CLIP最终权重留集群，预计<1GiB；上传只小源码包，无数据/模型。本地不进行8192正式训练。集群启动前以du估算200GiB计划容量并保留15GiB，不能视作实时配额查询。

冻结协议：[protocol.json](../cluster_packages/bridge_image_augmentation_v1/protocol.json)。准备时尚未启动神经网络预检或集群作业；后续执行状态另行追加。

## 本地入口检查已完成

仅执行预注册16次入口更新，耗时109.265秒、峰值保留5.074GiB、暖更新均值0.502秒。按8192更新并加1200秒余量估算5313秒（约1.48小时），是本地保守资源估算，不是排队或集群耗时承诺。原图中心在线编码与现有缓存最大差为0；增强不消耗原全局随机数，CLIP及其他冻结参数保持不变，Adam恢复后的参数、损失、预测精确一致。

独立审计只读小数组/日志，没有重复神经网络执行。五个训练快照位置及四元数测地角复算通过，最大指标差1.61e-6cm/0.0000444°；原本地初始化、真实状态及168原始输入检查与已有记录完全一致，原标签在1e-6内一致，16个增强批次与偏移/种子逐项核验。没有开发评分或本地正式训练，也没有产生权重。见[本地审计](../reports/bridge_image_augmentation_local_preflight_20261016/independent_audit.json)。Slurm脚本bash -n通过。

当前没有提交集群作业；先做下列预检。原泛化失败结论不变。完整拟合程序沿用已有核心，通过带计数保护的薄包装加入图像增强，实际执行源码留存于报告；原冻结实验源码未修改。

## 下一步命令（只提交一次预检）

本地PowerShell：

```powershell
scp "D:/ntu_related/dissertation/Zero_shot/cluster_packages/bridge_image_augmentation_v1.zip" "D:/ntu_related/dissertation/Zero_shot/cluster_packages/bridge_image_augmentation_v1.sha256" zixiao005@10.97.216.128:/projects/Zeroshot/
```

集群终端：

```bash
cd /projects/Zeroshot
mkdir -p logs
sha256sum -c bridge_image_augmentation_v1.sha256 && unzip -n bridge_image_augmentation_v1.zip && AUG_CHECK=$(sbatch --parsable bridge_image_augmentation_v1/preflight.sh) && echo "图像增强预检：$AUG_CHECK"
```

查看状态：

```bash
sacct -j "$AUG_CHECK" --format=JobID,State,ExitCode,Elapsed,NodeList
tail -n 12 "/projects/Zeroshot/logs/bridge-aug-preflight-$AUG_CHECK.out"
```

预检完成后再用包内validate_preflight.py核对与原集群202278输入/初始化、恢复和资源。fit.sh内也会强制该验证；尚未提交正式训练。无需更新pip或重装环境。

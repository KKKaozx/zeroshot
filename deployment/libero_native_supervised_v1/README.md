# LIBERO单任务域内监督对照：关顶层抽屉

2026-10-31，用户明确同意准备本对照。**代码和数据已准备；未提交训练、未运行新策略仿真，尚无任务成功成果。** 这是目标域监督工程参照，不是OXE→LIBERO零样本结果，不替代老师五条件/13模型。原229116、旧抓取成果与已关闭候选均保留。

## 一次训练、一项可见验收

任务为`libero_90/KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet`，指令`close the top drawer of the cabinet`。已核对不在正式40任务清单中。固定demo0–19训练、20–24开发、25–29只作最终五个初态回合；剩余演示不使用。准备训练包时没有解码评估演示的后续状态、动作或图像，整HDF5仅作文件SHA绑定。演示ID分开不代表未见任务/场景泛化。

本任务示教的夹爪命令始终是原生`-1`张开，不开展夹爪切换专项。原生动作各维按训练数据min/max归一化；常量维scale=0，逆变换严格返回该训练常量。此行为不是新夹爪能力。

## 输入、标签与执行合同已具体实现

保留冻结DINOv2 ViT-L/14、冻结CLIP文本、8层交叉注意力Adapter（CLS+patch mean）、多尺度U-Net/100步余弦x0 DDPM。新鲜seed42初始化Adapter、9→1024状态端口和原生动作后端，不读取/覆盖OXE权重，不把旧位姿权重改名为原生命令模型。可训练参数131,261,063；不声称只训练一个很小Adapter。

- 当前图像：agentview原始OpenGL128 RGB，翻转一次，再按项目224双线性/CLIP归一化处理，DINO封装转ImageNet通道统计；在线同一函数。
- 当前状态：实测Panda七个关节位置及两指位置共9D，均值/标准差仅由20条训练示教拟合，不输入demo编号、时间索引或未来状态。
- 数据生成代码先执行`actions[i]`再保存`obs[i]`。因此训练输入`obs[i]`的第一个目标为`actions[i+1]`，未来16条原生20Hz命令，时距0.05–0.8秒。每条演示最后无后续命令的观测不使用，尾部不足16用有效mask，不跨演示拼接。
- 输出7维为原生归一化OSC位置增量3、轴角增量3和关爪正命令1。它不是XYZ+四元数，不进行四元数单位化/坐标迁移/位姿追踪。
- 执行前4条20Hz命令后重新观察，每个新回合最多200tick/50次策略调用。用原生成功谓词验收，不以离线loss或动作误差冒充关抽屉成功。

本对照还更换了训练域、机器人状态含义、动作监督和初始化方式，与229116不是单变量因果比较；它提供一个目标域可执行参照，不据其成败认定唯一迁移根因。单指令不能验证语言增强效果。

## 固定预算与停止条件

4096更新、batch16、Adam常LR1e-4、wd0、clip1，单训练seed42；固定预生成训练批次，最后4096端点，不按开发集挑checkpoint。单A40预计20–40分钟，Slurm硬60分钟；CPU准入后3300秒alarm包含加载、1888张训练/开发图像的冻结编码、训练、离线采样及最终验证打包。估计未经该版真实GPU训练测量，不保证完成时间。

冻结特征在本次作业内计算并留RAM/GPU，**不依赖旧token缓存、不写数GB磁盘缓存**。预计6–10GiB显存、24GiB主机内存，8GiB输出/原子保存预留；提交前CPU检查完整训练导入、权重/材料身份及Linux项目空间估计。`du`不等于权威账户配额。每512保存完整模型/Adam/RNG；中断保留最近原子检查点，不把中途optimizer.step的部分状态冒充完整更新。超时/非有限/依赖或身份失败结束，不自动续步/换seed/换任务/重试。

开发只报动作误差；不以动作MSE阈值无限阻止预先登记的五回合。有限、完整4096端点经读回验证后，另手动取回模型执行五回合；不由训练自动启动。三次采样draw不是三训练seed。

## 本地验证及当前边界

原始HDF5和输入绑定已保存；训练1509观测、开发379观测，输入压缩包约39MB。6项CPU检查覆盖因果移位/尾部mask、训练专用统计、常量动作逆变换、图像单次翻转、原生采样不做四元数解码、合成2步更新/完整Adam读回、新旧CLIP文本命名空间严格完整恢复。

另实际恢复了冻结CLIP文本全部权重，并用**一张训练图像**在CPU通过真实冻结DINO、真实CLIP文本、完整8层Adapter及完整原生后端的一次前向；没有真实优化更新或策略成绩。本机Torch2.11+cu128，集群仍须已有Torch2.5.1+cu118/A40。提交前的完整导入检查在GPU申请前运行；无本轮SSH/远端检查或自动提交。

复用已存在的资产，支持CLIP的safetensors/bin容器并核对同一文本参数指纹，不要求安装/升级环境：

```text
/projects/Zeroshot/dinov2_large_frozen_assets_20261023/{model.safetensors,config.json,preprocessor_config.json}
/projects/Zeroshot/baseline_setup/clip-vit-large-patch14-v1/{model.safetensors 或 pytorch_model.bin}
/projects/Zeroshot/envs/multitask-preflight-v1
```

训练包不含大编码器权重或原HDF5，集群不需要再上传383MB原始任务文件。原数据只在本地后续五回合初始化时读取。

## 审阅后手动上传与唯一提交

本机PowerShell：

```powershell
scp "D:\ntu_related\dissertation\Zero_shot\cluster_packages\libero_native_supervised_v1_20261031.zip" "zixiao005@10.97.216.128:/projects/Zeroshot/"
```

集群终端：

```bash
cd /projects/Zeroshot
test ! -e libero_native_supervised_v1
unzip -n libero_native_supervised_v1_20261031.zip
cd libero_native_supervised_v1
sha256sum -c SHA256SUMS
bash submit_once.sh
```

```bash
JOB=$(cat /projects/Zeroshot/libero_native_supervised_v1/job_id.txt)
tail -n 60 -F /projects/Zeroshot/logs/libero-native-supervised-${JOB}.out
```

已有job_id/锁/结果不能删除绕过；新训练编号须由用户回传。需要先单独做CPU准入时：

```bash
/projects/Zeroshot/envs/multitask-preflight-v1/bin/python -B /projects/Zeroshot/libero_native_supervised_v1/train.py --root /projects/Zeroshot --check-only
```

## 完成后：模型与五回合真实执行

按真实JOB替换后在本机下载小报告和约525MB完整模型：

```powershell
scp "zixiao005@10.97.216.128:/projects/Zeroshot/runs/libero-native-supervised-JOB/small_results.zip" "D:\ntu_related\dissertation\Zero_shot\reports\libero_native_supervised_results_20261031\small_results.zip"
scp "zixiao005@10.97.216.128:/projects/Zeroshot/runs/libero-native-supervised-JOB/policy.pt" "D:\ntu_related\dissertation\Zero_shot\reports\libero_native_supervised_results_20261031\policy.pt"
```

本机入口默认只就绪检查；只有显式`--run`才执行。固定demo25–29的states[0]和XML，每回合只恢复一次，不读后续专家动作/状态、不作中途纠正，共最多1000tick/250策略调用、含加载600秒，不自动重跑。输出五份GIF、动作记录、每回合原生成功值和5次分母。

```powershell
Set-Location "D:\ntu_related\dissertation\Zero_shot"
$py = "D:\ntu_related\dissertation\Zero_shot\training_cache\libero_native_env_v1\Scripts\python.exe"
& $py -B deployment/libero_native_supervised_v1/evaluate_local.py --root "D:\ntu_related\dissertation\Zero_shot" --checkpoint "D:\ntu_related\dissertation\Zero_shot\reports\libero_native_supervised_results_20261031\policy.pt" --output "D:\ntu_related\dissertation\Zero_shot\reports\libero_native_supervised_results_20261031\rollout1"
# 确认完成端点后，独立追加 --run；当前未执行。
```

无论成功几次，都只称该单任务域内监督对照。OXE源能力、跨机器人迁移、语言增强库及13模型主实验不因本包准备完成而解除。236744相对位姿训练已结束，终端已留档，原小ZIP尚待回传复算，不自动扩展旧分支。

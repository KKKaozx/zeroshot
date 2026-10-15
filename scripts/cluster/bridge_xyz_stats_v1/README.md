# Bridge 完整训练标签 xyz 统计

本包是官方/项目数据模块对照的统计步骤，不是新训练。复用固定 Bridge 数据划分，1836 个训练窗口、405 条演示、29376 个未来位置目标。归一化拟合仅使用训练标签；开发窗口不进行 __getitem__ 解码。索引器仍扫描记录元信息，不宣称从未读取开发记录字节。

输出：现有 xyz 数值的各轴 min/max/mean/std、0/1/5/50/95/99/100 分位数、超出采样 ±3 上限的比例、逐任务统计和极值窗口定位。现有单位是工具系相对米制位移/0.10，乘10得到厘米。

另计算一个全局训练集 limits 候选（各轴范围映射至[-1,1]，常量轴按官方规则处理），记录候选标准差、反变换误差、scale² 对 MSE 有效分量权重的倍率。只作数值描述，不保存到模型、不改变标签、不按每个任务单独拟合。四元数和夹爪不改变。

解释：范围或标准差不均衡只是提出归一化对照的依据，不能证明失败根因；极值也不自动认定为坏数据。完整训练样本的窗口/目标重复权重与当前实际训练一致，重叠窗口不是独立样本。

无需 GPU、无需安装依赖、无需下载数据/权重、不会实例化 CLIP 或加载 checkpoint。使用已有 multitask-preflight-v1 环境。复用原模型源文件只是为了导入已有 train.py 的划分函数并校验原版哈希；不训练模型。输出通常数十 KB，作业时间上限20分钟不是排队/运行时间承诺。

## 操作

本地 PowerShell：
```powershell
scp "D:/ntu_related/dissertation/GPU cluster/bridge_xyz_stats_v1.zip" "D:/ntu_related/dissertation/GPU cluster/bridge_xyz_stats_v1.sha256" zixiao005@10.97.216.128:/projects/Zeroshot/
```

集群 Bash：
```bash
cd /projects/Zeroshot
mkdir -p logs
sha256sum -c bridge_xyz_stats_v1.sha256 && unzip -n bridge_xyz_stats_v1.zip
XYZ_JOB=$(sbatch --parsable /projects/Zeroshot/bridge_xyz_stats_v1/xyz_stats.sh)
echo "位置统计：$XYZ_JOB"
sacct -j "$XYZ_JOB" --format=JobID,State,ExitCode,Elapsed,NodeList
tail -n 8 "/projects/Zeroshot/logs/bridge-xyz-stats-$XYZ_JOB.out"
```

完成后仅取回 `/projects/Zeroshot/runs/bridge-xyz-stats-JOBID.json`。重连后将变量替换为实际编号。统计执行通过不代表模型预测准确或归一化已经改善性能。

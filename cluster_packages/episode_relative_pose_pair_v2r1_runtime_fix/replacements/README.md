# v2r1：236711导入错误的运行依赖修复

236711在导入SciPy时退出，尚未创建优化器或更新参数；没有sacct终态，不把Python错误冒充独立查询到的Slurm状态。旧v2及job_id保留。此修复保留v2实验：两组冻结Adapter/history，只训练扩散后端及状态端口，各2048更新，常LR1e-5、同批次/RNG；1张A40，硬上限1小时、CPU准入后3300秒alarm、8GiB输出余量，源端保护与停止门槛不变。

具体修复：

- 公共9D特征中的xyzw四元数转矩阵用NumPy实现，不再导入SciPy；预先准备好的训练输入文件保持原字节。
- 原生模型中仅将未使用的CLIP构造依赖延迟到完整编码器构造时导入；缓存训练不构造编码器，不需要Transformers。原生loss/sample代码未改。
- `--check-only`先导入与GPU进程相同的训练/度量/验证/状态模块，不建模型、不调用CUDA。该步骤在sbatch前执行，避免这类缺依赖错误到分配GPU后才暴露。

本地已验证：在阻断SciPy及Transformers导入的独立进程中运行模块全部导入成功；512个旋转与SciPy参考最大矩阵误差5.56e-16，1876行状态与已准备参考最大误差1.67e-16；这些是实现验证，不是新增策略结果。旧CPU科学检查保留用于本地参考，其中SciPy仅用于测试对照，不属于集群训练入口。本轮没有真实训练、采样或新仿真，也未改环境。

小修复ZIP仅含安装器和6个替换文件。安装器先校验旧v2全体成员和job_id236711，再复制已上传材料到全新`episode_relative_pose_pair_v2r1`；不复制旧job_id或锁，不删除旧目录/结果，不提交作业。目标已存在即拒绝。完整科学协议仍见本包`protocol.json`及保留的`../episode_relative_pose_pair_v2/README.md`；新checkpoint格式保持v2。

本机上传：

```powershell
scp "D:\ntu_related\dissertation\Zero_shot\cluster_packages\episode_relative_pose_pair_v2r1_runtime_fix_20261031.zip" "zixiao005@10.97.216.128:/projects/Zeroshot/"
```

集群安装和手动提交：

```bash
cd /projects/Zeroshot
unzip -n episode_relative_pose_pair_v2r1_runtime_fix_20261031.zip
cd episode_relative_pose_pair_v2r1_runtime_fix
sha256sum -c SHA256SUMS
/projects/Zeroshot/envs/multitask-preflight-v1/bin/python -B install.py --root /projects/Zeroshot
cd /projects/Zeroshot/episode_relative_pose_pair_v2r1
bash submit_once.sh
```

查看新编号的同一日志：

```bash
JOB=$(cat /projects/Zeroshot/episode_relative_pose_pair_v2r1/job_id.txt)
tail -n 60 -F /projects/Zeroshot/logs/episode-relative-pair-v2r1-${JOB}.out
```

修复版未自动提交，不增加更新预算、种子、夹爪训练或仿真。依赖/恢复守卫再失败则保留报错，不通过删标记或修改门槛继续。

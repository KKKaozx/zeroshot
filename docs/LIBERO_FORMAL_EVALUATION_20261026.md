# LIBERO原指令正式评估：每任务100次

2026-10-26用户明确选择每个任务100次。四套件Spatial/Object/Goal/LIBERO-10，各10任务，共40任务，**每模型4000次**。老师的四种DINO条件各三训练种子，加CLIP完整增强单种子，共13模型，计划52000次策略回合。原生零动作参照另4000次，只做一份复用。五种评估改写不包含在上述计数中，仍需单独固定文本及预算。

已固定任务顺序、官方任务语言、官方初态文件及每槽环境/采样种子。第i次取官方初态 `i % 初态数`，所以100次不保证100个不同场景；结果会记录唯一初态数。13模型使用同一槽位，不能失败后换更容易的初态。

每回合5个原生稳定步后，最多600个20Hz控制步；网络仍用完整DDPM100，预测的首个0.2秒目标由固定反馈执行4步再重规划。以原生BDDL目标是否完成评分。有效超时算失败，环境/加载错误算未完成，重启后仍处理原槽。完成回合带哈希保存，禁止覆盖；checkpoint、协议、运行软件、GPU类别及驱动变化会拒绝静默混用。

源数据单位、源基座到LIBERO和TCP的标定仍未完成。本实验评估此前公开约定的共享Cartesian执行方式，不能据接线通过声称物理标定或机器人抓取成功。

## 本次已经做完

- 新评分/续跑逻辑12项合成边界检查通过：错误、超时、重复、篡改、并发和不完整分母。
- 原有真实训练策略32tick排除任务工程结果直接复用，没有重训或重跑旧检查。
- 1086个官方源码/仿真资源文件逐字节绑定；最终包完整性复核通过。
- `libero_original_evaluation_v1.zip`：693932字节；评估代码、固定协议/槽位和已审核seed42终点身份。
- `libero_simulator_assets_8f1084e_v1.zip`：239398314字节；仿真资源及官方代码，不含任何演示数据、模型权重或.git。
- 未提交新集群作业，未运行正式回合，Linux/EGL及集群独立仿真环境尚未验收。

原协议候选保留，新固定协议：`reports/libero_evaluation_protocol_20261026/protocol_frozen.json`，SHA身份`5156ea881351f156b9d95ba756e6b61434f3fe8a44e199bb52dde9b4c40c3381`。

## 下一次操作

将两个ZIP上传到 `/projects/Zeroshot`，再执行：

```bash
(
set -euo pipefail
cd /projects/Zeroshot
unzip -n libero_original_evaluation_v1.zip
cd libero_original_evaluation_v1
bash setup_runtime.sh
bash submit_once.sh 1
)
```

`setup_runtime.sh`只在新的`envs/libero-eval-v1`安装仿真依赖，继承原有Torch；不修改正在用的OXE训练环境。不使用完整原版requirements覆盖旧Torch/Transformers。下载依赖使用无缓存安装。

首段只跑一个**完整且计数**的正式回合，用来检查Linux/EGL并取得真实完整耗时；不是另跑后丢弃的试验。若成功完成，下一段不会重跑该回合。先查看日志和验收结果，再确定每个作业应跑多少回合，当前不根据8决策工程耗时夸大全套耗时。

```bash
JOB=$(cat /projects/Zeroshot/libero_original_evaluation_v1/job_id.txt)
sacct -j "$JOB" --format=JobID,State,ExitCode,Elapsed,NodeList
tail -n 35 "/projects/Zeroshot/logs/libero-s42-${JOB}.out"
/projects/Zeroshot/envs/libero-eval-v1/bin/python /projects/Zeroshot/libero_original_evaluation_v1/verify_output.py --output /projects/Zeroshot/runs/libero-original-dino-noaug-s42
```

后续手动执行`bash submit_once.sh N`，N为本段最多新增的完整回合数，总任务数始终4000，不调整每回合控制预算。没有自动续提。跨作业由原子回合记录续跑；技术中断不会丢弃已完成结果。其他训练条件/种子的checkpoint就绪后各自审核终点身份，使用同一评估协议，不重复基线。

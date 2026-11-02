# DINO同预算对照终点：开发场景出现明显改善

2026-10-25收到222758原日志：训练保存至84168，完成原dev20闭环并输出三组比较。已用用户粘贴的20个DINO最大重叠奖励与本地固定CLIP/原生参考独立复算，均值、满奖励计数、配对增益及预定20000次场景bootstrap区间一致。完整checkpoint/更新记录验证器及下载文件尚待；没有重新训练或重跑场景。

| 固定84168终点 | 平均最大重叠奖励 | 满奖励场景 |
|---|---:|---:|
| 原CLIP+8层Adapter，215999 | 0.445488 | 5/20 |
| 冻结DINO+同8层Adapter，222758 | 0.829858 | 7/20 |
| 原生Diffusion Policy，212158 | 0.845878 | 10/20 |

DINO减CLIP平均增益0.384369，配对场景95%区间[0.200636,0.575308]；12场更好、2场更差、6场相同。按平均奖励衡量，原CLIP到原生差距缩小约96%，新DINO仍低原生0.016021。该数值接近不能证明统计等效；场景区间不包含训练seed的不确定性。

## 可得出的判断

在数据/窗口/更新预算、原八层Adapter/池化/投影及作者扩散后端保持的情况下，替换冻结视觉表示恢复了大量开发表现。这对“冻结视觉表示及其与后续前端接口的匹配是重要瓶颈”提供强支持；也说明保留原八层Adapter的系统可以获得较高闭环奖励。不支持“Adapter必然起反作用”或“CLIP是唯一根因”的断言。

此对照连同骨干改变了原生归一化/最终LayerNorm，图像几何仍采用原224resize/224crop；原CLIP训练A6000、新DINO训练A40，只有一个训练seed，且dev20已有多次开发使用。因此结果不是独立最终测试、完整跨任务泛化、机器人抓取成功，不能自动替换冻结CLIP主模型。

`reference_competence_gate_passed=false`的具体原因是DINO满奖励比例0.35未达既定0.50，平均奖励0.829858已超过0.80。这与观察到相对CLIP的明显收益可以同时成立；不调整满奖励定义或门槛来使结果过关。

## 立即执行

先核验原完整输出，核验成功再打包以下九个小文件；5GB左右完整checkpoint继续留集群，不打包不存在于第二段的`startup_checks.json`。其startup证据已包含在report和latest侧车。

```bash
DINO_RESULT=/projects/Zeroshot/runs/pusht-dino-part2-222758
LD_LIBRARY_PATH=/projects/Zeroshot/envs/official-pusht-v1/lib \
  /projects/Zeroshot/envs/official-pusht-v1/bin/python -B \
  /projects/Zeroshot/pusht_dino_training_v1/verify_dino.py "$DINO_RESULT" &&
tar -czf /projects/Zeroshot/pusht_dino_222758_results.tar.gz \
  -C "$DINO_RESULT" \
  report.json latest.json backbone_binding.json updates.jsonl \
  development.json comparison.json dino_worker.log \
  development_seed_5000000.mp4 development_seed_5000001.mp4
```

本机PowerShell：

```powershell
scp zixiao005@10.97.216.128:/projects/Zeroshot/pusht_dino_222758_results.tar.gz "D:\ntu_related\dissertation\Zero_shot\"
```

收到并核对小输出和固定视频后，可据此向老师汇报该骨干对照。当前不追加训练/预算/变体、不重复本轮实验、不访问final50；后续确认设计及LIBERO主基准仍按老师目标推进，不能把这一个开发结果当完整论文实验已完成。

# 给老师的进度与范围确认稿（未发送）

日期：2026-10-29。以下英文供用户审阅、修改后自行发送。GPU候选尚未执行或获得新预算；邮件没有发送。

Subject: Progress update and scope clarification for OXE-to-LIBERO experiments

Dear Professor Lipo,

Following your latest guidance, I have retained frozen DINOv2 ViT-L/14 as the main visual backbone and CLIP as the visual ablation, with the frozen CLIP text encoder. In the completed Push-T development comparison, mean overlap rewards were 0.84588 for the native Diffusion Policy, 0.44549 for our CLIP frontend, and 0.82986 for our DINO frontend. These results support the backbone choice, but do not establish OXE-to-LIBERO transfer.

Two issues currently prevent me from responsibly expanding the LIBERO evaluation. First, the current four-source recipe has not established accurate decoded actions on the sampled training windows, so the difficulty is not limited to held-out generalization. The latest bounded experiment used 512 training windows, 128 per source, and 4,096 additional updates. On the fixed 32 training anchors and 88 development windows, all 14 available source-level pose error means in each split were worse than both the matched-budget control and the static baseline. I downloaded the raw predictions and independently verified the reported metrics on CPU. This branch is closed without further training.

There is also positive evidence: the earlier Bridge recipe beat the static baseline offline, DINO worked under that earlier recipe, and the current model improved substantially when trained repeatedly on eight fixed Bridge windows. These results show that learning is possible in limited settings, but they do not identify the cause of the current multi-window failure. The full 65,536-update no-augmentation runs completed for two seeds used epsilon prediction; the later x0/current-pose diagnostics were bounded prefixes, not completed full x0 runs.

Second, the action execution contract remains unresolved. Our labels describe future observed end-effector poses at nominal 5 Hz, whereas LIBERO uses native 20 Hz controller commands. On an excluded engineering trajectory, native action replay and a 20 Hz expert-pose execution succeeded, while the current 5 Hz representation and the tested interpolation control did not. Physical units, source-to-target coordinates, TCP conventions and contact timing also require stronger evidence. Accurate source-domain predictions alone would therefore not yet justify the full LIBERO rollout matrix.

Could you advise on three implementation and scope decisions?

1. May I prioritize establishing a single-source Bridge capability baseline under the current action representation before resuming the four-source augmentation matrix? I would label it as a diagnostic, and would not present an improvement over mixed training as proof of interference unless Bridge exposure and optimization were properly controlled.
2. Is the intended 8D output best treated as a future reached-pose trajectory with a separately validated target-robot tracking controller? What target-robot unit, coordinate and TCP calibration, and excluded engineering demonstrations for controller validation, are permissible while preserving the intended zero-shot policy evaluation? I have not used LIBERO demonstrations to train the policy.
3. Under the current approximately 200 GB project storage limit, what training scale and experiment priorities should I retain? The current 128-shard pool is a limited subset of the four sources. The planned 13-model matrix is not complete, and the large LIBERO evaluation has not been completed. Please advise on priorities among the main LIBERO matrix, CLIPort, baseline comparisons, adapter-depth ablations and cross-robot evaluation if the full original scope cannot be completed within the available resources.

I will retain the completed results and keep Push-T as a separate diagnostic chapter. I will also keep the original OXE-to-LIBERO research question explicit rather than substituting target-domain supervised training. I would appreciate your guidance on the above decisions before committing substantial additional compute.

Best regards,
Zixiao

## 用户审阅备注

- 这里请求的是研究范围、单源诊断定位和控制标定边界；没有宣称老师已同意缩减实验或改任务。
- 尚未给老师承诺新训练日期。具体单源候选见`BRIDGE_CAPABILITY_DECISION_DRAFT_20261029.md`：约6小时纯训练、预计6.5–7.5小时占卡、8小时上限，只有一个训练seed；候选未授权执行。
- 暂停批量LIBERO是当前执行安排，不是老师撤销要求。用户原评分计划仍为40任务×每任务100回合/模型；没有完成，邮件中的大评估不是已取得的成功率。
- 233818最新结论依据原数组复核；完整权重/Adam与4096内部轨迹未下载，邮件没有称其逐字节验收。旧Bridge/DINO和Push-T正结果与当前四源失败分开表达。
- 未核实最新集群quota和Slurm终态，邮件不写实际剩余空间、当前作业状态或清理已完成。

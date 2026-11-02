# 主线进度与动作定义问题稿（未发送）

2026-10-30。供用户审阅和自行发送；没有调用邮件工具。请回复老师最新原线程、保留原主题及历史邮件；下面不新设邮件主题。

Dear Professor Lipo,

I am keeping frozen DINOv2 ViT-L/14 as the main backbone and the five LIBERO conditions you specified: four augmentation conditions with three training seeds each, plus combined augmentation with CLIP at one seed. This 13-model benchmark is not complete. The completed no-augmentation epsilon runs for seeds 42 and 43 are retained, but they are not results for the later x0/current-pose recipe or the full augmentation matrix.

Two technical issues currently block meaningful expansion. In a completed 65,536-update Bridge-only x0 run, training path-position error was 25.3% below the static baseline, but path-rotation error was 13.05 degrees versus 10.80 degrees for static. Development rotation was also worse than static, and the confidence interval for the path-position improvement crossed zero. Subsequent bounded diagnostics found unreliable low-noise reconstruction, without establishing a unique cause. I have stopped those branches without extending their budgets.

The execution issue is separate. Our current labels are future observed end-effector poses at nominal 5 Hz, rather than native controller setpoints. On one engineering demonstration excluded from final scoring, native replay and dense 20 Hz expert-pose execution succeeded; the 5 Hz targets, exact gripper timing, and the tested interpolation did not preserve task success. I will not repeat these tests or treat dense expert information as a policy improvement.

Before another training change, I would like to settle the shared action contract. Should I retain future reached-pose trajectories with a separately specified target-robot tracking interface, or is the intended output a source-supported desired controller pose? The latter would require a new label definition and retraining. The contract also needs the coordinate/unit/TCP conventions for both the output and the current-pose input. The latest diagnostic uses a source-normalized 18-dimensional current-pose branch; directly substituting LIBERO world coordinates would not establish the same input semantics. I have not trained the policy on LIBERO demonstrations.

I have closed the language-generation tooling work. The small controlled-template candidates are archived as preparation only; they are neither a complete augmentation bank nor robot benchmark results. The current training pool also remains a limited four-source subset, so I would appreciate your guidance on the intended final scale under the approximately 200 GB project storage constraint. I am not assuming that CLIPort, published-baseline comparisons or the other requested evaluations have been waived.

I have prepared a concise input/output contract decision sheet, and will fix the action definition and one bounded implementation plan before committing additional training compute. I will report progress in terms of which source-capability or execution blocker has actually been removed.

Best regards,
Zixiao

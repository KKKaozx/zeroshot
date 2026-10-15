# Current observed pose pair

Reuses frozen bridge_growth64_v1 and bridge_growth128_v1 inputs, production code,
CLIP and environment. No new data/model downloads or package installs.

Same augmented architecture in both arms: bias-free zero-initialized9-to1024
state projection added to unchanged Adapter/history context. Zero vs current
observed XYZ + rotation columns is the only conditioning intervention. XYZ
statistics use128 training windows only. No future-state arrays are bundled.

Same128 train/40 development windows, initialization43, batch8,8192 updates,
late learning rate1e-4 then1e-5 at4096, native DDPM100. Frozen CLIP and gripper.
Preflight must prove exact zero-port replay and actual real-state gradients.
Formal training requires a successful same-cluster preflight report.

Execution passing does not establish fitting, generalization or robot grasping.
See docs/BRIDGE_STATE_CONDITIONING_PAIR_20261013.md in the project repository for
screening/stopping rules and submission instructions. Keep checkpoints on cluster.

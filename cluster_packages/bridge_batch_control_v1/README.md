# Fixed 32-window batch organization control

See manifest.json for preregistered budgets and limitations. Both arms reuse the project's production Adapter and DDPM. Only pose is trained. Preflight success checks execution, not model quality. Formal fit requires a successful same-package cluster preflight. Development data never enter the optimizer.

No pip/conda install or data/model download occurs. Reuse /projects/Zeroshot/envs/multitask-preflight-v1, bridge_expansion_training_v1/data and baseline_setup/clip-vit-large-patch14-v1. Output contains reports and predictions, without large checkpoints.

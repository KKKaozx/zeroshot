# Frozen paired xyz normalization evaluation

Evaluate both newly trained groups from197220, not the historical193761 weights. Two groups share initialization/order/optimization budgets but use original versus fitted xyz scales. No training, installs, downloads, normalization fitting or checkpoint selection occurs.

Loads final.pt for baseline/axis_limits from /projects/Zeroshot/runs/bridge-xyz-train-197220; checks their SHA256 and every training contract source file. Evaluates all1836 training/40development windows from405/10episodes with three sampling seeds0/1/2. No reserved-test targets used. Main result is native100-step DDPM; one-stepx0 is separate. No oracle best-of-selection.

Outputs are converted to original tool-relative xyz/.10m before pose clipping, gripper conditioning and metric calculation. cm conversion is10*value; quaternion errors account for sign equivalence. Rechecks frozen CLIP, trainable weights, checkpoint files and paired initial/final CUDA RNG streams. Per-task and per-episode reports allow analysis without treating overlap as independent episodes.

Contains the immutable r1 training source files for hash binding; files are copied, not new implementations. New evaluator reuses existing metric/population helpers. Original training folder and checkpoints are not overwritten.

Additional cluster output is about10–20MB (JSON and compressed paired predictions), not extra checkpoints. Local ZIP is small and contains no weights/data. Download report JSON first; leave checkpoints and prediction arrays on cluster unless needed. Two-hour Slurm limit is an allocation limit, not an ETA.

Success means evaluation integrity; predictive improvement must be judged from actual metrics. Offline grasp/release timing refers to label transitions, not observed successful grasps. Horizon endpoint is the16-step window endpoint, not the end of the whole task.

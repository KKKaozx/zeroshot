# Matched epsilon versus x0 diagnostic

Cluster job 186009. Fourteen selected training windows, 224 action targets,
2000 updates per arm. Initial parameters, noisy training inputs, timesteps and
update coverage match. Context and gripper are frozen. Training seed 42;
sampling seeds 1101/1102/1103. These are not three training runs.

| Prediction target | Mean position cm | Mean rotation degrees |
|---|---:|---:|
| epsilon | 1.257 | 3.896 |
| x0 | 0.157 | 0.685 |

The x0 arm improved fitting under this budget. The result does not prove epsilon
is mathematically invalid, generalization improves, or the teacher's complete
architecture is verified.

Source snapshot: `diagnostics/target_pair/`.
Exact metrics, checks, hashes and scope: `reports/bridge_target_pair/`.
Historical cluster submission script: `scripts/cluster/train_diffusion_target_pair.sh`.
Cached inputs and initial/final weights are retained outside Git.

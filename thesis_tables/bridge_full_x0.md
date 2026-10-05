# Full training-set frozen-context x0 diagnostic

Recorded for the progress table, not a final thesis success-rate comparison.

| Item | Recorded value |
|---|---|
| Cluster job | 186047 |
| Scope | Frozen-context pose diffusion-head fit; separate gripper frozen |
| Dataset | BridgeData V2 0.0.1, one `sweep into pile` task |
| Split | 17 train / 3 development / 5 reserved test episodes; 316 / 55 / 69 windows |
| Prediction horizon and representation | 16 targets; tool-relative 3D position, xyzw quaternion, binary gripper |
| Context source | Original two-layer, 256-attention-channel adapter checkpoint |
| Training target | x0 (`diffusion_prediction_type=sample`) |
| Diffusion schedule | 100 steps, cosine `squaredcos_cap_v2` |
| Optimizer budget | 9875 updates, batch 64; exactly 2000 draws per train window |
| Learning rate / weight decay / gradient clipping | 3e-4 / 0 / 1.0 |
| Training initialization seed | 42; one training initialization |
| Sampling seeds | 1101, 1102, 1103; not independent training seeds |
| Selection | Fixed final update; no development checkpoint selection |
| Cluster GPU request | One `a6000` via Slurm script; allocation metadata beyond saved artifacts not re-inferred |
| Runtime | Python 3.10, torch 2.5.1 CUDA 11.8 build, NumPy 1.26.4 |
| Exact exported source | `diagnostics/full_x0/` |
| Config and provenance | `configs/bridge_full_x0/` |
| Numerical evidence | `reports/bridge_full_x0/` |
| Weights and cached inputs | Stored outside Git; original recorded hashes retained in provenance |

| Metric | Train | Development |
|---|---:|---:|
| Mean position error cm | 0.21 | 10.41 |
| Mean rotation error degrees | 0.95 | 17.07 |

These are action errors aggregated by the existing analysis over sampling draws,
not means/SD over independently trained policies. No task-success rate is available.
Development validation failed; test-target metrics remain unreported.

Original submission: `sbatch /projects/Zeroshot/scripts/train_diffusion_full_fit.sh`.
The script invokes the exported `run_full_fit.py` with a fresh output directory.
Source, archived inputs, initial weights and integrity hashes are needed to repeat
the run. This repository alone does not contain the full reproduction bundle.

The publication commit identifies this source snapshot. Earlier source hashes in
the artifact provenance remain the authority for the already executed experiment;
do not replace them with hashes of subsequently edited code.

# Architecture and code map

## Main implementation

| File | Responsibility |
|---|---|
| `code/dataset.py` | Schema-driven parsing, CLIP image preprocessing, action conversion |
| `code/adapter.py` | Visual-query / text-key-value cross-attention, residual FFN and context pooling |
| `code/diffusion_decoder.py` | Conditional temporal 1-D U-Net, timestep and context conditioning |
| `code/models.py` | Frozen CLIP, adapter, selectable heads, noise schedule, training target and reverse sampling |
| `code/train.py` | Episode-group splits, optimization, metrics, checkpoint/configuration recording |
| `code/test.py` | Canonical evaluation entry forwarding to `evaluate_offline.py` |
| `code/evaluate_offline.py` | Offline prediction metrics and controlled input diagnostics |
| `code/eval_closed_loop.py` | Existing PyBullet prototype; benchmark completion not demonstrated |

Default adapter construction supports eight layers and 512 attention channels.
Saved experiment configurations override defaults. Do not infer a checkpoint's
architecture from constructor defaults: the recent contexts came from two layers
and 256 attention channels. CLIP remains frozen; normal joint training optimizes
the adapter and action head. Frozen-context diagnostic runners bypass CLIP and
the adapter and train only the selected head.

## Action contract

`tool_relative_pose_open_positive_v2` uses a 16-target trajectory:

```text
[local_dx, local_dy, local_dz, dqx, dqy, dqz, dqw, gripper]
```

Translation is relative to the input observation's tool frame, divided by
0.10 metres. Quaternion order is xyzw. Gripper supervision is -1 closed / +1 open;
this is a binary command convention, not universally measured gripper width.
For the Bridge diagnostic, input gripper measurement remains continuous and
command supervision uses the explicitly versioned `reverse_scan_v1` policy.
Reaching observation j is paired with command j-1. These checks do not establish
hardware latency, physical calibration or simulation-controller equivalence.

Zero motion requires the identity quaternion [0,0,0,1]. An all-zero eight-vector
is not a valid generic pose/inactivity label. Counterfactual null actions must be
defined against the actual action/control contract before augmentation is used.

The latest DDPM diagnostic diffuses seven pose dimensions and uses a separate
frozen gripper predictor. Its x0 target is named `sample` in the configuration.
100 reverse diffusion steps are computational iterations, not 100 robot moves.

## Diagnostic script index

| Script/group | Scope |
|---|---|
| `test_loader.py` | Loader checks and dataset-free model mathematics |
| `audit_dataset.py`, `audit_bcz.py` | Source semantics, timing, gripper and split audits |
| `diagnose_bridge_modules.py`, `render_bridge_diagnosis.py` | Module evidence and visual report |
| `probe_bridge_heads.py`, `evaluate_bridge_probe.py` | Frozen-context head fitting and evaluation |
| `compare_bridge_adapter_training.py`, `finalize_bridge_adapter_comparison.py` | Matched adapter-training comparison |
| `compare_bridge_current_pose.py`, `compare_bridge_pooling.py` | Input/pooling diagnostic comparisons |
| `compare_bridge_saved_checkpoints.py` | Read-only checkpoint comparison |
| `audit_bridge_input_neighbors.py`, `audit_bridge_patch_features.py` | Feature distinguishability diagnostics |
| `audit_bridge_diffusion.py` | Independent numerical diffusion audit |
| `prepare_bridge_diffusion_fit.py`, `diagnose_bridge_diffusion_fit.py` | Export and verify small-window diffusion fit |
| `prepare_bridge_diffusion_full_fit.py` | Export full frozen-context experiment |
| `export_bridge_official_subset.py`, `stage_training_shards.py` | Subset export and storage staging |

Diagnostic tools are retained to explain existing results; they are not additional
main training pipelines. Many require separately preserved artifacts and fixed
local paths. No script is moved in this update, preserving existing imports.

## Known limits

The non-separate eight-dimensional diffusion masked-loss path has a recorded
defect in the existing audit. The verified seven-dimensional separate-gripper
path does not establish correctness of every optional configuration.
Simulation action mapping and multi-source physical semantics remain incomplete.

A scoped input/output review is in [MODEL_BENCHMARK_ACTION_CONTRACT.md](MODEL_BENCHMARK_ACTION_CONTRACT.md). The current 16 tool-relative targets have no verified mapping to CLIPort world-frame pick/place primitives. The native expert replay control does not validate that mapping or the learned model.

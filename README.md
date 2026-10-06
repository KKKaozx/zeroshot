# Zero-shot robotic manipulation with vision-language models

Dissertation implementation using a frozen CLIP backbone, cross-attention adapter,
and conditional diffusion action decoder. Last progress update: **2026-10-07**.

**Current status: training-set action fitting is supported by diagnostic results;
generalization to held-out demonstrations has not passed. No benchmark rollout or
zero-shot task-success result is claimed.**

## Start here

- [Current progress and module evidence (中文)](docs/PROGRESS.md)
- [Minimal execution protocol and readiness checks (中文)](docs/MINIMAL_EXECUTION_PROTOCOL.md)
- [Architecture, action contract, and code map](docs/ARCHITECTURE.md)
- [Comparison with the original Diffusion Policy implementation (中文)](docs/DIFFUSION_POLICY_COMPARISON.md)
- [Executed DDPM/U-Net numerical and gradient comparison (中文)](docs/DDPM_RUNTIME_COMPARISON.md)
- [Storage snapshot and next-experiment budget](docs/STORAGE_BUDGET.md)
- [Next paired decoder experiment: upload and GPU preflight](docs/CLUSTER_DECODER_PAIR.md)
- [Environment and execution instructions](docs/RUNNING.md)
- [Supervisor requirements and completion status](docs/REQUIREMENTS.md)
- [Dataset versions and diagnostic split](docs/DATASETS.md)
- [Experiment records and evidence](thesis_tables/README.md)

## Architecture

```text
RGB observation -> frozen CLIP vision tokens --+
                                               +-> cross-attention adapter -> context
Language instruction -> frozen CLIP text tokens-+                                |
                                                                                v
Noisy action trajectory + diffusion timestep -> conditional temporal U-Net -> DDPM sampler
                                                                                |
                                                                                v
                                                            predicted robot action trajectory
```

The intended architecture has eight adapter layers with 512 attention channels.
The recent Bridge diagnostics instead use contexts from a **two-layer adapter
with 256 attention channels**. Recent diffusion fitting freezes both those
contexts and the separate gripper head; it is not joint adapter/decoder training.
CLIP ViT-L/14 has 1024-dimensional vision hidden tokens and 768-dimensional text
hidden tokens. The adapter projects these into a common attention dimension.

## Latest diagnostic result

One Bridge task (`sweep into pile`), 25 demonstrations, split by complete episode:
17 training episodes / 316 windows; 3 development episodes / 55 windows;
5 held-out test episodes / 69 windows. Each window has 16 action targets.

| Final x0 diffusion-head fit | Train | Development |
|---|---:|---:|
| Mean position error | 0.21 cm | 10.41 cm |
| Mean rotation error | 0.95 degrees | 17.07 degrees |

This run used 9,875 optimizer updates, batch size 64, and exactly 2,000 draws per
training window. 311/316 training windows passed all three sampling-seed pose
checks. Training mean accuracy does not imply perfect fitting of every action.
Development gripper balanced accuracy is about 67.4%; transition-pair correctness
is 0/13 for open-to-closed and 0/6 for closed-to-open. The gripper head was frozen.

Three **sampling** seeds are not three independent **training** runs. The three
development episodes have been repeatedly inspected and are not a fresh final
test. No held-out test-target metric is reported.

See [the full experiment record](thesis_tables/bridge_full_x0.md) and
[published numerical evidence](reports/bridge_full_x0/).

## Repository layout

```text
code/                 Main implementation and existing diagnostic tools
experiments/          Historical argument files, including diagnostic configurations
configs/              Recorded experiment configuration and provenance
scripts/cluster/      Historical NTU Slurm scripts and author-baseline runners
diagnostics/          Exported standalone diffusion experiment source snapshots
docs/                 Setup, architecture, datasets, progress, and limitations
docs/archive/         Previous README snapshots, kept for traceability
reports/              Selected small numerical reports, not weights or raw data
thesis_tables/        Experiment-to-evidence reproduction records
```

The main training entry is `code/train.py`. The evaluation entry is `code/test.py`,
which forwards to `code/evaluate_offline.py`. Dataset-free mathematical checks:

```bash
python code/test_loader.py --model-math-only
python code/train.py --help
python code/test.py --help
```

Consult [RUNNING.md](docs/RUNNING.md) before launching an experiment. Historical
argument files and cluster scripts contain machine-specific paths and must not
be treated as a portable, complete reproduction package.

## Next milestone

Complete the remaining observation, gripper-state, timing and execution contracts
in the [minimal protocol](docs/MINIMAL_EXECUTION_PROTOCOL.md), then validate a bounded
fixed-period contact control before model integration. Source and numerical
comparisons with published diffusion-policy code are documented above; they did
not resolve held-out Bridge generalization. Formal dataset splits and fair baseline
protocols remain required before additional training or benchmark claims.

LIBERO/CLIPort rollouts, four augmentation conditions, independently trained seeds,
RT-1/CLIPort comparisons, adapter-depth ablations, and cross-embodiment results
remain outstanding. PyBullet rollout code exists, but its presence is not evidence
that the requested benchmark evaluation is complete.

## References and provenance

- [CLIP](https://github.com/openai/CLIP): pretrained vision-language backbone.
- [DDPM paper](https://arxiv.org/abs/2006.11239): diffusion formulation.
- [Diffusion Policy](https://github.com/real-stanford/diffusion_policy): published
  robot action-diffusion reference; no full reproduction is claimed here.
- [BridgeData V2 author code](https://github.com/rail-berkeley/bridge_data_v2):
  loader and separate LCBC diagnostic baseline, pinned to `bc60a35` in historical setup.
- [Open X-Embodiment](https://robotics-transformer-x.github.io/),
  [LIBERO](https://libero-project.github.io/), [CLIPort](https://cliport.github.io/).

The initial repository mentioned VLA-Adapter as a proposed reproduction target.
That initial plan is preserved in the archive; VLA-Adapter is not integrated into
this codebase and must not be presented as a completed baseline.

Datasets, pretrained weights, checkpoints, private correspondence, and credentials
are excluded from Git. Preserve checkpoints and full artifacts separately with
their hashes. Commit substantive progress at least weekly.

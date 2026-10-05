# Environment and execution

Run commands from the repository root. This update does not initiate training.

## Environments

The full local implementation uses PyTorch, Transformers, NumPy, Pillow,
PyBullet, h5py, TensorFlow and TFDS. `requirements.txt` records versions observed
in the local environment; an installation on a clean machine has not been
validated. Install a hardware-appropriate PyTorch build first. Use a project-local
environment and configure caches outside restricted home/temp storage as needed.

```bash
conda env create -f environment.yml
conda activate zeroshot
# Install a PyTorch build appropriate for your GPU/runtime before the next command.
python -m pip install -r requirements.txt
python -m pip check
```

The successful frozen-context cluster diffusion runs used a separate environment:
Python 3.10, PyTorch 2.5.1 with CUDA 11.8 wheels, NumPy 1.26.4. They do not need
CLIP or TensorFlow because the visual/language contexts are already cached.
The separate author Bridge baseline uses JAX 0.4.13 and its own dependencies;
do not merge all three environments without compatibility checks.

## Safe entry checks (no dataset or training)

```bash
python code/test_loader.py --model-math-only
python code/train.py --help
python code/test.py --help
```

## Historical full Bridge regression training

```bash
python code/train.py @experiments/bridge_single_task_full_validation.args
```

This experiment is already completed. The argument chain contains local dataset,
cache, plan and output paths. Inspect all referenced `.args` files before reuse;
choose a new output directory and verify the frozen episode split. It trains a
regression configuration, not the later standalone frozen-context x0 run.

## Offline evaluation

```bash
python code/test.py --checkpoint /path/to/checkpoint.pt --split validation --cache-dir /path/to/hf_cache --batch-size 8 --max-batches 7 --samples-per-batch 8 --output-json /path/to/new-report.json
```

This explicitly requests development validation. The evaluator's default split is
test; do not rely on the default during development. Check the saved coverage:
7 batches of at most 8 samples cover 55 windows only when the restored split is
the expected one. Small default batch/sample limits are partial evaluations.
Standalone diagnostic head checkpoints have a different format and are not
automatically compatible with the main evaluator.

## Reproducing the latest frozen-context x0 diagnostic

The exact executed source snapshot is under `diagnostics/full_x0/` with recorded
configuration in `configs/bridge_full_x0/`. It depends on separately retained
`full_context.pt`, `shared_initial_weights.pt`, and the export integrity metadata.
Full reproduction is unavailable from source alone because those files are not
published in Git. Preserve and verify their hashes before using the runner.

Historical Slurm scripts are in `scripts/cluster/`; they assume `/projects/Zeroshot`,
account/QOS `msc`, partition `cluster02`, the named environments and export ZIPs.
They also require the pre-existing `scripts/ssd_cache.sh`. They are records of the
executed setup, not portable automatic installers. A Slurm-submitted job should
run GPU work on an allocated compute node.

## Limitations

No new dataset download, training run, simulation rollout or baseline reproduction
is performed by this repository update. CLI checks and mathematical self-checks
do not establish physical robot success or a clean-environment reproduction.

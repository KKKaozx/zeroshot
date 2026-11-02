# Push-T existing-asset inventory (CPU only)

This is a feasibility check for a possible matched native-frontend versus
frozen-CLIP + project-Adapter comparison. It does not implement or authorize training.

Uses the existing `official-pusht-v1` Python on a **CPU Slurm node (16 GB, 15 min)**.
The trusted, previously downloaded 4.3 GB official checkpoint is loaded on CPU to
read saved progress, optimizer counters, configuration, and conditional-weight
shapes. No policy or dataset constructor, model forward, rendering, GPU, pip,
conda, network, training, or checkpoint modification is used. CLIP weights are
not loaded. Source and environment directories are not modified.

Training Zarr search is limited to the checkpoint-configured path relative to
the source, official task root, and project root, plus corresponding `.zip`
paths. A manually supplied `--dataset /absolute/path` can inspect a different
known location. No recursive disk search. Only array metadata and small
`meta/episode_ends` values are read, never images/actions/states. Structural
consistency **does not prove all frame chunks exist or that data is canonical**.
Window counts reproduce the saved author's recipe; they do not define a new
split. The author's validation complement includes episodes excluded by its
train downsampling. The author normalizer used the whole replay; any new fair
protocol must explicitly give both groups training-only statistics.

Distribution metadata from both existing Python environments is collected
without importing torch/transformers in the secondary environment. This cannot
establish runtime compatibility. Actual GPU resource/time measurement is still
pending. The previous checkpoint SHA record is copied with
`sha_freshly_verified=false`; the 4.3 GB file is not hashed again by this job.

New output only: `/projects/Zeroshot/runs/pusht-module-assets-JOBID/report.json`.
An existing output directory is rejected. An inventory exit of 0 is not a
readiness, improvement, or generalization verdict. No formal job is submitted.
207558, all frozen assets, and sealed Bridge final25 remain untouched.

After uploading this directory to `/projects/Zeroshot/pusht_module_assets_v1`:

```bash
PUSHT_ASSETS=$(sbatch --parsable /projects/Zeroshot/pusht_module_assets_v1/inventory.sh)
echo "Push-T资产盘点：$PUSHT_ASSETS"
```

Do not execute the checkpoint inspector directly on the login node.

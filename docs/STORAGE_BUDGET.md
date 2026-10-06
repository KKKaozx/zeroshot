# Storage budget for the next decoder comparison

Snapshot: 2026-10-06. Local capacity checks are read-only; no old artifacts deleted.

User clarification: the constraint is **approximately 30 GB free on the local
disk**, not a request to inspect cluster quota. Treat this reported figure as the
planning limit; the earlier measured snapshot below is informational.
See [the local archive-candidate review](../reports/LOCAL_STORAGE_REVIEW.md).

| Location | Observed free space / quota |
|---|---:|
| C: | Approximately 17.1 GiB free |
| D: | Approximately 38.4 GiB free |
| E: | Approximately 161.8 GiB free |
| Cluster SSD project | 200 GB assigned; current usage must be checked before submission |

The local project `results/` occupies approximately 29.2 GiB and `training_cache/`
approximately 5.7 GiB. Historical checkpoints remain preserved. Neither `df`
filesystem free space nor local disk space establishes the cluster user's quota
remaining; use the actual assigned project and its current usage.

## Next experiment constraints

- Reuse the existing frozen-context inputs and fixed split; do not download or copy
  complete datasets, CLIP weights, or raw images for this decoder comparison.
- Put new caches/environments under the D: project, not C: defaults. Cluster caches
  remain under the assigned SSD project, following its existing cache setup.
- Run the next training comparison on the cluster. Download only small reports
  and selected figures to the local computer, with a **100 MiB local budget** per
  comparison. Do not automatically download checkpoints or complete result bundles.
- Preflight cluster output estimates with a target cap of **6 GiB** per comparison.
  Do not submit if the estimate exceeds the cap or leaves insufficient free space.
- Save final decoder/head weights, configuration, source/input fingerprints and
  small metric reports. Do not repeatedly serialize frozen CLIP or full image data.
- If resumable training is needed, retain at most one active resumable checkpoint,
  explicitly count optimizer state, and check peak temporary-write space.
- Preserve existing results; identify redundant files separately before deciding
  any deletion. No checkpoint pruning is authorized by this capacity check.

For the runtime-tested author network (277,105,287 parameters), float32 weights
alone require about **1.03 GiB**. A conventional Adam checkpoint with those weights
and two float32 moment tensors requires about **3.10 GiB**, before metadata and
temporary writes. These are estimates, not new files or measured training memory.
GPU memory, gradients and activations require a separate budget.

The model-size difference does not require downloading a pretrained 277M-parameter
checkpoint: the decoder can be initialized from its source implementation.

## Optional cluster check before eventual training

```bash
du -sh /projects/Zeroshot/ /projects/Zeroshot/envs/ /projects/Zeroshot/.tmp/ /projects/Zeroshot/runs/
```

The first total already includes the subdirectories; do not sum all four results.
The trailing slash resolves the project-directory link for the total. A missing
`runs/` directory can be reported as absent rather than created merely for this check.
Current cluster usage has not yet been independently obtained in this session.
The user clarified that this check is not needed for the present local inventory.

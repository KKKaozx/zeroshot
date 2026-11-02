# Frozen-CLIP frontend boundary and resource preflight

This GPU job is an interface/resource check, not a performance experiment. No
package installation/download, original environment/source/checkpoint edit, or
automatic formal training. Bridge207558 and final25 remain untouched.

Stages (separate processes on one GPU, so old and new Torch are never mixed):

1. Official environment: select 64 training windows from the confirmed 90-episode,
   10,726-window pool. Read just their two real observations and 16 actions; numeric
   normalization-fit inputs use every frame of those 90 train episodes only.
2. Existing project environment: load local CLIP, hard-freeze every parameter and
   use eval/inference mode. Cache float32 visual/text tokens for the 128 observation
   appearances. Verify full parameter-state hashes before/after. CLIPImageProcessor
   is frontend-specific preprocessing, distinct from native random84/fixed84 crops.
3. Official environment: original policy, full original [512,1024,2048] U-Net,
   epsilon/cosine100/DDPM100, native encoder; five real training updates with EMA.
4. Official environment: reinitialize exactly the same original backend, replace
   only observation encoder with pinned project eight-layer Adapter + Linear1024->64.
   Preserve native normalized 2D agent position in its actual original feature
   columns, determined by intervention rather than assumed order. Five updates/EMA.
5. Verify shared initial backend and normalizer fingerprints, frozen CLIP, gradient
   flow, update timing and memory. One EMA sampling call per group (batch2,100 DDPM
   steps) checks shapes/finite output and timing; no environment rollout or score.

Both use effective batch64 via microbatch8*accumulation8, FP32 and AdamW. Two warmup
and three measured updates; same 64 windows repeated for timing. No checkpoint or
claim of fitting/generalization. Diffusion random draws are not paired across the
different frontends in this timing probe, so its losses cannot be used to compare
effectiveness. The module comparison includes the frontend's preprocessing and
feature mapping, not the Adapter alone.

Adapter executable source is unchanged; a future-annotations header allows its
Tensor|None annotations to load on Python3.9. CPU tests prove same state/output on
small instances. Actual original Torch1.12 runtime is checked by this job, not
assumed compatible. No new dependencies or API fallbacks on failure.

Cached frozen features allow both training groups to run in the working official
environment. A future closed-loop project policy still needs live-CLIP IPC plus
cache/live feature identity checks; this job does not pretend that path is ready.

Resource projection reports 84,168 updates each as a reference scale near the
saved author checkpoint, not a finalized or auto-authorized training budget. It
excludes full cache preparation, checkpoint/logging, rollouts and live IPC costs.
The original 0.9309 pretrained score is never a short-budget comparison target.
Training-only statistics deliberately differ from the old full-replay recipe.

Resource: one A6000, CPU4,22GB host memory,45min maximum. Actual values remain
unknown until the job. Writes only runs/pusht-front-resource-JOBID (~150MB caches
and small JSON reports). No whole-dataset token cache. Full cache size is estimated
from actual training frame count at FP32, not silently changed to FP16.

Final output report.json (only if every stage succeeds); any stage failure writes
STAGE_error.json and stops without automatic installation, budget growth or rerun.

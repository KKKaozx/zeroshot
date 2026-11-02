# Bounded sampling-policy comparison: same source model and four-source pool

User selected4096updates per arm, roughly45minutes total rather than a full new training run. Estimate excludes queue time and may increase with fixed77token training; one-hour Slurm cap and committed halfway states bound cost. No task is submitted by the agent. No new model module, source removal, downsampling, training seed, LR tail experiment or LIBERO use is included.

Both roles restore job232587 real_pose10240: learned coreb53adab6…, pose porta3ca9225…, source contract079e989b…, frozen91175b…, all three original AdamW moment groups and RNG. This is NOT the8window memorization policy233426. DINOv2-L/14 and CLIP text remain frozen, current CLS readout, native x0 DDPM100, measured pose input, source-coordinate labels/masks, gripper joint loss/dropout/clipping and original65536LR horizon remain unchanged.

- `replacement`: actual retained source plan's next4096batches, matching the archived original full-run suffix10241…14336.
- `cycle`: same source in each individual slot and same batch length, but select within each source using shuffled eligible-ID cycles without replacement. Reset each source queue at the intervention start; do not pretend the common10240prefix used this policy. All145812eligible training windows remain the pool. No validation label/prediction selects the schedule.
- Both add4096updates, ending at source LR clock14336. LRs come from archived actual Linux history; no Windows cosine regeneration. This short comparison does not complete the full65536schedule or guarantee full-pool coverage.

Both roles have identical average additional presentations per source window. The intervention tests distribution/coverage, not increasing average practice count or lowering learning rates. In this suffix, Bridge unique-window coverage is planned at55.5%versus80.1%; BC-Z coverage differs only8.10%versus8.44%. A weak BC-Z effect cannot refute a full-budget coverage hypothesis.

Training text is padded to77tokens in BOTH roles, preserving word IDs/availability masks while keeping dropout operation shapes constant. This is an explicitly shared training invocation change relative to earlier dynamic-padding runs; the matched causal comparison is between these two new arms only. Evaluation preserves original120-window padding/encoding/batch conventions. Before training, replay all120original232587predictions for seeds1101/1102/1103 exactly. Probe the two real first batches from identical source RNG with zero optimizer updates; primitive noise/time and post-backward RNG must match, weights and Adam must stay unchanged. During all4096steps, candidate actual noise/time and before/after model RNG are checked against control. Different targets imply different noisy-action hashes and gradients: those are deliberately not required to equal. Fixed padding alone is not assumed sufficient; actual runtime traces must pass.

Before sbatch, CPU-only admission reads the ACTUAL source checkpoint with mmap, verifies learned/port/contract identities, one-GPU RNG metadata/runtime identity, all4096LRvalues, and the next32768window indices directly from the retained source plan's current order. It checks immutable target manifests/source codes/all batch-mask sums and cycle uniqueness. Existing corrected source loader bytes are reused without depending on failed233339logs or doing another GPU diagnosis. No guard is removed or tolerance relaxed.

Paired endpoint generation uses the same published120windows:32seen-training and88development, with Bridge64groups primary, the same3sampling seeds and native100DDPM steps. It reports replacement/cycle/source10240regression reference/static. Source10240is not an equal-budget causal arm. Sampling-effect gates, basic-static-competence gates and all-source regression flags are separate; none triggers automatic main-model replacement, extra budget, seed or final/test/LIBERO access. One starting checkpoint and reused development groups do not establish training-seed robustness or full generalization.

Full base/port/Adam/RNG/trace checkpoints are saved at each2048boundary with an explicit intervention format/counter and plan/protocol hashes; original source plan is never overwritten. Reserve6GiB for two latest states and atomic replacement overhead. No images/encoder caches/frozen models are copied to disk. A manual `--resume` can recover the same output directory/counter after interruption under the same qualified environment, recording all training job IDs; no automatic retry or extra updates. Completed roles are not retrained to repair evaluation. Multi-job resume is reported, not claimed to be one allocation.

```bash
cd /projects/Zeroshot
unzip -n oxe_sampling_pair_v1.zip
cd oxe_sampling_pair_v1
bash submit_once.sh
```

```bash
JOB=$(cat /projects/Zeroshot/oxe_sampling_pair_v1/job_id.txt)
tail -n 60 -f "/projects/Zeroshot/logs/oxe-sampling-pair-${JOB}.out"
```

Read-only saved-output verification after completion:

```bash
ROOT=/projects/Zeroshot
JOB=$(cat "$ROOT/oxe_sampling_pair_v1/job_id.txt")
"$ROOT/envs/multitask-preflight-v1/bin/python" -B "$ROOT/oxe_sampling_pair_v1/sampling_verify.py" --root "$ROOT" "$ROOT/runs/oxe-sampling-pair-${JOB}"
```

Local tests exercise native CPU state/checkpoint/control flow with toy encoders/decoder and real numeric targets, not actual DINO/CUDA performance. Actual source replay and stochastic matching remain mandatory on the cluster. Prior closed experiments, original files and the thesis shared-protocol/13condition plan remain preserved; this is a separately declared development diagnostic.

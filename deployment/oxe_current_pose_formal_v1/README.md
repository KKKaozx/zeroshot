# Current observed pose: bounded continuation pair

Purpose: test whether supplying the currently observed XYZ/orientation helps the existing x0 policy. This is a source-generation diagnostic, not a full training or LIBERO run.

Both roles restore the same sample/x0 checkpoint from job 231297 at update 8192, including original Adam moments, batch plan, 65536-step learning-rate schedule and RNG. They continue exactly 2048 updates to 10240 on the same A40 allocation, same batches, supervision masks and actual diffusion noise. Frozen DINOv2 and CLIP text, target labels, original adapter/denoiser/gripper modules remain as admitted. The qualified preflight 232371 is read and pinned; its disposable learned weights are never reused.

- `real_pose`: observed current XYZ plus rotation6D, with availability masks.
- `mask_control`: zero values with identical availability masks, identical capacity and parameter count.
- Both add a zero-initialized bias-free Linear(18,1024), 18432 parameters, to the context and a cold third optimizer group. The first two original optimizer groups retain Adam state. Global clipping includes the new port in both roles.
- Statistics come only from 145812 training-window current rows. No future state enters the input. Language Table supplies XY only, with absent Z/rotation zeroed and masked. Recorded coordinates retain source units; no physical frame or unit calibration is claimed.

The exact first two updates must reproduce the passed preflight, apart from wall time. New formal checkpoints have their own format; they do not impersonate native or engineering checkpoints. Full base/port/Adam/plan/schedule/RNG state is saved at 9216 and 10240, replacing only this run's latest checkpoint atomically. The source file is never overwritten. Reserve 6 GiB for this pair and atomic-save overhead. No cleanup is performed.

Endpoint evaluation reuses the fixed 120 windows (32 seen training, 88 validation), sampling seeds 1101/1102/1103, 100 sampling steps and batch size 8. The matched contrast is real_pose versus mask_control at 10240. Archived x0 at 8192 is a regression reference, not an equal-budget causal control. Static is retained. Bridge64 is primary; all four sources and both partitions are reported. Group bootstrap uncertainty excludes training-seed uncertainty. Gates separate evidence of an input effect from beating static; no gate triggers automatic training, model replacement, new seeds or LIBERO evaluation.

Resource estimate from actual preflight: 21.71 minutes of paired updates, excluding archival replay, final generation, initialization, checkpoints and I/O. Job wall limit is one hour; queue time is separate. No completed performance result exists for this package.

## Submit on cluster

```bash
cd /projects/Zeroshot
unzip -n oxe_current_pose_formal_v1.zip
cd oxe_current_pose_formal_v1
bash submit_once.sh
```

Requires the existing `oxe_current_pose_preflight_v1`, original `oxe_target_pair_short_v1`, 231297 source full checkpoint, 232091 original evaluation artifacts, 232371 actual preflight report, raw data, immutable target stores and existing model assets/environment. No model or dataset bytes are bundled, downloaded or installed.

```bash
JOB=$(cat /projects/Zeroshot/oxe_current_pose_formal_v1/job_id.txt)
tail -n 60 -f "/projects/Zeroshot/logs/oxe-pose-pair-${JOB}.out"
```

After completion, verify without new model inference:

```bash
ROOT=/projects/Zeroshot
JOB=$(cat "$ROOT/oxe_current_pose_formal_v1/job_id.txt")
"$ROOT/envs/multitask-preflight-v1/bin/python" -B "$ROOT/oxe_current_pose_formal_v1/verify_formal.py" --root "$ROOT" --with-evaluation "$ROOT/runs/oxe-pose-pair-${JOB}"
```

There is no automatic retry/resume. An interrupted pair retains its last committed formal state. Explicit `run_formal.py --resume --output <same directory>` is available only inside a properly allocated A40 job with the same environment; it records every training job ID and checks prefix provenance. If evaluation alone fails, use `evaluate_formal.py` in an allocated job after preserving/renaming the incomplete evaluation directory. Do not retrain completed roles to repair an evaluation.

Local validation uses CPU toy models and encoder/sampler stand-ins to exercise checkpoint and evaluation control flow. Actual frozen encoders, CUDA kernels and real task performance are qualified on the cluster; CPU checks do not establish these.

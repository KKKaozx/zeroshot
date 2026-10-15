# Matched xyz normalization control

Prepared, not executed. No package installation or downloads. Existing A40-compatible project environment and frozen Bridge pack are reused.

First run preflight.sh: baseline and axis_limits each execute six temporary real updates, discard resulting weights, check shared initialization, all trainable initialization, frozen CLIP, data order, gradient paths and finite losses. This is an engineering preflight, not a quality result.

Only after passing, train.sh accepts XYZ_PREFLIGHT=<job ID> and verifies the unchanged bundle contract. Both groups start fresh: same 1836 training windows, 405 episodes, 20 epochs, batch2, 18360 updates per group, seed42, AdamW1e-4/weight decay1e-4, cosine per epoch, gradient norm1.0, raw weights without EMA. Forty development windows are only indexed; validation targets are not materialized in training or normalization fitting. Reserved tests are unused.

Baseline uses identity normalization with exact original training/sample method calls. Axis_limits uses frozen training xyz scale/offset from job196693. Quaternion, gripper labels and physical-unit gripper trajectory conditions remain unchanged. Returning inverse-transformed predicted x0 and targets to original policy_loss implements xyz 1/scale² loss compensation. Diffusion operates in transformed coordinates; every clean estimate and final output uses the same original physical xyz limits ±30cm; intermediate quaternion clipping and final unit normalization are preserved.

This tests the position normalization recipe, including relative action/noise geometry and offset. It is not a complete official Push-T replication and cannot uniquely identify a module bug. Full 100-step generation is the primary eventual evaluation; one-step x0 is a separate diagnostic. Final evaluation must compare both new checkpoints on all 1836 training and40 development windows with matched sampling seeds0/1/2, original physical units, path/endpoints, per-episode statistics and gripper events. No evaluation is launched by this bundle.

Local bundle stays small and contains no data/weights. Preflight saves no weights. Two trained checkpoints add about1.1GB on cluster only. Each final.pt excludes frozen CLIP. Time limits15min/3h are scheduling limits, not runtime predictions. Keep original checkpoints.

Upload ZIP and checksum; run sha256sum -c, unzip -n, then sbatch bridge_xyz_normalization_v1/preflight.sh. Do not submit full training until paired real-data preflight is checked.

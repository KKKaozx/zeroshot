# Current multiscale prediction-target control

Prepared; not run on cluster. Follow-up to direct fixed-commit official/project U-Net and loss comparison, not a full Push-T reproduction. Previous193195 epsilon experiment used the compact network; this pair uses unchanged current multiscale[256,512,1024], kernel5, time128, original CLIP/Adapter and separate state gripper.

Only intended change is sample(x0) versus epsilon supervision, with matching native100-step DDPM interpretation. Preserve original xyz/.10m, quaternion/gripper targets, image/text/current-gripper inputs, clipping bounds, architecture, optimizer, initialization and budget. No EMA or new normalization. Two fresh training groups, one seed42, full1836train windows/405episodes, batch2,20epochs,18360updates each. AdamWlr1e-4/wd1e-4/defaultbetas, cosine each epoch, gradient norm1.0. Validation only identity-indexed, not materialized for fitting; reserved tests unused.

CPU checks use core model diffusion_loss/sample with a small actual multiscale network and no CLIP load: same timesteps/noisy inputs/RNG, correct targets, finite gradients and equivalent known-clean100-step outputs. These checks do not measure predictive quality.

preflight.sh then executes six temporary real updates in each group and discards weights. Pair checks all-trainable/shared initialization, frozen CLIP, training order and hashes, plus actual denoiser noisy-input/time streams. Different learned contexts after updates are expected and not forced equal. Loss values are not comparable across objectives.

After preflight passes, train.sh requires TARGET_PREFLIGHT and binds the identical source contract. Saves final.pt/report/history per group. About1.1GB new cluster checkpoint storage, no local weights or new downloads. Old checkpoints preserved. The15min/3h Slurm time limits are allocation limits, not ETA estimates.

Full read-only evaluation must follow after training: same complete1836train/40development windows, paired sampling seeds0/1/2, native100-step DDPM, original physical-unit path/endpoints, per-episode/task metrics, gripper balance and event timing. No oracle selection or denoising-loss verdict. Evaluation is not automatically launched by this bundle.

The outcome applies to this prediction-target recipe and fixed budget; it cannot uniquely prove a module bug. Do not simultaneously change network width, conditioning state/history, normalization or optimization settings.

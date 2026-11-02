# Matched train-time visual variation control

One new arm compared with the retained202278 real-state control. Same128 train/40 development windows, original8-layerAdapter, multiscale x0-DDPM100, trainable parameter sets and initialization43. All query visual token sequences are replaced at both training and evaluation by the same mean of128 original training token sequences. Original language and current observed-state inputs remain available. No future inputs or supervised action prior is added.

Purpose: test the net generalization effect of visual-input variation while retaining diffusion. This is a diagnostic input ablation, not a final vision-free robot policy. The state-only ridge6.28cm is a stronger auxiliary position reference, not the output of this neural network. A benefit would not uniquely identify CLIP or Adapter as faulty.

The original raw reader, labels, original real-state scaling, frozenCLIP/gripper and nativeDDPM100 stay unchanged. Training statistics for the constant come only from128 train rows. One8192-update final-only arm; batch8,4096 at1e-4 then4096 at1e-5,512 draws/window. No early selection, extra seeds, changes in mean strength, new normalization or fitting-based extension. Report all40 development windows, all10 episodes, position/rotation, static and stronger auxiliary baseline.

Reuse the original frozen runner via explicit source-count guards; archive its effective source. Reuse the existing shared token cache; no dataset/model downloads or installations. One final nonCLIP checkpoint includes the constant257x1024 token tensor (about1MiB), needed for reconstruction. No optimizer checkpoint or intermediate weight collection. DefaultA6000; current availability is unknown.

Preflight performs4 common,4 constant-LR,4 restored-same-LR replay and4 reduced-LR operational updates; only four fixed training windows are sampled for execution checks, no development score. Execution passing is not a generalization improvement. First submit preflight.sh, then verify report before fit.sh with VISUAL_PREFLIGHT_JOB. Do not rerun the old full-visual control.

The original40-window development set has been reused repeatedly. This is exploratory same-task/new-episode evidence, not an untouched test, unseen-task generalization or grasp success. Gripper remains frozen.

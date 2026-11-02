# Bridge internal checkpoint-selection control

This is one fresh direct x0/DDPM100 trajectory, not a continuation of the closed residual-prior trial.

- 48 optimization episodes/96 windows, 16 inner selection episodes/32 windows, original10 outer development episodes/40 windows. All partitions isolate complete episodes.
- Fresh initialization43; original frozen CLIP/8layerAdapter/multiscale decoder/history/current-state ports. Only actual96 optimization windows fit observed-state scaling. Original fixed0.1m action units are retained.
- One6144-update run, batch8,512 exposures/window;3072 updates at1e-4 then3072 at1e-5. No extra fitting gate or budget extension.
- NativeDDPM100,3 fixed sampling seeds; select every384 updates by the worse of episode-equal position/static and rotation/static ratios. Ties retain the earlier checkpoint.
- Outer40 are scored for selected and final weights only after writing selection_lock.json. Outer labels are decoded for existing input identity checks, never for optimization, scaling or selection.
- Frozen encoder cache can be reused from bridge_state_prior_v1/full_tokens.pt. No residual-prior code, weights, targets or coefficients are used.
- Formal storage: best.pt+final.pt combined<1200MiB; small16 inner prediction arrays and4 final scoring arrays; no optimizer checkpoints. Existing token cache<256MiB. No package installation/data/model downloads.
- Preflight:8 reference updates without monitoring, then the same8 with selection at4/8. Parameters, Adam and losses must exactly agree. Saved best weights must replay original inner predictions exactly, then the transient preflight weight is removed. No outer scores in preflight.

The same-run selected-vs-final comparison isolates checkpoint choice. Old128-window results have a different actual optimization set and are background only. The outer40 have been reused for development; this is not an untouched test or unseen-task zero-shot result. Gripper is frozen; pose errors do not establish grasp success.

Run preflight.sh first. fit.sh requires SELECTION_PREFLIGHT_JOB and independently checks the successful preflight report and current code/input/statistics fingerprints. SHA256SUMS verifies transferred files. Use the project documentation for upload commands and submit each stage only once.

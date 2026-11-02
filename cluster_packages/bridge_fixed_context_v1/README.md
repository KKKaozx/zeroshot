# One resumed806-pool context-update intervention

Objective: reduce independent-demonstration development error. Use the already completed joint continuation208586 as control; do not repeat its8192 prefix or full7978-step control branch.

Resume `/projects/Zeroshot/runs/bridge-budget-fit-208586/training_state_u08192.pt`, including Adam. Missing file or any hash/replay mismatch stops the experiment; no weights-only restart, optimizer reset, tolerance relaxation or automatic reconstruction.

“Fixed context” means fixed Adapter/history/state parameters. Keep original context train mode/dropout and global clipping. Compute context gradients to preserve clipping, then set these gradients toNone before Adam.step. Context parameters, buffers and optimizer moments/step counters must remain unchanged. Only diffusion_decoder receives parameter/Adam updates. This avoids adding an eval-context-vs-train-context change to the intervention.

Preflight: real checkpoint restore, old4 prediction replay,4 joint updates checked against existing saved logs, restore,4 masked updates, restore and repeat those4 exactly. Total12 operational updates, no new candidate development scoring. Formal: restore8192 directly and perform exactly7978 updates to16170. Evaluate all3624 training and40 development windows once at endpoint. Training fit is diagnostic, not admission criterion. No new dataset, image encoding, augmentation, gripper training, or reserved test.

Run preflight.sh once. After successful completion, validate the FULL report with:

```bash
/projects/Zeroshot/envs/multitask-preflight-v1/bin/python /projects/Zeroshot/bridge_fixed_context_v1/validate_preflight.py /projects/Zeroshot/runs/bridge-fixed-preflight-JOBID
```

Only after validation, formal launch requires `FIXED_PREFLIGHT_JOB=JOBID`. The formal script runs the same validator again. Do not pre-submit it now. Download only reports, snapshots, update log, replay NPZ and final train/development NPZ; leave PT and CLIP/token caches on cluster.

Acceptance: compare window-weighted and episode-equal path position/rotation against208586; all4 no worse and at least1 strictly better for overall development gain. Report all10 episodes/5 tasks, endpoints, original8192 and static references. No cherry-picking, further budgets, seed sweeps or automatic final-test use. A gain implicates context update dynamics in this finite setting, not a unique code bug; failure closes this branch.

The local32-parameter CPU probe and AST checks verify update masking, Adam skipping/restoration and source preservation. They do not establish actual model replay, resource usage or generalization. The Slurm preflight is necessary for those operational checks.

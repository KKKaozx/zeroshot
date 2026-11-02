# Current OXE fixed-eight generation learnability check

Existing success is preserved: CLIP fixed-scene PyBullet grasping succeeded, and the matched DINO-only Bridge comparison229116 improved the old recipe. This check targets the later current OXE implementation and its labels, not whether CLIP, DINO or diffusion can work in general. It does not repeat an old backbone comparison or train on LIBERO.

Source: completed232587 real_pose endpoint at10240, with frozen DINOv2-L/14, frozen CLIP text, current adapter/readout, multiscale x0 DDPM100, observed18D pose port, masked source-coordinate targets and joint pose/gripper loss unchanged. Restore all three AdamW groups including their moments and RNG. The source file is never rewritten.

Only the same eight already published Bridge seen-training windows from the fixed120-window selection are trained. They have eight different groups and are selected without new performance screening. Previously they had only1–5exposures each; six were not drawn during the last2048updates. Now repeat exactly this ordered batch8 for512updates: each window gains512presentations. Original total LR horizon65536 continues at positions10240…10751; no restart or learning-rate search. This is an explicit new fixed-window diagnostic schedule, not continuation of the original source-balanced batch plan. It cannot enter the main experiment table as a new full-training result.

Cache frozen vision/text tokens only; adapter, pose port and decoder continue learning. Retain original batch8 training token padding and batch4 encoding/full120text-padding evaluation conventions separately. Before training, replay these first8source predictions for the same three seeds1101/1102/1103 bit-exactly; then compare native encoder training with cached training using the same source RNG, exact losses/all learned gradients/actual noise/RNG, with zero optimizer updates. Abort on disagreement. No disposable gradient probe weights or optimizer changes are carried into training.

Exactly one endpoint after512updates is evaluated using complete native100-step DDPM, not clean-target-fed denoising. No intermediate sample-based selection or early success checkpoint. Training dropout, targets, masks, gripper loss and clipping remain original. Frozen source features are derived from actual images/text once, not from future actions. The existing120text metadata are used only to reproduce original evaluation padding; no validation row is trained or scored here.

Predeclared pose-fit gate: four mean path/endpoint position/rotation errors at most10%of corresponding static errors, with floors0.001normalized position and0.1degree; at least7of8windows satisfy all four errors at most20%of their static errors, with floors0.002position and0.2degree. Floors are numerical diagnostic tolerances, not physical-unit calibration. These thresholds express a strong training-fit check; neither outcome proves task success, grasping, broad generalization or mathematical learnability/unlearnability. A failed512budget does not authorize extra updates or a variant. No final/test/LIBERO data are accessed.

Expected GPU work is a few minutes using cached frozen features; exact cache-path timing is not yet measured on CUDA. Updates and generation have a600second software cap checked during the update loop; Slurm has a20minute total cap. Queue time is separate. Stop on nonfinite values, failed identity/cache/replay guards, or the fixed endpoint; no automatic retry/resume, extra seeds/budget or variants. Keep one diagnostic policy (~0.5GB), small predictions/report, no encoder/cache copy or extra full Adam checkpoint. Reserve2GiB for safe output. Source full checkpoint remains available.

Dependencies: existing passed preflight232371, original target-pair231297 bundle/source, formal232587 bundle/full real_pose checkpoint/report/predictions, existing assets/data/target stores and environment. No new install, download, remote access or submission by the agent.

```bash
cd /projects/Zeroshot
unzip -n oxe_fixed8_learnability_v1.zip
cd oxe_fixed8_learnability_v1
bash submit_once.sh
```

```bash
JOB=$(cat /projects/Zeroshot/oxe_fixed8_learnability_v1/job_id.txt)
tail -n 60 -f "/projects/Zeroshot/logs/oxe-fixed8-${JOB}.out"
```

After completion, read-only verification:

```bash
ROOT=/projects/Zeroshot
JOB=$(cat "$ROOT/oxe_fixed8_learnability_v1/job_id.txt")
"$ROOT/envs/multitask-preflight-v1/bin/python" -B "$ROOT/oxe_fixed8_learnability_v1/verify_output.py" --root "$ROOT" "$ROOT/runs/oxe-fixed8-${JOB}"
```

CPU tests use toy models and stand-in encoders/samplers for actual runner/cache/gradient/save/verification control flow. They do not replace actual CUDA qualification or constitute a real performance result. Old proven CLIP grasp and DINO Bridge artifacts remain unchanged.

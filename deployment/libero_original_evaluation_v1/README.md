# Frozen OXE policy -> LIBERO original-instruction evaluation

User selected **100 trials per task** on2026-10-26.40 tasks,4 suites,4000 episodes per checkpoint.13 planned teacher conditions/seeds =52000 policy episodes. Native zero-action baseline is4000 additional shared episodes, not repeated13 times. No training demos are included or read.

This release admits only the independently verified DINO no-augmentation seed42 final65536 checkpoint. It does not invent trained checkpoints for the other12 runs. New checkpoints need their own small endpoint admission JSON. No active OXE training source/environment is modified.

## Upload these two files to /projects/Zeroshot

- libero_original_evaluation_v1.zip (code, protocol, fixed slots, original frozen coefficient profile)
- libero_simulator_assets_8f1084e_v1.zip (pinned simulator source/assets only; no datasets, .git, pretrained weights or policy checkpoints)

The completed seed42 checkpoint and frozen CLIP/DINO weights already exist on the cluster. Do not upload another17GB OXE dataset bundle.

## Prepare isolated Linux simulator environment, then submit ONE counted episode

```bash
(
set -euo pipefail
cd /projects/Zeroshot
unzip -n libero_original_evaluation_v1.zip
cd libero_original_evaluation_v1
bash setup_runtime.sh
bash submit_once.sh 1
)
```

Preparation installs pinned simulator dependencies in envs/libero-eval-v1 only, inheriting the existing torch2.5.1+cu118. It neither replaces the training environment nor starts training. Linux EGL remains untested until the first actual allocated-GPU episode. A node/driver/EGL failure is an incomplete technical case, not a task failure; preserve the error and retry the same fixed slot.

```bash
JOB=$(cat /projects/Zeroshot/libero_original_evaluation_v1/job_id.txt)
sacct -j "$JOB" --format=JobID,State,ExitCode,Elapsed,NodeList
tail -n 35 "/projects/Zeroshot/logs/libero-s42-${JOB}.out"
cat /projects/Zeroshot/runs/libero-original-dino-noaug-s42/summary.json
/projects/Zeroshot/envs/libero-eval-v1/bin/python /projects/Zeroshot/libero_original_evaluation_v1/verify_output.py --output /projects/Zeroshot/runs/libero-original-dino-noaug-s42
```

After this first full counted episode, review actual wall time and runtime errors before choosing a practical segment size. `bash submit_once.sh 100` would request up to100 NEW complete episodes, skipping all already completed slots. It does not reduce the required4000 or change600 policy ticks per episode. No automatic chain or duplicate submission is enabled. A run stops starting new episodes after11h; one last full episode may finish before the12h scheduler limit. A scheduler kill leaves that episode incomplete; the next run retries exactly that slot.

## Fixed rules and resume

- Official commit8f1084e3132a39270c3a13ebe37270a43ece2a01. Suites spatial/object/goal/10,10tasks each. Official filename-derived task language, not substituted BDDL wording.
- Slot i uses official initial state i modulo the asset's state count. Report unique states and repeats;100 trials do not imply100 distinct scenes. Paired environment/sampler seeds exclude condition and training seed.
- Five official zero-action settling steps; then at most600 actual20Hz native policy ticks. First predicted0.2s waypoint is executed for4ticks with fixed feedback conversion before replanning. Unknown gripper history initially; no measured-state substitution into the network.
- Strict trained learned/frozen/tokenizer/nominal-clock identity. Original schedule coefficients may be restored only from the explicitly bound training profile; no relaxed encoder check. DDPM100, full original sampler.
- Native BDDL goal success is checked after steps; valid600tick timeout is failure. Technical exceptions leave the episode incomplete. No easier replacement, resampling, checkpoint selection, expert actions or target-domain training.
- Per-episode atomic, hashed records; one kernel lock per output directory. Resume requires identical checkpoint, protocol, source/assets, simulator packages, GPU class and driver. It refuses silent runtime mixing. Kernel locks release on process death; never delete completed episode records to rerun them.
- Completed4000-episode denominator only for final model rate; incomplete reports publish completion and explicitly provisional rates. Three-training-seed SD is unavailable until the three independently completed model results exist.
- Source-to-LIBERO base/TCP registration and exact exported physical units are still uncalibrated. This evaluates the disclosed shared-Cartesian convention; it does not certify physical calibration or robot grasp success.
- Evaluation paraphrases and their budget are separate unfinished work.4000 applies to original instructions only.

No real benchmark episodes were run locally to prepare this release. New ledger/scoring checks use a synthetic environment; the prior excluded32tick trained-policy engineering check is reused.

## Shared baseline (later, same simulator runtime)

`sbatch --parsable baseline.sh N` requests up to N new baseline episodes. The same entry supports `--baseline --libero-root ... --output ... --max-new-episodes N`. It uses600 native zero actions, the same slots/seeds/settling, and a separate ledger. Run it once and share those4000 results across model comparisons; do not count a policy episode as the baseline or rerun it per model. Do not submit this baseline before the first Linux episode has established the new environment works.

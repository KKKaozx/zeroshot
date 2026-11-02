# Matched Push-T project frontend group

This is the previously planned project group, not another resource/wiring test.
Native212158 is retained. CLIP vision and text are strictly frozen: training reads
the verified213439 FP32 cache, and endpoint rollouts use the unchanged211193 live
worker with freeze/hash checks. No native learned policy weights are loaded.

Fresh author backend seed42, existing8-layer Adapter/projection seed43. Same90
episodes/10726windows/11356frames; same train-only normalizer, initial backend SHA,
shuffle, independent diffusion draw plan, original epsilon compute_loss, original
cosineDDPM100/AdamW/EMA. Effective64, micro8, final logical38 weighted correctly.
Native LR horizon512400/warmup500 is unchanged, not compressed to this budget.

One continuous84168-update run in two manually submitted parts:42084,84168.
Full raw/EMA/optimizer/LR/EMA counter/all RNG resume, complete-state digest.
Mandatory first-update loss/gradient audit and update2 disk replay are embedded;
the audit replay does not add to the accepted update budget. Every2000 updates
and segment endpoint atomically saves latest.pt plus SHA sidecar. Both segments
use the already GPU-validated normalizer restore repair. Interruptions may resume
only an audited project checkpoint; preserve the source and use a new output job.

Part1 does not score development. Part2 evaluates fixed endpoint EMA on the exact
native20 development seeds and sampling plan, logging reward series and first2
videos. Final50 and Bridge207558/final25 remain untouched. Results compare the
frontend family, including preprocessing, not Adapter in isolation. Mean overlap
reward is not grasp success. No download/install, tuning, best-checkpoint search,
native retraining or automatic subsequent job submission.

## Fresh part1 (after local checks and human submission)

Upload pusht_project_training_v1.zip to /projects/Zeroshot. Extract ONLY when the
directory does not already exist; inspect an existing directory/job before retry.
Run `bash /projects/Zeroshot/pusht_project_training_v1/submit_part1_once.sh`.
Logs: logs/pusht-project-part1-JOB.out. Output:runs/pusht-project-part1-JOB/.
Existing official Python3.9 trains. Existing project Python3.10 is used only by
the live CLIP worker at the endpoint. Each segment time limit20h; projected pure
training about12h each, not a guarantee, plus checkpoint I/O/evaluation/queue.

## Between parts

Run official Python on `verify_segment.py runs/pusht-project-part1-JOB/report.json`.
Download only report.json/latest.json/startup_checks.json, not5GB latest.pt or11GB
cache. After review, manually submit part2.sh with
`--export=ALL,PUSHT_PROJECT_RESUME=/projects/Zeroshot/runs/pusht-project-part1-JOB/latest.pt`.
No automatic dependency submission. Both parts use the same immutable contract.

Part2 output uses runs/pusht-project-part2-JOB/. Same verifier reviews full logs,
complete checkpoint SHA and20 reward series; review matched performance afterwards.

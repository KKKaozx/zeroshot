# Fixed prefix coverage pair after verified209420

An additive launcher. Requires the unchanged bridge_prefix_pair_v1 directory,
its1558-row delta_visual_tokens.pt, and the ACTUAL complete209420 report.
No preflight rerun, original8192-prefix rerun, data expansion, downloads or installs.
Both arms reload the SAME207558 trained weights, zero request port and freshAdam.
Each4096 new updates, batch8, LR1e-5, oddK16/evenK1, original frozen matched draws.
ControlK1=old1836; candidateK1=all3394. CLIP/gripper frozen. Both final weights
saved and roundtrip-checked before any development scoring. No model selection.

Only the new10/50 development set is scored. Full16 (DDPM100,3 seeds) is primary;
K1 (also100 steps) is auxiliary. Original207558 and static use exactly this set.
Both aggregate weighting schemes, every episode/task, endpoint and first-target
errors are saved. Same predeclared admission gates, no revision after scoring.
No final25 payloads opened; even admission success does NOT evaluate them here.
Failure retains207558 and closes the branch without sweeps/repeated final tests.

Submit exactly once with bash submit_formal_once.sh. It verifies209420 first;
there is no afterok dependency on an expired Slurm job. A lock/id file avoids
duplicate submissions. No automatic retry/resume after a failed formal job.
Logs: /projects/Zeroshot/logs/bridge-prefix-formal-JOB.out
Output: /projects/Zeroshot/runs/bridge-prefix-formal-JOB/
Validate completed output with verify_formal.py RUN_DIRECTORY (CPU metrics only).
Do not download the two530MB weights or1.5GB feature cache just to review results.

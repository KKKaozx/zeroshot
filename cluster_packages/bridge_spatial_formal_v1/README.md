# Fixed spatial-readout formal pair after VERIFIED210291

This is an append-only code package, not a modification of preflight sources or207558. CLIP visual/text weights and gripper stay frozen. Both arms continue the same207558 for4096 updates, freshAdam1e-5, batch8, exact same1836 windows and schedule, original full16/DDPM100. Candidate learns1026 region scores; control fixes those scores at zero. Original base trainables match. No K1/request port or added/deleted windows.

The launcher first verifies the existing210291 report; it does NOT rerun preflight. Runtime binds immutable preflight SHA sums/protocol, old cache and checkpoint, training reference, qualified10-development exports, and original209481 baseline arrays. First8 losses must match the successful corresponding preflight arm within1e-6; actual calculate() AST is unchanged.

Both final weights are saved and roundtrip-checked BEFORE development inputs are encoded/scored. Only final weights, no model selection or intermediate/Adam checkpoints. Same already-used10-development episodes/50 full16 windows; old207558 predictions reused, no old1836/40/13 rescoring. Per-arm3 sampling seeds; first target comes from the ORIGINAL full16 trajectory, not a new K1 policy. ALL prior protocol admission gates apply. Failure closes this branch, no sweeps or automatic final25.

Dependencies already on cluster: bridge_spatial_readout_v1, VERIFIED runs/bridge-spatial-preflight-210291/report.json, bridge_prefix_pair_v1 qualified validation exports, original training assets/cache, runs/bridge-prefix-formal-209481 small baseline outputs. Missing/changed assets STOP, no downloads/fallbacks. New validation50 frozen encodings are checked against209481 SHA before comparing.

Submit ONCE after unpacking to /projects/Zeroshot/bridge_spatial_formal_v1:

    bash /projects/Zeroshot/bridge_spatial_formal_v1/submit_formal_once.sh

Inspect completed output with:

    /projects/Zeroshot/envs/multitask-preflight-v1/bin/python /projects/Zeroshot/bridge_spatial_formal_v1/verify_formal.py /projects/Zeroshot/runs/bridge-spatial-formal-JOBID

Verifier checks final-weight SHA and independently recomputes full16/first-target metrics and admission from NPZ. A successful job or verifier is NOT itself an improvement verdict. No closed-loop, gripper-learning, unseen-task or statistical-significance claim. Final25 payloads never opened.

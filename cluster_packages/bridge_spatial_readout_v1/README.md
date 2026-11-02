# Frozen-CLIP post-Adapter spatial readout: ONE preflight

CLIP visual/text weights MUST remain frozen. Same original405 training episodes/1836 windows, full16 target and native DDPM100. The candidate learns1026 readout scores after the original eight Adapter blocks; control scores remain zero. Both keep the same original base trainables. No late-input/requested-horizon port, image augmentation, new labels or dataset scan.

This release has NO formal-training entry. protocol.json declares the bounded planned4096 updates/arm, unchanged data/schedule and development admission gates BEFORE any development scoring. The one preflight only replays actual207558, zero readout+native sampler, and runs8 updates per arm with weights/Adam restore checks. It does not score development or read final25. A pass is operational evidence, not improvement.

Existing dependencies: original207558 checkpoint and saved train predictions, bridge_training_coverage_v1/full_tokens.pt, bridge_expansion_training_v1/data, frozen local CLIP, existing multitask-preflight-v1 environment. Missing/stale assets STOP; no download or fallback/re-encoding. Only the405 pinned training payloads are parsed numerically; no training images decoded. Old40 development token rows are present in the inherited cache for identity checking but are not sampled or scored.

Run on cluster after unpacking to /projects/Zeroshot/bridge_spatial_readout_v1:

    bash /projects/Zeroshot/bridge_spatial_readout_v1/submit_preflight_once.sh

Inspect output and verify once completed:

    sacct -j "$SPATIAL_PREFLIGHT" --format=JobID,State,ExitCode,Elapsed,NodeList
    tail -n 30 "/projects/Zeroshot/logs/bridge-spatial-preflight-$SPATIAL_PREFLIGHT.out"
    /projects/Zeroshot/envs/multitask-preflight-v1/bin/python /projects/Zeroshot/bridge_spatial_readout_v1/verify_preflight.py "/projects/Zeroshot/runs/bridge-spatial-preflight-$SPATIAL_PREFLIGHT"

The submit-once helper prints the job ID and records it; it does not export a variable into your parent shell. Set SPATIAL_PREFLIGHT to that printed numeric ID. Do not resubmit automatically after a failed report.

No weights in this ZIP; temporary restore checkpoints are removed by exact filename after successful verification. Keep the effective207558 reference. No grasp-success or unseen-task claims.

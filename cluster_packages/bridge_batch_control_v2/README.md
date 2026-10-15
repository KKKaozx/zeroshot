# Batch control v2

Input-check repair only; same 32 train/40 development rows, models and budgets as v1. Requires the already uploaded bridge_batch_input_audit_v1/reference. Its metadata and array SHA256 are pinned. Raw image/state/action/instruction and decoded RGB must match exactly. Processed images/pose values have absolute tolerance 1e-6, rtol=0; gripper labels/current remain exact. Same-cluster preflight-to-fit hashes remain exact. No packages, weights, data downloads or large checkpoints. Preflight execution does not establish fitting/generalization.

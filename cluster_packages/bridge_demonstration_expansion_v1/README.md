#806-demonstration fixed-budget control

Prepared2026-10-17.401 new audited training episodes are appended to the original405, producing3624 training windows. The original40 development windows and65536 task-label slots are unchanged. No model was run locally.

Only401 new records are transferred in13 data archives (each below64MiB). They unpack into this package's isolated data directory. Original source dataset, old subset and old full token cache must not be overwritten. Original record IDs are retained by empty TFRecord placeholders.

Run unpack_data.py with the existing project Python, then submit preflight.sh only. It verifies all3664 input windows and816 source payloads, reuses1876 old token rows, encodes1788 additions, compares the existing207558 checkpoint on4 old development windows against the frozen reference (unchanged1e-4 context/prediction tolerances), and performs16 operational optimizer updates. The original context evaluation scope and prefix parameter flags are preserved. No candidate development or new-test scoring is done during preflight.

The actual underlying runner remains the original207558 file. run_expansion.py makes only recorded AST size/resource changes, selects expanded_inputs.py and inserts the preflight replay hook. Both advance() and context() ASTs are unchanged. Effective runtime source and wrapper snapshots are saved alongside the original source snapshot.

Only validate_preflight.py accepting the completed report may authorize fit.sh. Supply DEMO806_PREFLIGHT_JOB. Formal: seed42, batch8,8192 updates, Adam,1e-4 then1e-5 after4096, frozen CLIP/gripper, unchanged x0 diffusion loss/DDPM100. One final full training diagnostic and original40 development assessment; no extra updates, seeds, weight sweep or fitting ladder. Do not submit fit before reviewing complete preflight evidence.

The new25 test candidates remain metadata only. Neither their payloads nor consumed208056 test payloads are exported. This package has no new-test evaluator. No grasp or unseen-task claim.

Local validation covers actual401 exported payloads/record IDs, all data archives, numeric references,65536 task slots, AST preservation, syntax and CPU remapping only. Actual CLIP/Adapter/decoder, CUDA resources, old4 replay, gradients and branch restoration must be verified by the cluster preflight. Execution passing alone is not a generalization improvement verdict.

Resource limits:12GiB reserved CUDA memory,5GiB new token cache, estimated formal time below12600s. Project200GiB is a conservative planning limit with15GiB headroom, not a live quota API. Local raw export intermediates were removed after compressed-content hash verification; original E: records were read-only.

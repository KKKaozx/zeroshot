# One preflight only

No fit/automatic-submission entry is included. The raw validation_data directory
contains exactly the10 newly qualified development records with empty index
placeholders. The final25 are metadata only in protocol.json; no payloads.

Reuse remote207558 final_trainable.pt, reduced_u8192_train.npz, and existing
bridge_training_coverage_v1/full_tokens.pt. The frozenCLIP asset must already
exist; no installation/downloads. Only1558 new train images are encoded once.
Both CLIP token caches stay onCPU and only requested batches transfer toGPU.

The8 updates per arm are operational probes, NOT the4096-update formal run.
The temporary trainable/Adam serialization file is removed after exact restore.
Only the new visual token cache persists; no new final model is saved.

After uploading the ZIP, extract under/projects/Zeroshot and submit preflight.sh
once. On completion run verify_prefix_preflight.py with its run directory.
Do not submit another job if a prior preflight is still queued or running.
Execution passing is not a generalization verdict.

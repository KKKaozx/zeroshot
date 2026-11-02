# Restore original context execution after208034

Keep the original56MB package, data, frozen protocol, checkpoints207558/207746, numerical tolerances and one-time test claim untouched. This code-only wrapper imports the original evaluator and overrides only its contexts function: explicitly leave the caller's inference_mode, enter no_grad, and restore original parameter requires_grad flags. No optimizer/backward/update is created. CLIP encoding and production sampler remain unchanged.

The terminal208034 old-window comparison showed exact0 contexts and predictions for original-style context execution on both checkpoints, regardless of tested CLIP encoding styles or state array shapes. Failed-style context execution reproduced prediction differences0.000116825 and0.000137717. This identifies the joint context-execution bundle as the new replay discrepancy; it does not isolate a backend kernel or explain historical poor generalization.

Before any GPU model execution, the wrapper checks the actual208034 report, source snapshots and all16 saved prediction/context arrays against original reference values. This evidence was not independently downloaded to the local workspace; actual saved-array verification happens automatically on the cluster. Missing or inconsistent evidence stops the job. Both original strict old4window replay guards still run before any new13episode prediction. No tolerance is widened.

The original evaluator remains the underlying source snapshot; the wrapper additionally saves repair_runner_snapshot.py, repair_contract_snapshot.json and repair_entry.json and annotates the final report with the actual repair identity. The original test-set claim blocks repeated scoring across old and repaired entry points.

Local checks cover real wrapper scope restoration using tiny synthetic CPU tensors, original prefix flags, exact CPU output, unchanged toy parameters, syntax, archive integrity and availability of the original helper body. They do not load an actual model/checkpoint or run GPU inference. Submit evaluate_fixed.sh once, using the existing environment/data/weights. No new preflight or old diagnostic job is needed.

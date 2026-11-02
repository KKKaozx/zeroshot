# Diagnose208005 on four old windows only

Keep the original heldout package, protocol, both checkpoints and1e-4 tolerance unchanged. Do not resubmit208005 or bypass its assertion. This diagnostic does not perform any new13episode/60window inference and does not authorize automatic continuation to that evaluation.

At the failed assertion, raw-input/checkpoint/init and context-tolerance checks had already passed. The new entry re-encoded frozen CLIP tokens, used inference_mode for context creation and changed all parameter requires_grad flags to False. The original used cached tokens and no_grad with its original parameter flags. These are confirmed source differences, not yet a proven cause of the prediction delta.

Check the sampler from the exact original saved contexts, then old cached tokens and freshly encoded tokens with controlled context modes. Keep same checkpoints, raw four old observations, batch2, three sampling seeds and DDPM100. Bind the cached168tokens to original raw-input identity, CLIP hashes, tokenizer hashes and tensor hashes. Compare original168-row state features with the failed64-row arithmetic. Save per-case predictions and contexts even if the unchanged tolerance is exceeded; report xyz in native units and cm, quaternion and physical angle, and gripper separately. Exactly equivalent contexts/modes reuse sampling results to avoid redundant GPU work.

The two original checkpoints and old token cache remain on the cluster. No new environment, data or weights; this small archive contains code/contracts only. One A6000 job,10-minute limit. The actual cached-key and GPU/model checks remain pending; local validation is CPU AST/unit checks only. Preserve208005 report and all new diagnostic evidence.

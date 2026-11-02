# One fixed Bridge generalization candidate

Use the previously exported task/current-pose ridge predictor as a fixed XYZ prior.
Train the unchanged CLIP + eight-layer Adapter + DDPM model on position residuals.
Quaternion targets and frozen gripper scope remain unchanged. Reconstruct XYZ
before all original-unit metrics. This is an exploratory new recipe, not a proven
bug fix. The original202278 real-state control is reused, never retrained here.

Requires the existing sibling `bridge_growth128_v1`, `bridge_growth64_v1`,
`bridge_state_conditioning_v1` and inherited input-contract dependencies, the
existing dataset/CLIP/environment and retained reports. No downloads or installs.
The source hashes in `protocol.json` guard those dependencies. Do not substitute
latest production source or silence a failed identity check.

Run `preflight.sh` once, then `fit.sh` only after its successful completion. The
preflight checks16 total updates including the four-update restore replay. Its
fit gate is not an improvement criterion. It scores only four fixed training
windows, not the development set. A shared full-token cache (<256MiB) prevents
preflight/formal re-encoding on the same input, source and software identity.

Formal: one8192-update arm, original128 train/40 development windows and final
checkpoint, native DDPM100 with3 existing seeds, original learning-rate phases.
No mid-run development selection, extra seeds, added data, simulations or prior
strength searches. Score after adding the prior, never score residuals as actions.
The archive contains only code, protocol and coefficients; no images or weights.
Store at most one final trainable checkpoint (<1GiB), no optimizer checkpoints.

Keep the original training-quality gate. The candidate screen requires lower
development position error than static8.212cm and old policy8.349cm, at least6/10
episodes position-better than the old policy, and rotation <=15.276deg. Also report
the stronger fixed state-prior reference6.280cm: beating the old diffusion policy
alone does not establish that the diffusion residual adds value over its prior.
No result here establishes gripper, physical grasp or untouched-test success.

The local Windows preflight passed. It is not a cluster result or a formal
generalization result. The first draft was stopped before any update to correct
token-cache metadata serialization; incomplete artifacts were preserved locally.

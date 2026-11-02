# Read-only234100native sampler rotation trace

Zero optimizer updates and no training/gradient calculation. Reconstruct the
immutable bridge_full_capability_v1 runtime, load only234100final_policy.pt,
verify its learned tensor identities on CPU and again after GPU restoration.
Never load latest_state.pt or change existing source code/results.

Select the FIRST8original training rows and FIRST8original development rows
(global original evaluation indices0..7and128..135). Do not select by target or
error. For each original seed1101/1102/1103, trace the native100step sample
method with read-only CPU tensor copies. The method equations and random calls
are unchanged. For the development batch, skip only the original RNG calls of
15intermediate B8batches. This skip is checked against the actual first batch's
RNG consumption; each output must match the existing original NPZ exactly.
Any replay mismatch fails; do not enlarge tolerance or resample.

Records raw x0 quaternion, current noisy quaternion, reconstructed clean and
clipped clean at all100steps, plus the final pre-normalization7Dtrajectory.
Truth is used only afterward in CPU summaries, never as decoder input here.
The angular meaning of a nonzero quaternion is invariant to unit normalization;
diagnose raw norm/component error/angular direction/fallbacks without assuming
normalization itself rotates an otherwise correct prediction.

This differs from the closed20261027epsilon/context-permutation five-t probes:
new234100fresh Bridge/x0 endpoint, original on-policy sampling path, no fabricated
noisy expert input, no context swap, no objective/backbone/weight variant.

Expected2–4minutes including loading; one A40job10minute hard cap, recorder5minute
cap after construction. Need256MiB conservative output reserve. Actual runtime
and quota unknown until user executes; metadata checks use du, not authoritative
quota. Original data/model/Adam/results all retained. No follow-up submission.

Submit manually once via submit_once.sh after upload/unpack. Any job_id.txt or
SUBMITTING record must be retained. The runner cannot submit jobs or resume training.
Output report.json, trace.npz and effective_sample.py are packed in small_results.zip.
No model weights in output.16windows are a mechanism sample, not a new full-policy
score or evidence of3training seeds. Original metrics and gates stay unchanged.

CPU tests use native100step sampler on toy networks and real numeric labels.
They check exact original output replay including RNG skips, unchanged parameters/
empty Adam/gradients, negative replay rejection, AST instrumentation, saved-array
summaries and source policy metadata rejection. They do not validate real DINO/CUDA.

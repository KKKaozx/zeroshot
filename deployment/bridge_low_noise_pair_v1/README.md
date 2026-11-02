# Single t0teacher-style versus existing native response

Same234100endpoint, original16fixed windows and conditions. No training, no
backward, no optimizer update, no new full sampling. Reuse235002's raw t0outputs.
Only add6direct denoiser forwards at t0:2B8batches x3fixed Gaussian draws.

Reconstruct full7Doriginal INITIAL Gaussian draws from the old sampling seeds
and native RNG skip. Require the quaternion parts to match235002's saved initial
latent exactly, before any new denoiser probe. Use them as standard noise in
the native training forward-corruption formula at t0. This is paired by condition
and original random origin, NOT identical to the final posterior innovation.

All7input coordinates are teacher-corrupted together. Eval-mode frozen model;
not a replay of training dropout. Labels are used only to form the explicitly
teacher-style diagnostic input and evaluate it, never to fit or change weights.
No context permutation, no old epsilon/context five-t probe, no new policy score.

Interpret output vs noisy input AND vs archived native t0output. A model may
improve one contrast while failing the other; do not force a binary root cause.
One expert continuation,16development/engineering windows,one training seed.
No revised gate, no added seeds, loss weight sweep or follow-up job.

Expected1–2minutes including loading; typed A40hard cap5minutes. Direct-forward
loop cap120seconds.256MiB conservative output reserve. No new weights/cache;
full source234100and235002assets are immutable dependencies. Parent CPU admission
checks actual policy tensor identities before requesting GPU.

Source model identity, empty newly constructed Adam, gradientsNone and original
policy file metadata must remain unchanged. Output only report.json and
paired_inputs_outputs.npz, packed into small_results.zip. Verify offline using
verify_low_pair.py and downloaded235002source trace. Source weight identities
remain cluster-supported if full weights are not downloaded locally.

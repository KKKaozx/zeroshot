# Fresh full-current-Bridge capability v1

One user-selected capability run, not a mixing causal control or LIBERO release.
Fresh trainable modules/empty Adam and zero pose port; frozen DINO/CLIP text.
10352 current finite-pool Bridge windows, no other-source supervised batches.
65536 updates, B8, x0, nominal5Hz H16, original masked joint loss, three AdamW groups.
Warmup1024 and cosine65536, LR2e-4. No restart from old learned endpoints.
State normalization is fitted only on Bridge training windows; inherited numeric
target stores retain all-source provenance but no other-source images/labels enter training.

Expected6.5–7.5 GPU hours; A40 wall limit8h; training loop cap7h. Need8GiB
project reserve; CPU admission uses conservative project du, not authoritative
user quota. If quota is lower, do not submit. No cleanup is performed.
Initial+final192window native100step generation, three sampling draws; one training seed.
192 evaluation rows =128old fixed Bridge training groups +64old Bridge development groups.
Not all10352 training windows evaluated; development has been reused.

Run `bash submit_once.sh` only once after upload/unpack. It verifies SHA, runs
login-node CPU metadata/plan/target checks, then requests one typed A40 job.
job_id.txt and .submit.lock are never deleted to bypass duplicate submission.
A SUBMITTING record means an ambiguous submission must be checked manually.
The runner never submits anything. It retains latest_state.pt with Adam/RNG
every4096 updates and at a safe stop boundary. USR1/TERM request checkpointing
at the next optimizer boundary; evaluation/initial loading may still be interrupted.
No automatic requeue/resume, no extra steps or seed. Do not resubmit job.sh as
a resume command; any later continuation requires explicit budget review.

Final files: report.json, predictions_and_targets.npz, initial_predictions.npz,
initialization.json, zero_update_guard.json, verification.json; final_policy.pt
and latest_state.pt remain on cluster. `bcap_verify.py OUTPUT` verifies the full
saved state/policy/schedule/plan and array-derived metrics. `--small-only` verifies
the small download without pretending to inspect optimizer/model bytes.

The only constructor restoration allowed is the archived posterior coefficient
buffer constants, with exact full frozen identity. No pretrained encoder or learned
parameters are copied from previous policies. Vendored code is byte-preserved
from oxe_target_pair_short_v1; pose feature/runtime helpers are byte-preserved from
oxe_current_pose_preflight_v1. The new bcap_* entry and contracts are independent.

Local CPU tests use real numeric targets/native x0 loss and100step sampling with
toy frozen encoders/decoder; they are not real DINO/CUDA performance validation.
Windows shell scripts are checked for LF/shebang/text only, not claimed bash execution.

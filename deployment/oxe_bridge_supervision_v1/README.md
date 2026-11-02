# Fixed-Bridge-exposure source-supervision diagnostic

Question: do other-source supervised updates harm Bridge in the CURRENT main
architecture? This is not a repeat of the old1836window x0/pose-state experiment.
Freeze current DINO/CLS/epsilon/source-Cartesian model, source clocks/masks,
AdamW schedule, all65536global batches and RNG behavior. All input rows remain.
Remove ONLY non-Bridge contributions from supervised pose/gripper numerators;
retain ORIGINAL full-batch denominators. Bridge131244presentations are fixed.
All524276input presentations remain, including6617zero-Bridge steps. Never skip
those optimizer steps: zero gradients still permit AdamW momentum/decay effects.
Gripper graph participation follows the original valid mask, not keep count.

Reference is completed mixed seed42. No matching initial learned digest was
archived. Therefore full original initialization code/contract/runtime must
match, and BEFORE any optimizer step the exact original first-batch full loss,
pose loss, gripper loss and unclipped gradient norm must replay. Its temporary
reference backward changes neither model parameters nor RNG; clear gradients
before the counterfactual backward. Stop on ANY unexplained mismatch, never
relax identity guards. Record a new full initial digest during engineering
preflight and require that digest for the fresh formal candidate. Preflight
learned weights are NOT used in formal initialization.

The original code files are byte preserved. A declared runtime hook changes
training loss and adds audit/checkpoint metadata; the effective source hash,
loss-module hash, fixed schedule hash and intervention contract travel inside
EVERY checkpoint/export. Original model/encoder/tokenizer identity checks stay
active. These are explicitly diagnostic policies, not main thesis conditions.

Validation uses original train.policy_loss unmodified. The training-only hook
is called only at the original optimization call site, so zeroing other-source
supervision never leaks into validation. Full source validation losses at the
original endpoints remain descriptive and are not used for model selection.
Exact AdamW/RNG/plan state and intervention counters are saved for stage2.
The original interop-thread value48 is explicit, to match reference/restore.

No downloads, package installation, raw datasets or model weights in this ZIP.
Reuse existing oxe_dino_main_preflight_v1/inputs, data/oxe_nominal_pool_v1,
frozen assets and source binding. New files: one engineering resume state and
one active formal resume state (about1.6GB each), then a0.53GBfinal policy.
Require8GiB conservative project reserve. Nothing existing is deleted. The
completed candidate's own optimizer is retired only after original export
roundtrip checks, retaining all learned parameters and provenance.

ONE NEW engineering preflight:2updates, same65536LR horizon, no dev scoring.
Commit update1, save, reconstruct the original entry and restore its actual
AdamW/plan/RNG checkpoint, then commit update2 (two updates total).
It verifies the new loss/old first-batch replay and initial digest. It is not
an assertion that generalization or the source hypothesis already improved.
No extra preflight ladder or automatic formal submission.

After upload, on the cluster:

    cd /projects/Zeroshot
    unzip -n oxe_bridge_supervision_v1.zip
    cd oxe_bridge_supervision_v1
    bash submit_once.sh preflight

After reviewing success:

    bash submit_once.sh fit 1 PREFLIGHT_JOB_ID

After stage1receipt verifies:

    bash submit_once.sh fit 2 PREFLIGHT_JOB_ID

Each stage32768globalupdates;65536total, never add steps based on fit.
Reference training was about5h40total; expect roughly5-6htraining excluding
queue, plus a separate short paired inference job. Each stage wall cap4h;
this is a time safety limit, not an enlarged optimizer budget. No auto retries.

After final export:

    bash submit_once.sh evaluate

Inference compares both the old mixed final and new diagnostic final on the
SAME allocated A40 with identical frozen features and three sampling seeds.
Primary64 distinct Bridge validation groups were frozen before this training;
original8diagnostic groups are retained as a secondary anchor. New56groups
use metadata SHA rank and middle existing window, never target/error selection.
This improves coverage; it does NOT turn reused development into final test.
No LIBERO final scenes, target training, extra seeds or checkpoint search.
Results remain normalized recorded-position units; no new physical calibration.

Lower Bridge error after removal supports interference in this specific
setup. A failure to improve does not prove mixed training is always harmless.
Either outcome is retained. Neither trial automatically replaces the main
four-source model, and no dataset is permanently dropped from the thesis plan.

Local tests are synthetic loss/gradient/zero-Bridge AdamW/resume/source-hook
checks plus immutable-reference validation. Actual A40 first-batch replay and
initialization must pass in the new preflight; the cluster has not run this yet.

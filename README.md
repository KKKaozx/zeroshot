# Zero-shot robotic manipulation with vision-language models

Dissertation implementation using a frozen CLIP backbone, cross-attention adapter,
and conditional diffusion action decoder. Last progress update: **2026-10-15**.

Read the [verified status as of 15 October](docs/STATUS_20261015.md) first.
It distinguishes completed results from the limits of the end-to-end Bridge logic audit.
The [local storage review](docs/LOCAL_STORAGE_REVIEW_20261015.md) lists cleanup
candidates; no files were deleted during that review.

**Current status: training-set action fitting is supported by diagnostic results;
generalization to held-out demonstrations has not passed. No benchmark rollout or
zero-shot task-success result is claimed.**

The current Bridge diagnostic phase closes under its preregistered stopping rule.
History pair204258 is independently audited: all17 result files are complete,
the control's8192 losses and saved initial/final outputs exactly replay202278,
both train gates pass, but real history worsens development position error by4.79%
and improves only4/10 episodes. No extra history seeds or parameter sweeps follow.
The remaining root cause is not uniquely identified. The next scoped objective is
single-task, same-scene, whole-demonstration held-out prediction; no new dataset
split or training is underway. [Phase closure and evidence](docs/BRIDGE_DIAGNOSTIC_PHASE_CLOSE_20261015.md).

Latest positive control: the project's eight-layer Adapter and diffusion/gripper
policy was trained on one native PyBullet demonstration, reloaded from disk, and
physically grasped/lifted a cube in 3/3 sampling runs in that same training scene.
There was no expert handoff or artificial object attachment. This is a seen-scene
overfit result, with residual hover drift and gripper chatter, not generalization
or transfer of the Bridge checkpoint. [Protocol, traces and actual rollout](docs/PYBULLET_FIXED_GRASP_20261011.md).

Follow-up paired control: replacing one ascending window with a stationary hold
window passed both offline fits but did not stabilize closed-loop holding (0/3
in both arms; grasp/lift gate 3/3 control vs 2/3 treatment). The treatment is not
adopted as a fix. [Paired results and failure traces](docs/PYBULLET_HOLD_PAIR_RESULT_20261011.md).

Frozen follow-up: all 336 saved executed gripper decisions were reproduced.
At the first erroneous opening while holding, substituting only a training hold
image changed the decision to closed; substituting only pose or measured width
did not. This localizes that failure to image-conditioned readout behavior,
without identifying a unique faulty layer or demonstrating a closed-loop fix.
[Actual images, single-input comparisons and limits](docs/PYBULLET_FROZEN_GRIPPER_PROBE_20261011.md).

The supervisor's normalization, noise-schedule and input-preprocessing checks
were re-audited on all 1,876 frozen Bridge windows and an available retained model.
No missing inverse scale or train/eval preprocessing/schedule mismatch was found
in the inspected scope. Bilinear image resizing differs from official CLIP
bicubic preprocessing; its causal effect is untested.
[Checks, source-version verification and limits](docs/TEACHER_THREE_CONTRACT_CHECKS_20261011.md).

A controlled history-input experiment fitted 16 training demonstrations across
five tasks. Native DDPM position error was 2.67mm with repeated-current images and
2.08mm with real-previous images; both arms passed the train-only fit criteria.
A frozen same-task history swap increased error to 6.41mm. History is used on these
seen samples, but this does not establish the full-data failure cause or generalization.
[Pre-registered protocol and evidence](docs/BRIDGE_HISTORY_PAIR_20261011.md).

A nested 16-to-32 training-window control matched each paired window's exposure
count, model initialization and common noise streams. All 32 windows passed the
native DDPM fit criteria (2.24mm / 1.00deg); the original 16 also improved. Each
demonstration contributed one window, so within-demonstration phase coverage and
held-out generalization remain separate questions.
[Scale control, learning curve and verification](docs/BRIDGE_SCALE32_20261011.md).

A same-episode timepoint control added one earlier window per retained episode
(64 windows from the same 32 episodes). Native DDPM error reached 3.47mm / 0.96deg;
57/64 windows passed every sampling seed, below the pre-registered 90% coverage
gate. Six of seven failures were in `close microwave`, all position failures.
Most samples fit well; this does not identify the full-data failure cause.
Its same-window-count follow-up is described below.
[Timepoint results, failed windows and verification](docs/BRIDGE_TIMEPOINT64_20261011.md).

A matched 32-window source control replaced other-episode windows with earlier
windows from the same 16 episodes. Initialization, task proportions, complete
batch/seed streams and exposure counts matched the completed 32-episode control.
Error was 3.76mm / 1.69deg and 25/32 windows passed, versus 2.24mm / 1.00deg and
32/32 in the control. Window count alone cannot explain the fitting difference;
sample content and optimization remain to be separated. One initial window,
`00008::25`, failed in both this and the 64-window run.
[Matched source control, predictions and limits](docs/BRIDGE_SAME_COUNT32_20261011.md).

A read-only audit of that repeatedly failing initial window independently
reconstructed the raw pose targets and inspected actual episode frames. No
coordinate/scale conversion error or identical CLIP-token collapse was found
in the inspected 32 windows. Its early-window error is concentrated in later
prediction horizons (7.87mm mean versus 2.39mm at the later input); the unique
cause and the separate full-data failure remain unresolved. No training ran.
[Raw frames, world trajectories, feature checks and next bounded control](docs/BRIDGE_REPEATED_FAILURE_AUDIT_20261011.md).

A preregistered minimal control then trained that initial window alone and
together with its later window, using three independent initializations.
At the fixed 4,000-draw budget both arms passed for all initializations; the
joint early window reached 0.62--2.58mm and the late window 0.22--0.42mm.
The 512-draw checkpoint passed 3/3 alone and 2/3 jointly. This demonstrates
local memorization, while the larger-set fitting failure and generalization
remain unresolved. Only pose was trained; no gripper or simulation result.
[Minimal fitting protocol, all seeds and independent verification](docs/BRIDGE_MINIMAL_PAIR_20261011.md).

A preregistered control fixed the previously failing 32 training windows
and compares mixed batches with replicated single-window batches. Both arms
had 2,048 updates and exactly 512 draw slots per window, with three independent
initializations. Formal evaluation includes all 40 development windows from
ten held-out demonstrations. Local execution/resource preflight passed; the
completed cluster results are summarized below.
[Fixed protocol, verified small upload bundle and cluster commands](docs/BRIDGE_BATCH_CONTROL_20261012.md).

Cluster preflight 198833 stopped at an exact normalized-image hash assertion
before model loading or training. CPU diagnostic 198856 confirmed exact raw and
decoded inputs across all 72 windows, with processed image/target differences
below 4.8e-7 across environments. Revised v2 guards preserve exact raw identity
and use an absolute 1e-6 float tolerance; models and training budgets remain
unchanged. This repairs a preflight false positive, without identifying the
prior fitting/generalization failure cause. Cluster v2 execution subsequently completed.
[Failure evidence](docs/BRIDGE_BATCH_INPUT_FAILURE_20261012.md)
and [verified v2 bundle and current cluster commands](docs/BRIDGE_BATCH_CONTROL_V2_20261012.md).

Cluster batch control 198871 completed its execution checks. In all three
initializations, mixed batches fitted the fixed 32 windows better than repeated
single-window batches. Mixed initialization 44 passed (31/32 windows; 3.03mm /
1.62deg), while 42 and 43 failed the full gate. Every development result remained
worse than the static baseline on position and rotation. This supports a batch
organization effect and shows unstable training fitting, without identifying
a unique faulty layer. All 12 saved prediction files have now been independently
verified in float64 on CPU. Initialization 43 has higher training position error
than 44 on all 32 windows, spanning all five tasks; its three sampling outputs
differ by only 0.0546mm on average. Initialization 44 improves held-out position
on 5/10 episodes but both position and rotation on only 2/10. The failure therefore
includes stable training prediction bias and inadequate held-out performance;
Intermediate dynamics were not available in the original report; the tracing
result below now supplies that evidence. This prediction analysis did not train.
[All six results, limitations and read-only next analysis](docs/BRIDGE_BATCH_RESULT_198871.md).
[Verified window, horizon and episode analysis](docs/BRIDGE_BATCH_ANALYSIS_198871_20261012.md).

The tracing bundle repeats only the fixed mixed arm and observes native DDPM train
errors at updates 0/64/256/512/1024/1536/2048, with per-module gradient logs.
The recipe, data and 2048-update budget are unchanged. Local monitoring-on/off
preflight gave identical parameters, losses and predictions (3.869GiB peak VRAM).
Final trainable checkpoints stay on the cluster, capped at 4GiB total; no weights
are downloaded locally. Cluster preflight 200038 passed with exactly zero
monitoring-on/off differences. Formal trace 200039 completed and all final
train/development predictions exactly reproduced the original mixed runs.
All three initializations passed the training pose gate at update 1024 (31/32
windows each). Initialization 43 later deteriorated from 2.68mm/1.43deg to
21.86mm/5.24deg; all 32 training windows had worse position error at the end.
Logged module gradients remained finite/nonzero and no late clipping occurred.
This identifies late training deterioration, not a unique faulty module;
intermediate development performance was not evaluated. The next proposed
control compares unchanged versus reduced learning rate from the same fitted
state, before further data expansion. Its local restore preflight passed with
exactly zero same-LR parameter/loss/prediction differences (3.852GiB peak VRAM).
Formal intervention 200150 completed. Seed43's reduced-LR arm kept all32 training
windows passing at every observed continuation point, ending at0.937mm/0.344deg;
the constant-LR arm exactly reproduced21.856mm/5.242deg and0/32 passing.
Common predictions and original final predictions replayed exactly. Development
error did not improve (8.963cm/19.067deg versus constant8.716cm/18.187deg).
This supports a late-update stability effect for this initialization, not resolved
generalization. Replication200208 is now independently verified: all38 small
files are present, all26 prediction snapshots and6144 logged updates checked.
Seeds42/44 reduced-LR arms pass32/32 at all four continuation observations,
ending at1.08mm/0.40deg and0.93mm/0.38deg (native GPU metrics). Original
control predictions replay exactly. Held-out performance remains worse than
the static baseline. The fixed32-window stability diagnosis is complete;
The expanded64-window fitting run200600 is now independently verified.
Retained32 and added32 windows both pass32/32, ending at1.24mm/0.59deg
and1.31mm/0.61deg; all64 pass at four late observations. All13 artifacts,
4096 logged updates and8 prediction snapshots checked. Held-out error remains
9.75cm/18.89deg, worse than the static baseline. One initialization, frozen
gripper; broader fitting and generalization remain unproven.
[64-window results and limits](docs/BRIDGE_GROWTH64_RESULT_200600_20261013.md).
[Verified seed42 replication201429](docs/BRIDGE_GROWTH64_REPLICATION_RESULT_201429_20261013.md):
all13 files,4096 logged updates and8 saved prediction snapshots independently
checked. Same inputs/labels/recipe; only initialization changes. Retained32
and added32 both pass32/32, overall1.50mm/0.64deg; all64 pass at four late
observations. Held-out9.69cm/18.71deg remains worse than static. Two training
initializations support64-window fitting, not broader fitting or generalization.
[The frozen128-window capacity check](docs/BRIDGE_GROWTH128_FIT_20261013.md)
retains those64 windows and adds64 from32 other training demonstrations.
Model and preprocessing are unchanged;8192 updates preserve512 draws per window,
with the learning-rate reduction at4096. Local input checks,64 short updates,
five saved predictions and exact optimizer-restoration replay independently
passed. Cluster preflight201662 and fitting201663 completed successfully.
The final log reports1.829mm/0.577deg training error and an overall passing gate;
development remains8.728cm/15.847deg, worse than static.
[Independent verification of all13 artifacts](docs/BRIDGE_GROWTH128_RESULT_201663_20261013.md)
confirms8192 logged updates and eight saved prediction snapshots. Retained64
and added64 both pass64/64; all four late observations pass both group gates.
This supports fitting these128 windows at initialization43, with frozen gripper;
held-out generalization remains inadequate. The capacity checks stop here;
no256-window run is planned automatically.
[Read-only generalization diagnosis](docs/BRIDGE_GROWTH128_GENERALIZATION_201663_20261013.md)
found a concrete sampling difference:12/128 training inputs are interior indexed
windows versus20/40 development inputs. Early development windows beat static
on average; interior and late windows do not. All tasks are seen, and marginal
action-range exceptions cannot explain every failure. Frozen-image neighbors
can still have very different future trajectories. These observations do not
identify a unique cause. The [same-size39-window substitution control](docs/BRIDGE_WINDOW_COVERAGE_CONTROL_20261013.md)
is now frozen, preserving64 training episodes,128 windows and8192 updates.
All168 input checks,64 short updates,five saved predictions and exact Adam
restoration replay passed independent verification; initialization matches the
baseline local check. The19.24MiB bundle reuses existing assets. Cluster202069
subsequently logged8192 updates and a passing overall train gate (2.103mm/0.762deg),
but development9.420cm/17.267deg is worse than control201663 andstatic.
The [independent artifact verification](docs/BRIDGE_WINDOW_COVERAGE_RESULT_202069_20261013.md)
confirms all13 files,8192 logs and eight prediction snapshots. Unchanged89 pass88/89
and substituted39 pass39/39; both groups pass all four late fit gates.
Opening the drawer improves, but the other four tasks worsen. The treatment is
not adopted and this branch stops; it does not uniquely exclude or establish
sampling coverage as a cause.
[The read-only current-state audit](docs/BRIDGE_CONDITION_INPUT_AUDIT_20261013.md)
verifies raw identity,index alignment and independent pose-label reconstruction
for all168 inputs. Current end-effector pose constructs labels but is not an
explicit numeric condition of the pose decoder in this frozen runner.
Similar-image neighbors can have different observed robot states; state-neighbor
retrieval reduces position differences but does not consistently improve rotation.
The [bounded state-conditioning pair](docs/BRIDGE_STATE_CONDITIONING_PAIR_20261013.md)
has completed cluster preflight202277 and full pair202278. Both arms fit128/128
training windows. Independent CPU verification of all27 required files and16
prediction snapshots reproduces the original201663 zero-control final predictions
exactly. True current pose improves development position on7/10 demonstrations;
aggregate path error changes from8.728cm/15.847deg to8.349cm/15.256deg.
This passes the prespecified candidate screen but remains worse than static
8.212cm/15.099deg. Only one further paired training initialization is allowed.
The [fixed initialization42 replication](docs/BRIDGE_STATE_REPLICATION_20261013.md)
has passed local replay, input, snapshot and launcher checks; its28.6KiB bundle
is frozen. Cluster preflight202462 and formal pair202463 have been submitted;
both are reported COMPLETED/0:0 on A6000 (18m55s and1h22m22s).
Independent CPU analysis verifies all27 formal files,16 prediction snapshots and
8192 updates/arm. Initialization42 yields127/128 zero and117/128 real-state training
windows passing; both meet the original90% gate, but real-state fitting is worse.
Development improves8.878cm/16.478deg ->8.382cm/15.994deg, with6/10 demos improving
position. The candidate screen repeats across the two fixed initializations;
both remain worse than static. This bounded pair closes as partial benefit,
without more seed/state sweeps. The next stage is an author implementation
reference on the same Bridge protocol; its preparation is described below.
See [replication result and limits](docs/BRIDGE_STATE_REPLICATION_RESULT_202463_20261014.md).
The [same-Bridge author Workspace resource preflight](docs/BRIDGE_AUTHOR_REFERENCE_PREFLIGHT_20261014.md)
completed as job203498 on A6000. Independent inspection verifies six small files,
168 input identities, four update logs, train-only normalization and roundtrip
checks; reported peak reserved memory is5.91GiB. Parameter/EMA changes were checked
from the report, without downloading checkpoints. Training fitting and development
improvement have not been evaluated.
The [fixed8192-update author reference](docs/BRIDGE_AUTHOR_REFERENCE_TRAINING_20261014.md)
is now packaged (11,978 bytes), reusing the cluster input cache and environment.
Final EMA is primary, raw parameters secondary; all128 train and40 development
windows are evaluated. An initial timing gate enforces a bounded three-hour job.
Formal reference job203699 reports COMPLETED/0:0 in19m12s on A6000, with8192 updates.
Logged final raw/EMA train errors are7.02/8.74mm, with5/128 and0/128 windows passing;
neither meets the fixed fitting gate. PrimaryEMA development8.378cm/15.220deg
remains slightly worse than static. All eight small artifacts,8192 update logs,
configuration/target identities and all saved physical metrics are independently verified.
Six of ten development demos improve position; broccoli failures and later horizons
raise overall error. Between-draw variation contributes only1.34% of observed EMA
development squared position error. No further training is launched; the next audit
examines shared observation/target suitability. See [independent analysis and limits](docs/BRIDGE_AUTHOR_REFERENCE_RESULT_203699_20261014.md).
The [broccoli observation audit](docs/BRIDGE_BROCCOLI_OBSERVATION_AUDIT_203699.md)
now verifies14 raw episodes and the same33 retained windows, without new model inference.
Same-task ranges and geometric/phase coverage differ, but all developmentXYZ labels
lie within the actual global training ranges. One in-range author prediction follows
a different training trajectory closely; the project's same-window prediction does
not show that same pattern. These are diagnostic clues, not a unique root cause.
A [frozen author image/state intervention](docs/BRIDGE_AUTHOR_OBSERVATION_PROBE_203699.md)
is now packaged (8,533 bytes), with CPU observation/RNG and launch checks passed.
It replays all40 development predictions before comparing9 fixed pairs, preserves
native100-step sampling and weights, and performs no optimizer updates.
Inference job203876 reports COMPLETED/0:0 in1m32s on A6000. Logged image-only and
state-only mean output changes are10.04cm and1.27cm; these measure sensitivity,
not prediction error or task improvement. Original40-window replay and9-window
interventions are independently verified, with exact original replay (max difference0).
For one valid observation pair, expert trajectories differ16.49cm but model outputs
only1.00cm; this localizes an expert-trajectory discrimination failure, not a unique
faulty layer or a closed-loop task result. The project's saved201663/202278 conditions
are distinct for that pair, so the author behavior cannot be assigned to its Adapter.
The author probe closes; a frozen project202278 condition-to-decoder check is
packaged (13,298 bytes), with CPU native-sampler observer/RNG and Bash checks passed.
It gates interventions on original40-window replay, then traces5 native diffusion
stages on9 fixed valid-condition pairs with paired noise. No optimizer, CLIP/Adapter
forward pass or new weights are involved. Job204017 reports COMPLETED/0:0 in3m09s
on A6000; all40 original predictions replay exactly across3 draws. Logged held-out
condition error is already10.056cm at the first estimate and10.197cm at the last,
while train-donor conditions stay near2mm. This points to an early conditional
readout gap in these9 pairs, not a unique faulty module or generalization fix.
Both result files (387,847 bytes) are now independently recomputed: original replay,
indices/targets/contexts, per-window final metrics and all raw/clipped x0 stages
match. All9 windows have at least90% of their final position error in the first
estimate; paired train observations stay below5mm window-mean position error in
all recorded draws/stages. One held-out episode is slightly better than static
for position, the other substantially worse; both are worse for rotation.
This frozen-chain diagnostic closes without a root-cause or improvement verdict.
The last planned check of the current setup is real past observations versus
repeated current observations on the same128/40 split, retaining the real-state
port. Raw-input audit passed on all168 windows:64 train and30 development windows
have a real preceding frame; the others repeat current at the record boundary.
A50,644-byte package reuses the frozen trainer's update/sampling/evaluation bodies;
its input helper, imports/CLI, Bash and archive checks passed. GPU preflight204150
reports COMPLETED/0:0 in12m18s on A6000; its short-update fit failures are not
history-effect verdicts. Read the small preflight report before gated formal
submission. Formal pair204258 now reports COMPLETED/0:0 in1h30m29s on A6000:
the real-history arm fits training at1.965mm/0.542deg but development error is
8.749cm/15.614deg. Independent recomputation is complete:17 files, full update
logs, original control replay, final train/development and episode/subgroup metrics
verified. The position10%/rotation/6-episode candidate criteria all fail; both
train gates and control replay pass. This history configuration closes without
more training or seeds. See [final results, plot and limits](docs/BRIDGE_HISTORY_GENERALIZATION_RESULT_204258.md).
Candidate screening requires at least10% overall
position reduction, no rotation worsening, improvement in6/10 development
episodes and both training/control gates; otherwise this setup's extra diagnostics
stop. See [bounded protocol and preflight submission](docs/BRIDGE_HISTORY_GENERALIZATION_BOUND_20261014.md).
See [independent results, plot and limits](docs/BRIDGE_PROJECT_CONDITION_RESULT_204017.md).
See [protocol and submission commands](docs/BRIDGE_PROJECT_CONDITION_PROBE_202278.md)
and [independent author results and limits](docs/BRIDGE_AUTHOR_OBSERVATION_RESULT_203876.md).
Multiple architectural/optimization differences
are explicit; this is an adapted reference, not a single-module causal test.
See [verified result and stopping rules](docs/BRIDGE_STATE_PAIR_RESULT_202278_20261013.md).
[Completed64-window protocol](docs/BRIDGE_GROWTH64_FIT_20261012.md).
This retains the original32 training windows and40 development windows, adds
16 training demonstrations, and preserves512 draws/window (4096 updates total).
All104 input contracts and the full cluster run passed execution checks.
The16.04MiB bundle contains preprocessing references,
with no raw TFRecords or model weights; one final trainable checkpoint stays
on the cluster.
[Replication results and evidence limits](docs/BRIDGE_LATE_LR_REPLICATION_RESULT_200208_20261012.md).
[Next: replication protocol and submission commands](docs/BRIDGE_LATE_LR_REPLICATION_20261012.md).
[Verified late-LR result and limitations](docs/BRIDGE_LATE_LR_RESULT_200150_20261012.md).
[Prepared late learning-rate control and commands](docs/BRIDGE_LATE_LR_PAIR_20261012.md).
[Verified trace result and bounded next control](docs/BRIDGE_TRAINING_TRACE_RESULT_200039_20261012.md).
[Training trace protocol and submission commands](docs/BRIDGE_TRAINING_TRACE_20261012.md).

## Start here

- [Current progress and module evidence (中文)](docs/PROGRESS.md)
- [Next: five-task joint-model offline pilot preparation (中文)](docs/FORMAL_EXPERIMENT_PROTOCOL.md)
- [Next research step: minimal official Octo reproduction (中文)](docs/OCTO_MINIMAL_REPRODUCTION.md)
- [Minimal execution protocol and readiness checks (中文)](docs/MINIMAL_EXECUTION_PROTOCOL.md)
- [Architecture, action contract, and code map](docs/ARCHITECTURE.md)
- [Comparison with the original Diffusion Policy implementation (中文)](docs/DIFFUSION_POLICY_COMPARISON.md)
- [Executed DDPM/U-Net numerical and gradient comparison (中文)](docs/DDPM_RUNTIME_COMPARISON.md)
- [Storage snapshot and next-experiment budget](docs/STORAGE_BUDGET.md)
- [Next paired decoder experiment: upload and GPU preflight](docs/CLUSTER_DECODER_PAIR.md)
- [Environment and execution instructions](docs/RUNNING.md)
- [Supervisor requirements and completion status](docs/REQUIREMENTS.md)
- [Dataset versions and diagnostic split](docs/DATASETS.md)
- [Experiment records and evidence](thesis_tables/README.md)

## Architecture

```text
RGB observation -> frozen CLIP vision tokens --+
                                               +-> cross-attention adapter -> context
Language instruction -> frozen CLIP text tokens-+                                |
                                                                                v
Noisy action trajectory + diffusion timestep -> conditional temporal U-Net -> DDPM sampler
                                                                                |
                                                                                v
                                                            predicted robot action trajectory
```

The architecture has eight adapter layers with 512 attention channels; the latest
joint Bridge small-set and native PyBullet positive controls execute this setup.
The earlier cluster Bridge diagnostics instead used contexts from a **two-layer adapter
with 256 attention channels**. That earlier diffusion fitting froze both those
contexts and the separate gripper head; those earlier results were not joint adapter/decoder training.
CLIP ViT-L/14 has 1024-dimensional vision hidden tokens and 768-dimensional text
hidden tokens. The adapter projects these into a common attention dimension.

## Latest diagnostic result

One Bridge task (`sweep into pile`), 25 demonstrations, split by complete episode:
17 training episodes / 316 windows; 3 development episodes / 55 windows;
5 held-out test episodes / 69 windows. Each window has 16 action targets.

| Final x0 diffusion-head fit | Train | Development |
|---|---:|---:|
| Mean position error | 0.21 cm | 10.41 cm |
| Mean rotation error | 0.95 degrees | 17.07 degrees |

This run used 9,875 optimizer updates, batch size 64, and exactly 2,000 draws per
training window. 311/316 training windows passed all three sampling-seed pose
checks. Training mean accuracy does not imply perfect fitting of every action.
Development gripper balanced accuracy is about 67.4%; transition-pair correctness
is 0/13 for open-to-closed and 0/6 for closed-to-open. The gripper head was frozen.

Three **sampling** seeds are not three independent **training** runs. The three
development episodes have been repeatedly inspected and are not a fresh final
test. No held-out test-target metric is reported.

See [the full experiment record](thesis_tables/bridge_full_x0.md) and
[published numerical evidence](reports/bridge_full_x0/).

## Repository layout

```text
code/                 Main implementation and existing diagnostic tools
experiments/          Historical argument files, including diagnostic configurations
configs/              Recorded experiment configuration and provenance
scripts/cluster/      Historical NTU Slurm scripts and author-baseline runners
diagnostics/          Exported standalone diffusion experiment source snapshots
docs/                 Setup, architecture, datasets, progress, and limitations
docs/archive/         Previous README snapshots, kept for traceability
reports/              Selected small numerical reports, not weights or raw data
thesis_tables/        Experiment-to-evidence reproduction records
```

The main training entry is `code/train.py`. The evaluation entry is `code/test.py`,
which forwards to `code/evaluate_offline.py`. Dataset-free mathematical checks:

```bash
python code/test_loader.py --model-math-only
python code/train.py --help
python code/test.py --help
```

Consult [RUNNING.md](docs/RUNNING.md) before launching an experiment. Historical
argument files and cluster scripts contain machine-specific paths and must not
be treated as a portable, complete reproduction package.

## Next milestone

Complete the remaining observation, gripper-state, timing and execution contracts
in the [minimal protocol](docs/MINIMAL_EXECUTION_PROTOCOL.md), then validate a bounded
fixed-period contact control before model integration. Source and numerical
comparisons with published diffusion-policy code are documented above; they did
not resolve held-out Bridge generalization. Formal dataset splits and fair baseline
protocols remain required before additional training or benchmark claims.

LIBERO/CLIPort rollouts, four augmentation conditions, independently trained seeds,
RT-1/CLIPort comparisons, adapter-depth ablations, and cross-embodiment results
remain outstanding. PyBullet rollout code exists, but its presence is not evidence
that the requested benchmark evaluation is complete.

## References and provenance

- [CLIP](https://github.com/openai/CLIP): pretrained vision-language backbone.
- [DDPM paper](https://arxiv.org/abs/2006.11239): diffusion formulation.
- [Diffusion Policy](https://github.com/real-stanford/diffusion_policy): published
  robot action-diffusion reference; no full reproduction is claimed here.
- [BridgeData V2 author code](https://github.com/rail-berkeley/bridge_data_v2):
  loader and separate LCBC diagnostic baseline, pinned to `bc60a35` in historical setup.
- [Open X-Embodiment](https://robotics-transformer-x.github.io/),
  [LIBERO](https://libero-project.github.io/), [CLIPort](https://cliport.github.io/).

The initial repository mentioned VLA-Adapter as a proposed reproduction target.
That initial plan is preserved in the archive; VLA-Adapter is not integrated into
this codebase and must not be presented as a completed baseline.

Datasets, pretrained weights, checkpoints, private correspondence, and credentials
are excluded from Git. Preserve checkpoints and full artifacts separately with
their hashes. Commit substantive progress at least weekly.

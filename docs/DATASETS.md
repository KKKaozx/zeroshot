# Dataset selection and evaluation roles

Latest actual local inventory and source-specific blockers (2026-10-07):
[Multi-source ingestion plan (中文)](MULTISOURCE_INGESTION_PLAN.md).
All four supervisor-requested training sources are present, including Fractal.
Their approximately 685 GiB total requires staged exports under the cluster quota;
presence is not proof of verified mixed-source training readiness.

Teacher-requested primary subsets below are **reference full-dataset counts from
the correspondence**, not counts downloaded, parsed or trained by this project.

| Source | Requested version/split | Reference episodes | Language |
|---|---|---:|---|
| bridge_data_v2 | 0.0.1 | 25,460 | Natural |
| language_table | Real-robot data only | 442,226 | Natural |
| bc_z | 1.0.0 | 39,350 | Templated |
| fractal20220817_data | Exact local version must be documented before formal use | 73,499 | Templated |

Do not substitute legacy `bridge`, `bridge_data_msr`, `bc_z/old1.0.1`, or
language_table simulation/oracle variants for these training sources.
Language Table has a stick end effector; its actions must not be silently treated
as full gripper pose commands. Report natural and templated language separately.

The completed Bridge diagnostic uses **25 selected episodes of one task**, not
the complete 25,460-episode dataset. Its internal episode-disjoint split is:

| Role | Episodes | Windows | Targets |
|---|---:|---:|---:|
| Training | 17 | 316 | 5056 |
| Development validation | 3 | 55 | 880 |
| Reserved final test | 5 | 69 | Not used for reported test metrics |

Overlapping windows from one episode are not independent demonstrations. The
development episodes have been inspected repeatedly. Raw-format audits may inspect
records outside an internal training partition; that is distinct from using test
targets for model training or reported performance. Do not claim universal
absence of test-data access across all historical format audits.

Data and weights remain outside Git. Historical local paths include
`E:/dataset/bridge_v2_0.0.1/0.0.1`; cluster project data uses
`/projects/Zeroshot/data/bridge_single_step_subset` (resolved SSD project path:
`/projects/_ssd/Zeroshot`). Configure paths explicitly on another machine.

Download/source information:
[Open X-Embodiment](https://robotics-transformer-x.github.io/),
[BridgeData V2](https://rail-berkeley.github.io/bridgedata/),
[LIBERO](https://libero-project.github.io/), [CLIPort](https://cliport.github.io/).
LIBERO and CLIPort are planned simulation benchmarks; their names do not denote
the current three-episode Bridge validation split.

CLIPort is already excluded by the unified loader because its world-frame
`pose0`/`pose1` primitives lack synchronized input TCP poses for the Bridge
contract. The legacy synthetic reader now rejects calls.
`read_cliport_native_episode` preserves the native records separately; it does
not provide unified training targets. Ten training episodes (61 primitives)
passed record checks; environment replay remains unverified.

"""Collect existing evidence and freeze one read-only diagnostic design; no models."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))


def sha(name):
    return hashlib.sha256((ROOT / name).read_bytes()).hexdigest()


sources = [
    'reports/pusht_project_endpoint_analysis_20261021/report.json',
    'reports/pusht_spatial_analysis_217660_20261021/report.json',
    'reports/pusht_bypass_analysis_217820_20261022/report.json',
    'docs/TEACHER_EMAIL6_PRIORITIES_20261020.md',
    'docs/DIFFUSION_CODE_AUDIT_20261009.md',
    'reports/pusht_float_data_210690.json',
    'scripts/cluster/pusht_spatial_diagnostic_v1/protocol.json',
    'scripts/cluster/pusht_project_training_v1/project_frontend.py',
    'scripts/cluster/pusht_project_training_v1/frozen_adapter.py',
]
evidence = [
    dict(part='Data/splits/train-only normalization', status='checked_in_existing_paths',
         finding='Native and project use the same90 episodes,10726 windows,11356 frames and training-only statistics.',
         limit='Does not certify all datasets or exclude distribution differences in unexamined cases.',
         action='Do not rerun generic preprocessing checks.'),
    dict(part='Diffusion loss/schedule and U-Net conditioning', status='checked_in_existing_paths',
         finding='Pinned author epsilon loss and DDPM100 are used; conditioning already reaches down/mid/up FiLM blocks.',
         limit='Correct interfaces do not imply the denoiser has learned accurate conditional actions.',
         action='Do not propose adding already-present multilevel FiLM.'),
    dict(part='Cached CLIP versus live CLIP, freezing, EMA/restoration', status='checked_in_existing_paths',
         finding='Reference replay, frozen weights, live wiring and complete restore were checked in existing jobs.',
         limit='Not an unconditional proof that every future entry is correct.',
         action='Reuse pinned code and existing evidence, do not repeat qualification without a change.'),
    dict(part='Whole frontend', status='performance_gap_observed',
         finding='Native mean reward0.845878 versus project0.445488 on the same reused development20.',
         limit='One source training seed; includes encoder/preprocessing/Adapter/readout/projection differences.',
         action='Keep whole frontend as priority, do not claim a unique defective module.'),
    dict(part='Frozen CLIP spatial representation', status='spatial_information_accessible_under_probe',
         finding='CLIP spatial readout17.10px block-position error; shuffled-label74.23px and mean72.47px.',
         limit='Linear accessibility is not sufficient evidence of a good control representation.',
         action='Keep CLIP frozen; do not say it has no spatial information.'),
    dict(part='Adapter transformation and pooling/projection', status='suspected_not_causally_isolated',
         finding='Matched4096-dimensional readout worsens after Adapter8 on38/40 position and36/40 angle episodes.',
         limit='May reflect nonlinear re-encoding; output64 dimensions alone are not proof of a bottleneck.',
         action='Separate representation probes from actual action-prediction evidence.'),
    dict(part='Raw CLIP spatial-moment bypass217820', status='closed_negative_policy_result',
         finding='Spatial0.390775 versus global0.281801; unchanged source0.445488; adoption failed.',
         limit='Both new groups resumed training; no unchanged continuation group. No conclusion about all spatial methods.',
         action='No extra seeds, budgets, variants or final50 access for this branch.'),
    dict(part='Frozen trained endpoints: train versus unseen-demo action prediction', status='evidence_gap_in_reviewed_pusht_records',
         finding='Existing spatial probes measure state readout, and rollouts measure closed-loop reward; neither supplies this matched action diagnostic.',
         limit='Earlier Bridge small-sample fitting is a different dataset and does not fill the Push-T gap.',
         action='One bounded read-only endpoint diagnostic; no optimization or retraining.'),
]
protocol = read('scripts/cluster/pusht_spatial_diagnostic_v1/protocol.json')
train = protocol['split']['fit_episode_ids']
other_keys = [k for k in protocol['split'] if 'episode_ids' in k and k != 'fit_episode_ids']
assert len(other_keys) == 1, other_keys
heldout = protocol['split'][other_keys[0]]
assert len(train) == 90 and len(heldout) == 40 and not set(train) & set(heldout)
raw_train = read('reports/pusht_float_data_210690.json')['author_recipe_counts']['train_episode_ids']
assert set(train) == set(raw_train)


def choose(ids, role):
    return sorted(sorted(ids, key=lambda i: hashlib.sha256(
        ('pusht-endpoint-action-audit-v1:%s:%d' % (role, i)).encode()).hexdigest())[:16])


selected = dict(train=choose(train, 'train'), unseen_demo=choose(heldout, 'unseen_demo'))
assert len(set(selected['train'] + selected['unseen_demo'])) == 32
plan = dict(stage='design_only_not_an_executable_cluster_package', goal='Distinguish endpoint action learning and unseen-demonstration mismatch before changing modules',
    checkpoints=dict(native212158='runs/pusht-native84k-212158/latest.pt', project215999='runs/pusht-project-part2-215999/latest.pt'),
    endpoint_updates=84168, selection='Fixed endpoint EMA, eval mode, all parameters frozen',
    clip_frozen=True, new_training_updates=0, new_policy_rollouts=0, final50_access=False,
    selected_episode_ids=selected,
    selection_rule='16 per role by deterministic SHA256, only from existing diagnostic90/40 roles; no labels or rewards used for selection',
    windows=dict(per_episode=2, total=64, training=32, unseen_demo=32,
        rule='Within each selected episode generate the existing author create_indices(horizon16,pad_before1,pad_after7) rows; select zero-based ranks floor(n/3) and floor(2*n/3). Require n>=3 and distinct rows; otherwise stop, do not substitute episodes.',
        exact_window_manifest_status='Not yet generated from server episode_ends; this is an episode selection and row rule only'),
    primary_readouts=['Generated action-coordinate L2 error in simulation pixels for full16 and executed8, equal episode weighting',
        'No-motion baseline repeats current agent position; separate training-only mean-action baseline',
        'Fixed two matched sampling seeds1001/1002; report separately and averaged, never best-of-seed'],
    execution_slice='Original author start=n_obs_steps-1=1, end=9; align labels to same indices',
    secondary_readouts=dict(type='Forward-noised expert-action epsilon MSE, matched noise/time for both frozen policies',
        timesteps=[0,49,99], purpose='Separate supervised denoising from free reverse-chain behavior; cannot substitute for generation'),
    output_requirements=['Exact episode/window/frame manifest, checkpoint/protocol hashes',
        'Per-window/per-seed predictions and expert targets, valid/padding and executed-slice metadata',
        'Per-role/per-episode action metrics and same-window native-project differences',
        'Representative original-context replay using existing saved reference where available, no repeated full qualification'],
    engineering_budget=dict(gpu_job_wall_time_limit_minutes=30, measured_runtime_available=False,
        execution='Serial endpoint loading, reuse fixed CLIP worker/cache; no new cache of trainable modules',
        if_limit_or_memory_failure='Mark incomplete and preserve partial artifacts; do not interpret timeout as a model failure or automatically extend/retry'),
    interpretation=dict(project_poor_on_train='Prioritize conditional action learning/optimization/front-end usability, not merely unseen-data overfitting',
        project_good_on_train_but_unseen_gap='Prioritize representation reliance and distribution coverage/generalization',
        both_similar_on_expert_windows_but_rollout_gap='Prioritize evidence about off-expert inputs and accumulated closed-loop errors; offline similarity alone cannot establish equivalence',
        multimodality='A different action trajectory may also succeed; expert-coordinate error is diagnostic, not a task success metric',
        short_check='Not a new policy improvement verdict; no threshold relaxed or formal run automatically triggered'),
    cluster_package_ready=False, cluster_job_submitted=False,
    exclusions=['No further bypass217820 variant', 'No schedule/normalization/cache consistency reruns without a changed path',
        'No new backbone, no CLIP unfreezing', 'No new fitting sweep', 'No final-scene access'])
result = dict(date='2026-10-22', evidence=evidence, source_sha256={p: sha(p) for p in sources}, next_diagnostic=plan,
    new_model_execution=False, new_training=False, new_cluster_submission=False)
target = ROOT / 'reports/pusht_frontend_evidence_review_20261022.json'
target.write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(dict(evidence_rows=len(evidence), selected_episode_ids=selected,
    plan_only=True, new_training=False, cluster_job_submitted=False), indent=2))

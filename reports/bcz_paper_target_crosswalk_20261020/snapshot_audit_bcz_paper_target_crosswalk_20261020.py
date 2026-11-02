"""Read existing evidence and AST only; never decode data or run a model.

Links BC-Z paper semantics to the live parser and execution boundary. Existing
roundtrip/fitting reports are historical evidence, not freshly repeated tests.
"""
import ast
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/bcz_paper_target_crosswalk_20261020'
OUT.mkdir(parents=True, exist_ok=True)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(relative):
    return json.loads((ROOT / relative).read_text(encoding='utf-8'))


def function_span(relative, name):
    tree = ast.parse((ROOT / relative).read_text(encoding='utf-8-sig'))
    nodes = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name]
    if len(nodes) != 1:
        raise ValueError('Expected one named function: ' + name)
    node = nodes[0]
    lines = (ROOT / relative).read_text(encoding='utf-8-sig').splitlines()
    return {'path': relative, 'first_line': node.lineno, 'last_line': node.end_lineno,
            'sha256': sha(ROOT / relative)}, '\n'.join(lines[node.lineno - 1:node.end_lineno])


protected = ('code/dataset.py', 'code/models.py', 'code/adapter.py', 'code/diffusion_decoder.py',
             'code/train.py', 'code/step_training_runner.py', 'code/libero_project_policy.py',
             'code/libero_waypoint_execution.py', 'code/libero_action_interface.py',
             'code/oxe_source_semantics.py')
before = {p: sha(ROOT / p) for p in protected}
evidence_paths = ('results/bcz_native_sequence_contract_v1/summary.json',
    'results/bcz_gripper_contract_v2/summary.json',
    'results/bcz_native_sequence_fit_v1/native_fit_contract.json',
    'results/bcz_native_sequence_fit_v1/history.json',
    'reports/oxe_units_time_evidence_20261020/report.json',
    'reports/oxe_bucket_release_evidence_20261020/verification.json',
    'training_cache/oxe_core/bc_z/features.json',
    'training_cache/oxe_core/bc_z/dataset_info.json')
evidence = {p: sha(ROOT / p) for p in evidence_paths}
native = read(evidence_paths[0])
gripper = read(evidence_paths[1])
fit = read(evidence_paths[2])
history = read(evidence_paths[3])
units = read(evidence_paths[4])['sources']['BC-Z']
binding = read(evidence_paths[5])
features = read(evidence_paths[6])
metadata = read(evidence_paths[7])
assert not native['errors'] and not gripper['errors']
assert len(native['episodes']) == fit['independent_episodes'] == 23
assert native['native_sequence_summary']['checked_input_steps'] == 3332
assert native['native_sequence_summary']['checked_8d_targets'] == 33320
assert native['native_sequence_summary']['gripper_mismatches'] == 0
assert fit['timing'] == 'unknown_not_executable'
assert history[-1]['diagnostic_training_fit']
assert all(e['timestep_field'] is None for e in native['episodes'])
assert metadata['version'] == '1.0.0'
for item in binding['bucket_metadata_byte_identity_checks']:
    if item['dataset'] == 'bc_z':
        path = 'training_cache/oxe_core/bc_z/' + item['file']
        assert item['byte_equal'] and sha(ROOT / path) == item['local_sha256'] == item['remote_sha256']

anchors = {}
for label, path, name in (
    ('native_single_target', 'code/dataset.py', 'bcz_first_command_action'),
    ('native_ten_targets', 'code/dataset.py', 'bcz_native_command_chunk'),
    ('live_bcz_parser', 'code/dataset.py', '_get_tfrecord_bc_z_pose'),
    ('live_frame_encoding', 'code/dataset.py', '_encode_oxe_reached_pose'),
    ('native_execution', 'code/libero_waypoint_execution.py', 'execute_policy_chunk'),
    ('policy_config_boundary', 'code/libero_project_policy.py', '__init__')):
    # The policy source has a single constructor; no model imported here.
    anchor, text = function_span(path, name)
    anchors[label] = anchor
    (OUT / (label + '.txt')).write_text(text, encoding='utf-8')
parser = (OUT / 'live_bcz_parser.txt').read_text(encoding='utf-8')
assert 'range(start + 1, stop)' in parser
assert 'target_index - 1, 0' in parser
assert 'xyz[target_index]' in parser and 'axis_angle[target_index]' in parser
assert 'residual.reshape(step_count, 10, 3)[start]' in parser
assert 'close.reshape(step_count, 10)[start]' in parser
assert 'commands.reshape(step_count, 10)[:start, 0]' in parser
assert 'range(10)' in (OUT / 'native_ten_targets.txt').read_text(encoding='utf-8')
policy = (OUT / 'policy_config_boundary.txt').read_text(encoding='utf-8')
assert "settings.get('chunk_size') != len(execution_contract.target_times_seconds)" in policy

field_tree = features['featuresDict']['features']['steps']['sequence']['feature']['featuresDict']['features']
action_fields = field_tree['action']['featuresDict']['features']
assert set(('future/xyz_residual', 'future/axis_angle_residual', 'future/target_close')) <= action_fields.keys()
native_fields = {key: {'shape': [int(x) for x in value['tensor']['shape']['dimensions']],
                       'dtype': value['tensor']['dtype']}
                 for key, value in action_fields.items() if key.startswith('future/')}

report = {
    'format': 'bcz-paper-live-target-crosswalk-v1',
    'review_complete': True,
    'scope': 'Existing reports, local metadata and live source AST; no raw records or checkpoints loaded',
    'paper': {'url': 'https://arxiv.org/html/2202.02005v1', 'version': 'v1',
        'sections': ['5.2', 'Appendix C', 'Appendix L'],
        'claims_paraphrased': {
            'adaptive_state_difference': 'Training target selection depends on arm/gripper motion, not a universal next-row interval',
            'inference': 'Predict ten targets but execute only the first, then condition on new observation',
            'metric_limit': 'Similar future-pose validation errors need not imply similar manipulation success'},
        'not_a_diffusion_policy': True,
        'not_proof_of_local_export_units_or_seconds': True},
    'local_release': {'name': metadata['name'], 'version': metadata['version'],
        'native_fields': native_fields, 'bucket_metadata_identity_reused': True,
        'public_catalog_version': '0.1.0', 'exact_exporter_recovered': False},
    'anchors': anchors,
    'target_crosswalk': {
        'native_commands_diagnostic': {
            'indices': 'For image row t, use the ten native slots k=0..9 from that SAME row',
            'position': 'p_t + xyz_residual[t,k]',
            'orientation': 'Convert rotation vector a_t + axis_angle_residual[t,k] to an absolute quaternion; then encode relative pose',
            'gripper': '1 - 2 * target_close[t,k] (binary local command)',
            'reference': 'All slots share p_t and a_t; residuals are not accumulated',
            'physical_times': None, 'executable': False},
        'current_reached_with_preceding_command_v2': {
            'indices': 'For image row t, j=1..H uses observation row t+j',
            'position': 'p[t+j] encoded relative to fixed p[t]',
            'orientation': 'q[t+j] encoded relative to fixed q[t]',
            'gripper': '1 - 2 * target_close[t+j-1,0]',
            'input_command': 'Strictly past first-slot commands, ending at row t-1; not sensed_close',
            'native_pose_future_fields_used': False,
            'pose_command_same_native_slot': False,
            'pose_command_physical_time_association_verified': False,
            'note': 'A deliberate reached-state label policy; not evidence of an off-by-one bug or guaranteed physical synchronization',
            'physical_times': None},
        'legacy_reached': {'gripper': 'Threshold future sensed_close at the target observation',
                           'kept_for_legacy_weights': True, 'not_official_binary_sensor_calibration': True},
        'frame_is_a_separate_axis': {
            'tool': 'Local displacement and right relative quaternion',
            'source_cartesian': 'Recorded-source displacement and left relative quaternion',
            'changing_frame_does_not_restore_native_target_indices_or_time': True}},
    'reused_evidence': {
        'native_roundtrip': native['native_sequence_summary'],
        'coverage_limit': '23 selected historical training episodes, not the current 643-episode BC-Z candidate pool',
        'missing_timestep_in_historical_episodes': sum(e['timestep_field'] is None for e in native['episodes']),
        'first_native_vs_next_observation_mean_difference_native_units':
            native['alignment_hypotheses_summary']['1']['position_error_cm'] / 100,
        'mean_difference_source': 'Old audit multiplied raw Euclidean differences by 100 under the project meter convention; divided back here',
        'difference_does_not_determine_delay_or_error': True,
        'fit': {'windows': fit['shape'][0], 'native_targets_per_window': fit['shape'][1],
                'independent_episodes': fit['independent_episodes'],
                'final_gripper_correct': 316, 'final_gripper_targets': 320,
                'seen_training_only': True, 'timing': fit['timing']},
        'source_units_proven': units['position_unit_to_meters_bound_to_cached_export'],
        'actual_record_timing_proven': units['nominal_rate_verified_as_actual_record_spacing']},
    'execution_gap': {
        'current_executor_requires_explicit_times_and_feedback': True,
        'current_policy_checks_geometry_representation_and_horizon_length': True,
        'checkpoint_target_semantic_id_and_per_source_times_bound_to_executor': False,
        'evidence_strings_are_not_physical_calibration': True,
        'issue': 'Shape Hx8 and quaternion decoding do not distinguish native command slots from observed reached states or prove their deadlines',
        'remedy_before_formal_execution': 'Bind label family, reference pose, frame, per-source target-time rule and pose/gripper association to window manifest/checkpoint; executor must match this qualified contract'},
    'decision': {
        'unique_root_cause_identified': False,
        'explains_historical_bridge_only_generalization': False,
        'bcz_native_sequence_fit_or_roundtrip_to_repeat': False,
        'blind_adaptive_subsampling_to_add': False,
        'new_training_ablation_admitted': False,
        'keep_push_t_frontend_comparison_as_planned': True,
        'next_bounded_item': 'Extend the planned formal window/target protocol with label family and time/command-pose binding; qualify with source evidence before selecting target times or training',
        'training_ready': False},
    'input_evidence_sha256': evidence,
    'protected_source_sha256': before,
    'raw_tfrecords_read': False, 'model_loaded': False, 'new_jobs_submitted': False,
    'new_generalization_verdict': None,
}
after = {p: sha(ROOT / p) for p in protected}
assert before == after
report['protected_sources_unchanged'] = True
report['script_sha256'] = sha(__file__)
shutil.copyfile(__file__, OUT / ('snapshot_' + Path(__file__).name))
(OUT / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print('BC-Z TARGET CROSSWALK COMPLETE: existing evidence reused; no training verdict')
print('Observed reached labels != native command slots; physical pairing/deadlines remain unqualified.')

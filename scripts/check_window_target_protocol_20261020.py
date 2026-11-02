"""Synthetic semantic-contract checks; no torch, model, raw data or simulation."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
from window_target_protocol import (FORMAT, CARTESIAN, NATIVE, REACHED,
    validate_protocol, window_layout, validate_window_layout, bind_artifacts,
    validate_binding, validate_checkpoint_binding, check_prediction_execution_binding)

OUT = ROOT / 'reports/window_target_protocol_20261020'
OUT.mkdir(parents=True, exist_ok=True)
positive, negative = [], []
identity = hashlib.sha256(b'synthetic windows, not an OXE manifest').hexdigest()
reference = {'uri': 'synthetic://fixture/declarations',
    'sha256': hashlib.sha256(b'Synthetic unit/time/pairing evidence only').hexdigest(),
    'claim': 'Declared unit/time/pairing in a synthetic fixture; not source qualification'}


def fixture(known=False, native=False):
    family = NATIVE if native else REACHED
    row = {'family': family, 'target_indices': [0, 1, 2] if native else [1, 2, 3],
        'gripper_rule': 'same_native_slot_absolute' if native else 'preceding_observation_first_command',
        'supervision_mask': [1] * 8, 'position_to_meters': 1 if known else None,
        'unit_evidence': [copy.deepcopy(reference)] if known else [],
        'timing': {'kind': 'fixed_offsets_seconds' if known else 'unknown',
                   'offsets_seconds': [.1, .2, .3] if known else None,
                   'evidence': [copy.deepcopy(reference)] if known else []},
        'pairing_evidence': [copy.deepcopy(reference)] if known else []}
    return {'format': FORMAT, 'scope': 'synthetic_fixture', 'dataset_identity': identity,
        'output': {'family': family, 'horizon': 3, 'action_dim': 8,
                   'representation': CARTESIAN, 'reference': 'fixed_input_observation',
                   'position_scale_meters': .1,
                   'shared_target_times_seconds': [.1, .2, .3] if known else None},
        'sources': {'synthetic_bcz': row}}


def artifacts(protocol):
    manifest = {'dataset_identity': identity, 'split_indices': {'train': [0], 'validation': [1], 'test': [2]}}
    config = {'action': {'representation': CARTESIAN, 'position_scale_meters': .1},
              'model': {'chunk_size': 3, 'action_dim': 8}}
    original = copy.deepcopy((manifest, config, protocol))
    bound, cfg = bind_artifacts(protocol, manifest, config)
    assert original == (manifest, config, protocol), 'Binding mutated source artifacts'
    checkpoint = {'config': cfg, 'target_binding': copy.deepcopy(bound['target_binding'])}
    assert validate_checkpoint_binding(bound, checkpoint)
    return bound, checkpoint


def runtime(binding):
    output = binding['protocol']['output']
    return {'kind': 'model_prediction', 'target_protocol_sha256': binding['sha256'],
        'action_representation': output['representation'], 'position_scale_meters': .1,
        'reference': 'fixed_input_observation', 'target_times_seconds': [.1, .2, .3],
        'native_control_hz': 20., 'execute_waypoints': 1}


def rejects(name, function):
    try:
        function()
    except ValueError as exc:
        negative.append({'case': name, 'reason': str(exc)})
        return
    raise AssertionError('Expected rejection: ' + name)


def mutate_protocol(base, path, value):
    modified = copy.deepcopy(base)
    item = modified
    for key in path[:-1]:
        item = item[key]
    item[path[-1]] = value
    return modified


unknown, known = fixture(), fixture(known=True)
validate_protocol(unknown)
observed = window_layout(unknown, 'synthetic_bcz', input_row=12)
assert observed['pose_rows'] == [13, 14, 15]
assert observed['gripper_rows'] == [12, 13, 14] and observed['gripper_slots'] == [0, 0, 0]
assert observed['target_times_seconds'] == [None] * 3
assert validate_window_layout(unknown, observed)
positive.append('Observed pose and preceding-command rows explicit; unknown times preserved')

native = fixture(native=True)
native_window = window_layout(native, 'synthetic_bcz', input_row=12)
assert native_window['pose_rows'] == native_window['gripper_rows'] == [12] * 3
assert native_window['pose_slots'] == native_window['gripper_slots'] == [0, 1, 2]
assert validate_window_layout(native, native_window)
positive.append('Native slots use the same input row and matching gripper slot')

tail = window_layout(known, 'synthetic_bcz', input_row=12, valid_targets=1)
assert tail['pose_rows'] == [13, None, None]
assert tail['target_times_seconds'] == [.1, None, None]
assert tail['supervision_masks'][1:] == [[0] * 8, [0] * 8]
assert validate_window_layout(known, tail)
positive.append('Short tails carry no padded supervision or invented deadlines')

per_window = fixture()
per_window['sources']['synthetic_bcz']['timing'] = {'kind': 'per_window_timestamps',
    'offsets_seconds': None, 'evidence': [copy.deepcopy(reference)]}
irregular = window_layout(per_window, 'synthetic_bcz', input_row=4, target_times_seconds=[.07, .19, .41])
assert validate_window_layout(per_window, irregular)
assert per_window['output']['shared_target_times_seconds'] is None
positive.append('Recorded per-window times remain per-window, not a shared inference grid')

manifest, checkpoint = artifacts(known)
binding = checkpoint['target_binding']
result = check_prediction_execution_binding(binding, runtime(binding))
assert result['native_ticks'] == [2, 4, 6]
assert result['contract_compatible'] and not result['physical_qualification_proven']
assert not result['learned_success_proven'] and result['synthetic_fixture']
positive.append('Window/config/checkpoint/runtime identity matches, without claiming calibration/success')

# A planar training source is permitted alongside a full-pose source. Its source
# labels remain partial; this is not a reason to deny a complete model prediction.
mixed = copy.deepcopy(known)
planar = copy.deepcopy(mixed['sources']['synthetic_bcz'])
planar.update(gripper_rule='no_gripper', supervision_mask=[1, 1, 0, 0, 0, 0, 0, 0], pairing_evidence=[])
mixed['sources']['synthetic_planar'] = planar
planar_window = window_layout(mixed, 'synthetic_planar', input_row=2)
assert planar_window['gripper_rows'] == [None] * 3
assert validate_window_layout(mixed, planar_window)
_, mixed_ckpt = artifacts(mixed)
assert check_prediction_execution_binding(mixed_ckpt['target_binding'], runtime(mixed_ckpt['target_binding']))['contract_compatible']
positive.append('Mixed partial supervision does not invent gripper labels or deny a full model prediction')

for name, base in (('unknown_time', unknown), ('native_unknown_time', native), ('irregular_not_shared', per_window)):
    _, ckpt = artifacts(base)
    rejects(name, lambda c=ckpt: check_prediction_execution_binding(c['target_binding'], runtime(c['target_binding'])))

for name, path, value in (
    ('native_family_disguised_as_reached', ['sources', 'synthetic_bcz', 'family'], NATIVE),
    ('native_slots_disguised_as_rows', ['sources', 'synthetic_bcz', 'target_indices'], [0, 1, 2]),
    ('boolean_horizon', ['output', 'horizon'], True),
    ('incremental_reference_not_fixed', ['output', 'reference'], 'previous_waypoint'),
    ('incoherent_quaternion_mask', ['sources', 'synthetic_bcz', 'supervision_mask'], [1, 1, 1, 1, 0, 1, 1, 1]),
    ('invented_gripper_on_planar_mask', ['sources', 'synthetic_bcz', 'supervision_mask'], [1, 1, 0, 0, 0, 0, 0, 0]),
    ('negative_unit_factor', ['sources', 'synthetic_bcz', 'position_to_meters'], -1),
    ('known_unit_without_evidence', ['sources', 'synthetic_bcz', 'unit_evidence'], []),
    ('known_time_without_evidence', ['sources', 'synthetic_bcz', 'timing', 'evidence'], []),
    ('source_grid_conflict', ['sources', 'synthetic_bcz', 'timing', 'offsets_seconds'], [.1, .3, .5]),
    ('nonfinite_time', ['output', 'shared_target_times_seconds'], [.1, .2, float('nan')]),
    ('duplicate_time', ['sources', 'synthetic_bcz', 'timing', 'offsets_seconds'], [.1, .1, .3]),
    ('synthetic_evidence_in_real_scope', ['scope'], 'recorded_source'),
    ('unfrozen_dataset_identity', ['dataset_identity'], None)):
    changed = mutate_protocol(known, path, value)
    rejects(name, lambda p=changed: validate_protocol(p))

for name, key, value in (
    ('wrong_native_pose_row', 'pose_rows', [12, 13, 14]),
    ('gripper_native_slot_mismatch', 'gripper_slots', [0, 2, 1]),
    ('float_row_index', 'pose_rows', [12., 12, 12]),
    ('boolean_slot_index', 'pose_slots', [0, True, 2])):
    changed = copy.deepcopy(native_window); changed[key] = value
    rejects(name, lambda w=changed: validate_window_layout(native, w))
changed = copy.deepcopy(observed); changed['gripper_rows'][0] += 1
rejects('preceding_command_replaced_by_target_row', lambda: validate_window_layout(unknown, changed))
bad_tail = copy.deepcopy(tail); bad_tail['supervision_masks'][1] = [1] * 8
rejects('padding_became_label', lambda: validate_window_layout(known, bad_tail))
bad_mask = copy.deepcopy(tail); bad_mask['supervision_masks'][0][0] = True
rejects('boolean_window_mask', lambda: validate_window_layout(known, bad_mask))
rejects('empty_all_padding_window', lambda: window_layout(known, 'synthetic_bcz', input_row=0, valid_targets=0))
rejects('padding_native_slots', lambda: window_layout(native, 'synthetic_bcz', input_row=0, valid_targets=2))
rejects('guess_0_1_seconds', lambda: window_layout(unknown, 'synthetic_bcz', input_row=0, target_times_seconds=[.1, .2, .3]))

for name, key, value in (
    ('different_runtime_digest', 'target_protocol_sha256', 'a' * 64),
    ('different_runtime_times', 'target_times_seconds', [.1, .2, .4]),
    ('different_runtime_scale', 'position_scale_meters', 1),
    ('different_runtime_frame', 'action_representation', 'tool_relative_pose_open_positive_v2'),
    ('different_runtime_reference', 'reference', 'previous_waypoint'),
    ('label_placeholder_used_as_prediction', 'kind', 'source_label'),
    ('fractional_native_tick', 'native_control_hz', 15),
    ('execution_prefix_over_horizon', 'execute_waypoints', 4),
    ('boolean_execution_prefix', 'execute_waypoints', True)):
    modified_runtime = runtime(binding); modified_runtime[key] = value
    rejects(name, lambda v=modified_runtime: check_prediction_execution_binding(binding, v))

no_pair = copy.deepcopy(known); no_pair['sources']['synthetic_bcz']['pairing_evidence'] = []
_, no_pair_ckpt = artifacts(no_pair)
rejects('unqualified_pose_gripper_association', lambda: check_prediction_execution_binding(
    no_pair_ckpt['target_binding'], runtime(no_pair_ckpt['target_binding'])))
no_unit = copy.deepcopy(known); no_unit['sources']['synthetic_bcz']['position_to_meters'] = None
_, no_unit_ckpt = artifacts(no_unit)
rejects('unqualified_units', lambda: check_prediction_execution_binding(
    no_unit_ckpt['target_binding'], runtime(no_unit_ckpt['target_binding'])))
tampered = copy.deepcopy(binding); tampered['protocol']['output']['position_scale_meters'] = 1
rejects('tampered_protocol_kept_old_digest', lambda: validate_binding(tampered))
wrong_config = copy.deepcopy(checkpoint); wrong_config['config']['model']['chunk_size'] = 4
rejects('checkpoint_horizon_changed', lambda: validate_checkpoint_binding(manifest, wrong_config))
missing_digest = copy.deepcopy(checkpoint); del missing_digest['config']['action']['target_protocol_sha256']
rejects('checkpoint_digest_missing', lambda: validate_checkpoint_binding(manifest, missing_digest))
different_manifest = copy.deepcopy(manifest); different_manifest['dataset_identity'] = 'b' * 64
rejects('manifest_changed_dataset', lambda: validate_checkpoint_binding(different_manifest, checkpoint))
rejects('overwrite_native_with_reached', lambda: bind_artifacts(native, manifest, checkpoint['config']))

# Check old runtime/runner evidence still points to exactly the same live code.
protected = {}
for report_path in ('reports/step_main_runner_20261020/report.json',
                    'reports/step_main_runner_cuda_20261020/report.json',
                    'reports/bcz_paper_target_crosswalk_20261020/report.json'):
    previous = json.loads((ROOT / report_path).read_text(encoding='utf-8'))
    for path, expected in previous.get('sha256', previous.get('protected_source_sha256', {})).items():
        actual = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        assert actual == expected, path
        protected[path] = actual
sources = ('code/window_target_protocol.py', 'scripts/check_window_target_protocol_20261020.py')
hashes = {}
for relative in sources:
    source = ROOT / relative
    hashes[relative] = hashlib.sha256(source.read_bytes()).hexdigest()
    shutil.copyfile(source, OUT / ('snapshot_' + source.name))
report = {'passed': True, 'scope': 'synthetic protocol and artifact compatibility only',
    'positive_checks': positive, 'rejected_cases': negative, 'source_sha256': hashes,
    'existing_runner_and_execution_sources_unchanged': protected,
    'actual_main_window_manifest_created': False, 'production_runner_integration_complete': False,
    'physical_qualification_proven': False, 'models_or_simulation_executed': False,
    'new_jobs_submitted': False, 'generalization_verdict': None}
(OUT / 'synthetic_known_protocol.json').write_text(json.dumps(known, indent=2), encoding='utf-8')
(OUT / 'synthetic_unknown_protocol.json').write_text(json.dumps(unknown, indent=2), encoding='utf-8')
(OUT / 'synthetic_bound_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
(OUT / 'synthetic_checkpoint_metadata.json').write_text(json.dumps(checkpoint, indent=2), encoding='utf-8')
(OUT / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print('WINDOW TARGET PROTOCOL PASSED:', len(positive), 'positive checks;', len(negative), 'rejections')
print('No real source qualification, main-runner integration, training or simulation.')

"""Prospective window/checkpoint/executor semantic binding, without model imports.

Validation checks declarations and consistency. An evidence URI/hash is a
reference, not an independently proven calibration. No source is admitted, no
timestamps are inferred, and existing runners are not implicitly opted in.
"""
import copy
import hashlib
import json
import math
import re

FORMAT = 'window-target-protocol-v1'
BINDING_FORMAT = 'window-target-binding-v1'
TOOL = 'tool_relative_pose_open_positive_v2'
CARTESIAN = 'source_cartesian_delta_pose_open_positive_v1'
REACHED = 'observed_future_reached_pose'
NATIVE = 'native_future_setpoint'


def _fail(message):
    raise ValueError(message)


def _keys(value, expected, label):
    if not isinstance(value, dict) or set(value) != set(expected):
        _fail('Unexpected or missing keys in ' + label)


def _number(value, positive=False):
    return (type(value) in (int, float) and math.isfinite(value)
            and (not positive or value > 0))


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def _sha(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def _evidence(rows, *, required, scope):
    if not isinstance(rows, list) or (required and not rows):
        _fail('Explicit evidence references required for a documented declaration')
    for row in rows:
        _keys(row, ('uri', 'sha256', 'claim'), 'evidence reference')
        if (not all(isinstance(row[k], str) and row[k].strip() for k in ('uri', 'claim'))
                or not _sha(row['sha256'])):
            _fail('Evidence requires URI, claim and content SHA256')
        if scope == 'recorded_source' and row['uri'].startswith('synthetic:'):
            _fail('Synthetic evidence cannot qualify a recorded source')


def _times(values, horizon):
    if (not isinstance(values, list) or len(values) != horizon
            or not all(_number(x, positive=True) for x in values)
            or not all(b > a for a, b in zip(values, values[1:]))):
        _fail('Target times must be H positive, finite, strictly increasing offsets')


def validate_protocol(protocol):
    """Accept explicit unknown units/time for offline use; never fill them in."""
    _keys(protocol, ('format', 'scope', 'dataset_identity', 'output', 'sources'), 'protocol')
    if protocol['format'] != FORMAT or protocol['scope'] not in ('synthetic_fixture', 'recorded_source'):
        _fail('Explicit protocol version and scope required')
    if not _sha(protocol['dataset_identity']):
        _fail('Frozen actual-window dataset identity SHA256 required, not an episode-only draft')
    output = protocol['output']
    _keys(output, ('family', 'horizon', 'action_dim', 'representation', 'reference',
                   'position_scale_meters', 'shared_target_times_seconds'), 'output')
    horizon = output['horizon']
    if (type(horizon) is not int or horizon < 1 or type(output['action_dim']) is not int
            or output['action_dim'] != 8 or output['family'] not in (REACHED, NATIVE)
            or output['representation'] not in (TOOL, CARTESIAN)
            or output['reference'] != 'fixed_input_observation'
            or not _number(output['position_scale_meters'], positive=True)):
        _fail('Supported fixed-reference 8D output declaration required')
    grid = output['shared_target_times_seconds']
    if grid is not None:
        _times(grid, horizon)
    sources = protocol['sources']
    if not isinstance(sources, dict) or not sources:
        _fail('At least one explicit source declaration required')
    for source, row in sources.items():
        if not isinstance(source, str) or not source.strip():
            _fail('Nonempty source identifier required')
        _keys(row, ('family', 'target_indices', 'gripper_rule', 'supervision_mask',
                    'position_to_meters', 'unit_evidence', 'timing', 'pairing_evidence'), 'source ' + source)
        if row['family'] != output['family']:
            _fail('Native slots and observed reached states cannot silently share an output family')
        indices = row['target_indices']
        if (not isinstance(indices, list) or len(indices) != horizon
                or not all(type(i) is int for i in indices)
                or not all(b > a for a, b in zip(indices, indices[1:]))):
            _fail('Ordered integer target indices required')
        if row['family'] == REACHED and indices != list(range(1, horizon + 1)):
            _fail('This reached-row contract explicitly uses t+1 through t+H')
        if row['family'] == NATIVE and not all(0 <= i < 10 for i in indices):
            _fail('BC-Z native target indices must refer to slots 0..9 in one observation')
        mask = row['supervision_mask']
        if (not isinstance(mask, list) or len(mask) != 8
                or not all(type(x) in (int, float) and x in (0, 1) for x in mask)
                or not any(mask) or len(set(mask[3:7])) != 1):
            _fail('Explicit nonempty binary 8D supervision; quaternion mask must be coherent')
        rule = row['gripper_rule']
        valid_rules = ('same_native_slot_absolute', 'no_gripper') if row['family'] == NATIVE else (
            'preceding_observation_first_command', 'observed_sensor_threshold_legacy', 'no_gripper')
        if rule not in valid_rules or ((rule == 'no_gripper') != (mask[7] == 0)):
            _fail('Gripper rule must match family and availability mask')
        factor = row['position_to_meters']
        if factor is not None and not _number(factor, positive=True):
            _fail('Unknown unit is null; documented conversion must be positive and finite')
        _evidence(row['unit_evidence'], required=factor is not None, scope=protocol['scope'])
        _evidence(row['pairing_evidence'], required=False, scope=protocol['scope'])
        if rule == 'no_gripper' and row['pairing_evidence']:
            _fail('A source without a gripper must not invent pose/gripper pairing evidence')
        timing = row['timing']
        _keys(timing, ('kind', 'offsets_seconds', 'evidence'), 'timing')
        if timing['kind'] not in ('unknown', 'fixed_offsets_seconds', 'per_window_timestamps'):
            _fail('Explicit unknown, fixed offsets, or recorded per-window timestamps required')
        if timing['kind'] == 'fixed_offsets_seconds':
            _times(timing['offsets_seconds'], horizon)
        elif timing['offsets_seconds'] is not None:
            _fail('Unknown/per-window timing must not pretend to be a fixed grid')
        _evidence(timing['evidence'], required=timing['kind'] != 'unknown', scope=protocol['scope'])
        if grid is not None and (timing['kind'] != 'fixed_offsets_seconds' or timing['offsets_seconds'] != grid):
            _fail('A declared common grid must match every source; nominal rates cannot substitute')
    return _digest(protocol)


def window_layout(protocol, source, *, input_row, valid_targets=None, target_times_seconds=None):
    """Generate expected indices/masks, not a data record or a qualified label."""
    validate_protocol(protocol)
    if source not in protocol['sources'] or type(input_row) is not int or input_row < 0:
        _fail('Known source and nonnegative input row required')
    row, output = protocol['sources'][source], protocol['output']
    horizon = output['horizon']
    count = horizon if valid_targets is None else valid_targets
    if type(count) is not int or not 1 <= count <= horizon:
        _fail('At least one measured target required; padding is not supervision')
    if row['family'] == NATIVE and count != horizon:
        _fail('Native slots are not padded with fabricated future targets')
    slots = row['target_indices']
    pose_rows, pose_slots, grip_rows, grip_slots, masks = [], [], [], [], []
    for j, index in enumerate(slots):
        valid = j < count
        native = row['family'] == NATIVE
        pose_rows.append((input_row if native else input_row + index) if valid else None)
        pose_slots.append(index if valid and native else None)
        rule = row['gripper_rule']
        if not valid or rule == 'no_gripper':
            grip_rows.append(None); grip_slots.append(None)
        elif native:
            grip_rows.append(input_row); grip_slots.append(index)
        elif rule == 'preceding_observation_first_command':
            grip_rows.append(input_row + index - 1); grip_slots.append(0)
        else:
            grip_rows.append(input_row + index); grip_slots.append(None)
        masks.append(list(row['supervision_mask']) if valid else [0] * 8)
    timing = row['timing']
    if timing['kind'] == 'per_window_timestamps':
        _times(target_times_seconds, count)
        times = list(target_times_seconds) + [None] * (horizon - count)
    elif timing['kind'] == 'fixed_offsets_seconds':
        if target_times_seconds is not None:
            _fail('Do not override the declared fixed timing per window')
        times = list(timing['offsets_seconds'][:count]) + [None] * (horizon - count)
    else:
        if target_times_seconds is not None:
            _fail('Unknown source timing cannot be filled by a guessed runtime interval')
        times = [None] * horizon
    return {'source': source, 'input_row': input_row, 'valid_targets': count,
            'pose_rows': pose_rows, 'pose_slots': pose_slots,
            'gripper_rows': grip_rows, 'gripper_slots': grip_slots,
            'supervision_masks': masks, 'target_times_seconds': times}


def validate_window_layout(protocol, window):
    validate_protocol(protocol)
    _keys(window, ('source', 'input_row', 'valid_targets', 'pose_rows', 'pose_slots',
                   'gripper_rows', 'gripper_slots', 'supervision_masks', 'target_times_seconds'), 'window')
    if not isinstance(window['source'], str) or window['source'] not in protocol['sources']:
        _fail('Window source is absent from the bound protocol')
    count = window['valid_targets']
    if (type(count) is not int or not 1 <= count <= protocol['output']['horizon']
            or not isinstance(window['target_times_seconds'], list)):
        _fail('Invalid valid-target count or time array')
    horizon = protocol['output']['horizon']
    for key in ('pose_rows', 'pose_slots', 'gripper_rows', 'gripper_slots'):
        entries = window[key]
        if (not isinstance(entries, list) or len(entries) != horizon
                or any(x is not None and (type(x) is not int or x < 0) for x in entries)):
            _fail('Window row/slot indices must be nullable nonnegative integers')
    masks = window['supervision_masks']
    if (not isinstance(masks, list) or len(masks) != horizon
            or any(not isinstance(row, list) or len(row) != 8
                   or any(type(x) not in (int, float) or x not in (0, 1) for x in row) for row in masks)):
        _fail('Window masks must be an explicit H by 8 binary array')
    expected = window_layout(protocol, window['source'], input_row=window['input_row'],
        valid_targets=window['valid_targets'], target_times_seconds=(
            window['target_times_seconds'][:window['valid_targets']]
            if protocol['sources'][window['source']]['timing']['kind'] == 'per_window_timestamps' else None))
    if window != expected:
        _fail('Window pose/gripper indices, masks or times differ from the target protocol')
    return True


def bind_artifacts(protocol, window_manifest, model_config):
    """Return new bound copies. Does not enumerate windows or rewrite artifacts."""
    digest = validate_protocol(protocol)
    if window_manifest.get('dataset_identity') != protocol['dataset_identity']:
        _fail('Window manifest identity differs')
    output = protocol['output']
    action, model = model_config.get('action', {}), model_config.get('model', {})
    if (action.get('representation') != output['representation']
            or action.get('position_scale_meters') != output['position_scale_meters']
            or type(model.get('chunk_size')) is not int or model['chunk_size'] != output['horizon']
            or type(model.get('action_dim')) is not int or model['action_dim'] != 8):
        _fail('Actual model geometry/horizon differs from the protocol')
    binding = {'format': BINDING_FORMAT, 'protocol': copy.deepcopy(protocol), 'sha256': digest}
    manifest, config = copy.deepcopy(window_manifest), copy.deepcopy(model_config)
    if 'target_binding' in manifest and manifest['target_binding'] != binding:
        _fail('Cannot overwrite a different existing manifest binding')
    for artifact in (manifest, config['action']):
        existing = artifact.get('target_protocol_sha256')
        if existing is not None and existing != digest:
            _fail('Cannot overwrite an existing different target binding')
        artifact['target_protocol_sha256'] = digest
    manifest['target_binding'] = binding
    return manifest, config


def validate_binding(binding):
    _keys(binding, ('format', 'protocol', 'sha256'), 'binding')
    digest = validate_protocol(binding['protocol'])
    if binding['format'] != BINDING_FORMAT or binding['sha256'] != digest:
        _fail('Target binding digest/version differs')
    return binding['protocol']


def validate_checkpoint_binding(window_manifest, checkpoint):
    """Prospective checkpoint shape: config + target_binding, independent of RNG."""
    protocol = validate_binding(checkpoint['target_binding'])
    if window_manifest.get('target_binding') != checkpoint['target_binding']:
        _fail('Checkpoint and window manifest use different target protocols')
    bound, config = bind_artifacts(protocol, window_manifest, checkpoint['config'])
    if (bound != window_manifest or config != checkpoint['config']):
        _fail('Missing target digest in checkpoint configuration or manifest')
    return True


def check_prediction_execution_binding(binding, runtime):
    """Compatibility only; neither calibration nor learned competence is proved."""
    protocol = validate_binding(binding)
    output = protocol['output']
    if runtime.get('kind') != 'model_prediction':
        _fail('A source label/placeholder is not a complete model prediction')
    if not _number(runtime.get('position_scale_meters'), positive=True):
        _fail('Runtime metric scale must be a finite positive number')
    if (runtime.get('target_protocol_sha256') != binding['sha256']
            or runtime.get('action_representation') != output['representation']
            or runtime.get('position_scale_meters') != output['position_scale_meters']
            or runtime.get('reference') != output['reference']):
        _fail('Runtime target identity/geometry differs from checkpoint')
    times = output['shared_target_times_seconds']
    _times(runtime.get('target_times_seconds'), output['horizon'])
    if times is None or runtime.get('target_times_seconds') != times:
        _fail('A source-backed common target grid must match runtime; no default rate')
    for source in protocol['sources'].values():
        if source['position_to_meters'] is None:
            _fail('Unresolved source units cannot qualify execution compatibility')
        if source['gripper_rule'] != 'no_gripper' and not source['pairing_evidence']:
            _fail('Pose/gripper association evidence is unresolved')
    hz, prefix = runtime.get('native_control_hz'), runtime.get('execute_waypoints')
    if (not _number(hz, positive=True) or type(prefix) is not int
            or not 1 <= prefix <= output['horizon']):
        _fail('Positive native frequency and explicit bounded execution prefix required')
    ticks = [value * hz for value in times]
    if any(not math.isfinite(x) or abs(x - round(x)) > 1e-8 or round(x) < 1 for x in ticks):
        _fail('Targets must coincide with native ticks; no implicit resampling')
    return {'contract_compatible': True, 'native_ticks': [round(x) for x in ticks],
            'physical_qualification_proven': False, 'learned_success_proven': False,
            'synthetic_fixture': protocol['scope'] == 'synthetic_fixture'}

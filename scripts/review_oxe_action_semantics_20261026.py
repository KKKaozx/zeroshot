"""Record a read-only semantic review; never import model/data/simulator code."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/oxe_action_execution_semantics_20261026'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8-sig'))


def excerpt(name, symbol):
    path = ROOT / name
    text = path.read_text(encoding='utf-8-sig')
    nodes = [n for n in ast.walk(ast.parse(text))
             if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == symbol]
    if len(nodes) != 1:
        raise ValueError('Unique source symbol required: ' + name + ':' + symbol)
    node = nodes[0]
    return dict(file=name, sha256=sha(path), symbol=symbol, line=node.lineno,
                end_line=node.end_lineno,
                excerpt='\n'.join(text.splitlines()[node.lineno - 1:node.end_lineno]))


def main():
    # Existing snapshots, not a renewed upstream documentation search.
    refs = [
        'configs/oxe_common_training_protocol_20261025.json',
        'reports/oxe_units_time_evidence_20261020/report.json',
        'reports/oxe_conversion_lineage_20261020/report.json',
        'reports/oxe_nominal_targets_20261025/report.json',
        'reports/libero_oracle_execution_20261026/summary.json',
        'reports/libero_oracle_interpolation_20261026/result.json',
        'reports/libero_oracle_interpolation_20261026/action_semantics_audit.json',
    ]
    units = read(refs[1])
    lineage = read(refs[2])
    training = read(refs[0])
    symbols = [
        ('scripts/prepare_oxe_nominal_targets_20261025.py', 'extract'),
        ('scripts/audit_bounded_oxe_train_targets_20261025.py', 'source_arrays'),
        ('code/oxe_nominal_timing.py', 'protocol'),
        ('code/oxe_nominal_timing.py', 'layout'),
        ('code/oxe_nominal_timing.py', 'chunk'),
        ('code/oxe_nominal_dataset.py', 'NominalOXEDataset'),
        ('code/libero_action_interface.py', 'waypoint_to_native_osc'),
        ('code/libero_waypoint_execution.py', 'execute_policy_chunk'),
    ]
    excerpts = [excerpt(*pair) for pair in symbols]
    formal_root = ROOT / 'deployment/libero_original_evaluation_v1/code'
    compared = {}
    for name in ('oxe_nominal_timing.py', 'oxe_nominal_dataset.py',
                 'cartesian_action_contract.py', 'libero_action_interface.py',
                 'libero_waypoint_execution.py'):
        frozen = formal_root / name
        if not frozen.is_file():
            raise ValueError('Missing released evaluation snapshot: ' + name)
        current = ROOT / 'code' / name
        compared[name] = dict(current_sha256=sha(current), frozen_sha256=sha(frozen),
                              identical=sha(current) == sha(frozen))
    if not all(row['identical'] for row in compared.values()):
        raise ValueError('Review requires unchanged actual core versus released snapshot')
    sources = {}
    for name, row in units['sources'].items():
        sources[name] = dict(
            cached_version=row['cached_version'],
            position_field=row['position_field'],
            orientation_field=row['orientation_field'],
            nominal_hz=row['nominal_control_hz'],
            measured_timestamp_fields=row['local_timestamp_fields'],
            exact_export_units_bound=lineage['sources'][name]['position_unit_to_meters_bound_to_exact_export'],
            exact_record_spacing_verified=lineage['sources'][name]['exported_observation_spacing_seconds_verified'],
            source_to_libero_registration_verified=row['source_to_libero_tcp_and_axis_alignment_proven'])
    report = dict(
        format='oxe-action-execution-semantics-review-v1', review_complete=True,
        scope='Static current-code and archived evidence review; not data re-audit or policy validation',
        upstream_webpages_revisited=False,
        source_rows=sources, current_training_assumptions=training['assumptions'],
        established=[
            'Motion targets use observed future reached poses, not native motion commands',
            'All sixteen pose targets use a fixed input reference; they are not cumulative increments',
            'Native LIBERO OSC desired setpoint and observed reached pose are different quantities',
            'Source-Cartesian displacement is decoded directly in LIBERO world axes under unverified unit/frame assumptions',
            'Existing one-task oracle failed to preserve native success with the current5Hz execution rule',
        ],
        not_established=[
            'That reached-pose prediction is intrinsically invalid',
            'That every source actually has incorrect units or axes',
            'That source quaternion encodings are all independently verified',
            'That the first formal failure or all generalization errors have one known cause',
            'That20Hz model training or raw native-command supervision would solve transfer',
        ],
        existing_bridge_constant_right_rotation_cancellation=lineage['bridge_default_rotation_derivation'],
        source_code=excerpts, released_core_comparison=compared,
        evidence_files={name:sha(ROOT / name) for name in refs},
        new_data_payload_reads=0, models_loaded=0, training_updates=0,
        simulator_steps=0, new_jobs_submitted=0,
        labels_weights_or_execution_parameters_modified=False,
        source_validation_model_decode_performed=False,
        bulk_benchmark_ready=False, primary_behavior_qualified=False,
        next_work='Establish executable source/target unit-frame-time and reached-trajectory/control-goal contracts before a repair; no parameter search or retraining',
        teacher_scope_question='Which cross-embodiment calibration/control adaptation is permitted within the intended zero-shot LIBERO setup?',
    )
    OUT.mkdir(exist_ok=False)
    (OUT / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(review_complete=True, unchanged_released_core_files=len(compared),
                         source_count=len(sources), models_loaded=0, simulator_steps=0,
                         report=str(OUT / 'report.json')), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

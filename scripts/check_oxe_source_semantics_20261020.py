"""New causal command and planar-projection rules, not old pose/RGB probes.

Reads four fixed existing train-candidate records' command fields only; no image
decode, optimizer, model or final benchmark access. Calibration tests synthetic.
"""
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
from oxe_source_semantics import (last_command_features, command_features_before_rows,
                                  calibrated_planar_projection, planar_xyz_loss)


def rejected(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError('Ambiguous semantics accepted')


def main():
    torch.set_num_threads(1)
    assert np.array_equal(last_command_features('bridge', [1]), [1, 1])
    assert np.array_equal(last_command_features('bc_z', [1]), [-1, 1])
    assert np.array_equal(last_command_features('language_table', []), [0, 0])
    assert np.array_equal(last_command_features('bridge', [1, .3]), [0, 0])
    assert np.array_equal(last_command_features('bridge', [-.009]), [0, 0])
    expected = np.array([[0, 0], [0, 0], [0, 0], [-1, 1], [-1, 1], [1, 1]], np.float32)
    stream = np.array([0., 0., 1., 0., -1., 0.])
    assert np.array_equal(command_features_before_rows('fractal', stream), expected)
    assert np.array_equal(command_features_before_rows('fractal', [0., 1., np.nan]),
                          [[0, 0], [0, 0], [-1, 1]])
    # Identical histories yield identical inputs regardless of future switches.
    changed = stream.copy(); changed[2:] *= -1
    assert np.array_equal(command_features_before_rows('fractal', stream)[:3],
                          command_features_before_rows('fractal', changed)[:3])
    for t in range(len(stream)):
        assert np.array_equal(command_features_before_rows('fractal', stream)[t],
                              last_command_features('fractal', stream[:t]))
    rejected(lambda: last_command_features('bc_z', [.3]))
    rejected(lambda: last_command_features('bridge', [np.nan]))
    rejected(lambda: last_command_features('language_table', [1]))

    theta = np.pi / 4
    rotation = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0],
                         [-np.sin(theta), 0, np.cos(theta)]])
    declared = {'domain': 'real', 'orientation_status': 'verified_at_input',
                'source_frame': 'synthetic-source', 'tool_frame': 'synthetic-tool',
                'evidence': 'Synthetic fixture declaration only; not actual real-robot calibration'}
    projection = calibrated_planar_projection(rotation, declared)
    observed = torch.tensor([[.02, -.01]], dtype=torch.float64)
    tool_target = torch.tensor((rotation.T @ [.2, -.1, .5])[None], dtype=torch.float64)
    null_direction = torch.tensor(rotation[2], dtype=torch.float64)
    error1 = planar_xyz_loss(tool_target, observed, projection, .1)
    error2 = planar_xyz_loss(tool_target + .7 * null_direction, observed, projection, .1)
    assert error1.item() < 1e-30 and error2.item() < 1e-30
    prediction = (tool_target + torch.tensor([[.04, .03, .02]])).requires_grad_()
    planar_xyz_loss(prediction, observed, projection, .1).backward()
    assert abs(float(prediction.grad @ null_direction)) < 1e-14
    rejected(lambda: calibrated_planar_projection(None, None))
    rejected(lambda: calibrated_planar_projection(rotation, dict(declared, domain='sim')))
    rejected(lambda: calibrated_planar_projection(np.ones((3, 3)), declared))

    # Fixed first-shard choices use previous engineering records, not model error.
    from tensorflow.core.example.example_pb2 import Example
    manifest = json.loads((ROOT / 'reports/oxe_baseline_candidates_20261020/episode_candidates.json').read_text())
    inventory = json.loads((ROOT / 'reports/oxe_baseline_candidates_20261020/report.json').read_text())
    selected = {'Bridge': ('bridge_data_v2-train.tfrecord-00000-of-01024', 0),
                'Language Table': ('language_table-train.tfrecord-00000-of-01024', 0),
                'BC-Z': ('bc_z-train.tfrecord-00000-of-00512', 2),
                'Fractal': ('fractal20220817_data-train.tfrecord-00000-of-01024', 0)}
    report_rows = {}
    for source, (shard, index) in selected.items():
        row = next(r for r in manifest['episode_rows'] if (r['source'], r['shard'], r['record_index']) == (source, shard, index))
        path = Path(next(f['path'] for f in inventory['sources'][source]['cached_files'] if f['shard'] == shard))
        with path.open('rb') as file:
            file.seek(row['payload_offset']); payload = file.read(row['payload_bytes'])
        assert len(payload) == row['payload_bytes']
        example = Example(); example.ParseFromString(payload)
        f = example.features.feature
        if source == 'Bridge':
            plan = json.loads((ROOT / 'configs/bridge_demonstration_expansion_eligible_20261017.json').read_text())
            blocked = {(r['origin_file_path'], r['episode_id']) for k in
                       ('development', 'consumed_test_identity_only', 'future_test_candidates_identity_and_metadata_only')
                       for r in plan[k]}
            origin = f['episode_metadata/file_path'].bytes_list.value[0].decode()
            episode_id = f['episode_metadata/episode_id'].int64_list.value[0]
            assert (origin, episode_id) not in blocked
            values = np.array(f['steps/action'].float_list.value).reshape(-1, 7)[:, -1]
            mode = 'bridge'
        elif source == 'BC-Z':
            field = f['steps/action/future/target_close']; kind = field.WhichOneof('kind')
            values = np.array(getattr(field, kind).value).reshape(-1, 10)[:, 0]; mode = 'bc_z'
        elif source == 'Fractal':
            values = np.array(f['steps/action/gripper_closedness_action'].float_list.value); mode = 'fractal'
        else:
            values = None
            rows_count = len(f['steps/observation/rgb'].bytes_list.value)
            features = np.tile(last_command_features('language_table', []), (rows_count, 1))
        if values is not None:
            features = command_features_before_rows(mode, values)
            for t in range(len(values)):
                np.testing.assert_array_equal(features[t], last_command_features(mode, values[:t]))
        canonical_sha = hashlib.sha256(example.SerializeToString(deterministic=True)).hexdigest()
        if source in ('Language Table', 'Fractal'):
            prior = json.loads((ROOT / 'reports/core_source_fixed_record_checks.json').read_text())
            prior_key = 'language_table' if source == 'Language Table' else 'fractal'
            assert canonical_sha == prior['sources'][prior_key]['payload_sha256']
        report_rows[source] = {'shard': shard, 'record_index': index,
                               'raw_payload_sha256': hashlib.sha256(payload).hexdigest(),
                               'canonical_example_sha256': canonical_sha,
                               'rows': len(features), 'known_last_command_rows': int(features[:, 1].sum()),
                               'unknown_rows': int((features[:, 1] == 0).sum()),
                               'first_row_features': features[0].tolist(),
                               'raw_past_command_range': [float(values[:-1].min()), float(values[:-1].max())] if values is not None else None}

    features_path = Path('E:/dataset/language_table_0.1.0/0.1.0/features.json')
    feature_schema = json.loads(features_path.read_text())
    obs = feature_schema['featuresDict']['features']['steps']['sequence']['feature']['featuresDict']['features']['observation']['featuresDict']['features']
    assert set(obs) == {'effector_translation', 'effector_target_translation', 'instruction', 'rgb'}
    report = {'checks_passed': True, 'new_command_feature_contract': '[last_open_positive_command, known]',
              'causal_history_only_no_future_initialization': True,
              'unknown_and_no_gripper_not_imputed_as_open': True,
              'source_prefix_results': report_rows,
              'prior_fixed_record_identity_reused': ['Language Table record 0', 'Fractal record 0'],
              'planar_projection_and_nullspace_gradient_checked': True,
              'planar_calibration_test_scope': 'Synthetic rotations only; no real orientation calibration obtained',
              'language_table_observation_fields': sorted(obs),
              'language_table_orientation_available_in_local_feature_schema': False,
              'language_table_features_sha256': hashlib.sha256(features_path.read_bytes()).hexdigest(),
              'existing_main_training_or_loader_modified': False,
              'images_decoded': False, 'numeric_pose_rechecks': False,
              'training_updates': 0, 'model_loaded': False, 'mixed_training_ready': False}
    output = ROOT / 'reports/oxe_source_semantics_20261020.json'
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

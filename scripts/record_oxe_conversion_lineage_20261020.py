"""Record bounded upstream source review; no dataset payload or model execution.

URLs below were inspected with the web tool. These are live branch references,
not a claim that an exact historical export commit has been recovered.
The numerical check is an independent matrix derivation for the newly found
Bridge default tool rotation, not another data round-trip or fitting test.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/oxe_conversion_lineage_20261020'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rotation(v):
    x, y, z = v
    cx, cy, cz = np.cos(v)
    sx, sy, sz = np.sin(v)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rz @ ry @ rx


def bridge_rotation_derivation():
    # The public controller uses this default before its optional base rotation.
    # It is evidence for that code, not a recovered per-episode calibration.
    default = np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
    angles = [[0., 0., 0.], [.2, -.3, .4], [-.8, .4, 1.2], [.6, .9, -.7]]
    worst_left = worst_legacy = 0.
    cases = 0
    for yaw in [0., .3, -.7, 1.1]:
        d = rotation([0., 0., yaw]) @ default
        for first in angles:
            for future in angles:
                recorded0, recorded1 = rotation(first), rotation(future)
                physical0, physical1 = recorded0 @ d, recorded1 @ d
                left_recorded = recorded1 @ recorded0.T
                left_physical = physical1 @ physical0.T
                worst_left = max(worst_left, float(np.max(np.abs(left_recorded-left_physical))))
                displacement = np.array([.03, -.02, .01])
                # Legacy R_recorded.T @ displacement differs from physical TCP axes.
                legacy_tool = recorded0.T @ displacement
                physical_tool = physical0.T @ displacement
                worst_legacy = max(worst_legacy, float(np.max(np.abs(legacy_tool-physical_tool))))
                cases += 1
    assert cases == 64 and worst_left < 1e-12
    assert worst_legacy > 1e-3
    return {
        'passed': True, 'synthetic_cases': cases,
        'public_code_relationship': 'R_recorded = R_tool @ D.T; R_tool = R_recorded @ D',
        'new_source_cartesian_rotation': 'R_future @ R_current.T; constant right D cancels',
        'max_left_rotation_matrix_difference': worst_left,
        'legacy_tool_position': 'R_recorded.T @ dp = D @ (R_tool.T @ dp)',
        'max_legacy_vs_physical_tool_component_difference_source_units': worst_legacy,
        'default_rotation_constant_across_each_case': True,
        'exact_cached_episode_D_recovered': False,
        'physical_unit_or_timestamp_verified_by_this_check': False,
        'old_bridge_generalization_root_cause_established': False,
        'production_code_changed': False,
    }


def main():
    OUT.mkdir(exist_ok=False)
    previous_path = ROOT/'reports/oxe_units_time_evidence_20261020/report.json'
    previous = json.loads(previous_path.read_text())
    frozen_path = ROOT/'reports/oxe_cartesian_interface_20261020/report.json'
    frozen = json.loads(frozen_path.read_text())
    hashes = {name:digest(ROOT/'code'/name) for name in frozen['source_sha256']}
    assert hashes == frozen['source_sha256'], 'Production source changed since the existing interface check'
    role_path = ROOT/'reports/oxe_episode_roles_20261020/episode_roles.json'
    assert digest(role_path) == '149c5dece84ad636760acf8a06cc69c367323bdac5a99c7859e4dc08540ea079'

    references = {
        'tfds_rtx': {
            'url':'https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/robotics/rtx/rtx.py',
            'location':'Bridge.get_relative_dataset_location and BcZ.get_relative_dataset_location',
            'finding':'Routes to preconverted GCS data; Bridge points to bridge_data_v2/0.0.1; BC-Z points to bc_z/0.1.0.',
        },
        'tfds_importer': {
            'url':'https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/robotics/dataset_importer_builder.py',
            'location':'DatasetImporterBuilder.get_ds_builder, _info, _dataset_importer_converter_fn',
            'finding':'Reads an existing GCS TFDS builder and converts features to numpy; strips some identity keys; is not the raw acquisition converter.',
        },
        'bridge_numpy_rlds': {
            'url':'https://github.com/kpertsch/bridge_rlds_builder/blob/main/bridge_dataset/bridge_dataset_dataset_builder.py',
            'location':'_generate_examples, _split_paths, _info',
            'finding':'Same numpy_256/scripted_numpy_256 lineage and file_path/episode_id pattern as cached metadata; copies state/action arrays. No time_stamp output. Version declared 1.0.0, while cache is 0.0.1; export commit not recovered.',
        },
        'bridge_raw_numpy': {
            'url':'https://github.com/rail-berkeley/bridge_data_v2/blob/main/data_processing/bridgedata_raw_to_numpy.py',
            'location':'process_state, process_time, process_dc',
            'finding':'Copies full_state and raw time_stamp into numpy observations; shifts pre-2021-07-23 observations against actions for camera latency. This correction is not automatically reapplied to an already converted cache.',
        },
        'bridge_robot_state': {
            'url':'https://github.com/rail-berkeley/bridge_data_robot/blob/main/widowx_envs/widowx_envs/base/robot_base_env.py',
            'location':'get_full_state and current_obs',
            'finding':'Transforms controller Cartesian pose through transform2state; full_state and ROS time_stamp are separate fields.',
        },
        'bridge_robot_rotation': {
            'url':'https://github.com/rail-berkeley/bridge_data_robot/blob/main/widowx_envs/widowx_envs/utils/transformation_utils.py',
            'location':'transform2state, state2transform, eulerAnglesToRotationMatrix',
            'finding':'Recorded Euler angles describe R_tool @ D.T; reconstruction is R_euler @ D. Euler matrix order is Rz @ Ry @ Rx. D is not generally identity.',
        },
        'bridge_default_rotation': {
            'url':'https://github.com/rail-berkeley/bridge_data_robot/blob/main/widowx_envs/widowx_controller/src/widowx_controller/widowx_controller.py',
            'location':'DEFAULT_ROTATION, WidowX_Controller.__init__',
            'finding':'D is a fixed nonidentity tool rotation composed with configured normal_base_angle; historical episode configuration is not included in the cache.',
        },
        'bridge_alternate_raw_rlds': {
            'url':'https://github.com/kvablack/dlimp/blob/main/rlds_converters/bridge_dataset/bridge_dataset_dataset_builder.py',
            'location':'BridgeDataset._info and _process_example',
            'finding':'A different raw-data export: version 1.0.0, 480x640 images plus depth and raw trajectory paths; cannot substitute it for the cached 256x256 numpy export.',
        },
        'language_table_author': {
            'url':'https://github.com/google-research/language-table/blob/main/README.md',
            'location':'Data, Datasets, Environment',
            'finding':'Real dataset identified by 442226 episodes; README example points to 0.0.1, cache is 0.1.0. Public simulation configuration uses 10Hz; does not establish real exported observation spacing.',
        },
        'bcz_author': {
            'url':'https://github.com/google-research/tensor2robot/blob/master/research/bcz/model.py',
            'location':'Inference conversion of predicted axis_angle residual to target pose',
            'finding':'Adds axis-angle residual to the current axis-angle representation; this model source is not the local TFDS 1.0.0 raw export converter.',
        },
        'octo_transforms': {
            'url':'https://github.com/octo-models/octo/blob/main/octo/data/oxe/oxe_standardization_transforms.py',
            'location':'rt1_dataset_transform, bc_z_dataset_transform, bridge_dataset_transform',
            'finding':'RT-1 reached quaternion is passed through as proprio, not independently converted or documented. Bridge transform explicitly distinguishes its newer copy from official OXE. Downstream training code is not independent proof of cached quaternion order.',
        },
    }
    sources = {}
    for source, old in previous['sources'].items():
        folder = ROOT/'training_cache/oxe_core'/old['cache']
        info = json.loads((folder/'dataset_info.json').read_text())
        assert digest(folder/'dataset_info.json') == old['metadata_sha256']
        assert digest(folder/'features.json') == old['features_sha256']
        sources[source] = {
            'cached_dataset_name': info['name'], 'cached_version': info['version'],
            'cached_builder_module': info.get('moduleName'),
            'metadata_sha256': old['metadata_sha256'], 'features_sha256': old['features_sha256'],
            'position_unit_to_meters_bound_to_exact_export': False,
            'exported_observation_spacing_seconds_verified': False,
            'nominal_controller_hz_only': old['nominal_control_hz'],
            'local_per_frame_timestamp_available': bool(old['local_timestamp_fields']),
            'formal_training_admitted_by_this_review': False,
        }
    sources['Bridge'].update({
        'converter_family_support': 'Matching original paths, metadata construction and direct array copies; exact version/commit remains unbound',
        'orientation_support': 'Public collector records default-adjusted XYZ Euler angles; recorded axes are not automatically physical TCP axes',
        'time_recovery_route': 'Matching original numpy observation time_stamp or raw obs_dict.pkl; field omitted by the inspected numpy-to-RLDS converter',
        'do_not_reapply_camera_latency_shift': True,
        'metadata_flags_caution': 'Inspected numpy converter tests image keys against the observation list rather than its dictionary; consistent with previously observed false flags. No new payloads read and no global correction inferred.',
    })
    sources['Language Table'].update({
        'orientation_support': 'No orientation field; retain planar XY mask and unknown gripper input',
        'missing_evidence': 'Real exporter version binding, XY unit/frame and real observation timing; simulation constants are insufficient',
    })
    sources['BC-Z'].update({
        'missing_evidence': 'Cached google3 TFDS 1.0.0 export code or release specification for xyz units, rotation vector and observation timing',
        'catalog_0_1_0_is_not_cached_1_0_0_proof': True,
    })
    sources['Fractal'].update({
        'missing_evidence': 'Cached exporter specification for meters and xyzw/wxyz orientation order plus observation timestamp/downsampling semantics',
        'downstream_proprio_pass_through_proves_quaternion_order': False,
    })
    report = {
        'review_complete': True, 'physical_contracts_complete': False,
        'scope': 'Bounded upstream conversion-lineage review and new Bridge matrix derivation',
        'source_references': references, 'references_inspected_date': '2026-10-20',
        'references_are_live_branch_urls_not_recovered_export_commits': True,
        'local_source_snapshots_downloaded': False,
        'previous_units_time_report_sha256': digest(previous_path),
        'frozen_role_manifest_sha256': digest(role_path),
        'production_sources_unchanged_against_existing_interface_report': True,
        'production_source_sha256': hashes,
        'sources': sources, 'bridge_default_rotation_derivation': bridge_rotation_derivation(),
        'window_manifest_created': False, 'resampling_or_interpolation_added': False,
        'production_normalization_or_pose_codec_changed': False,
        'source_payloads_read': False, 'training_or_inference_run': False,
        'old_bridge_generalization_root_cause_established': False,
        'training_ready': False,
        'bounded_next_evidence': [
            'Use matching raw/numpy timestamps and release metadata if already available; otherwise obtain dataset-maintainer documentation for the exact cached schema.',
            'Verify position units and Fractal quaternion ordering independently before freezing normalization or executable full-pose labels.',
            'Decide and document a target-time protocol only after export timing is supported; do not silently treat nominal control frequency as measured timestamps.',
            'Then enumerate eligible windows within the frozen episode roles; do not repartition or launch another fit to compensate for missing provenance.',
        ],
    }
    (OUT/'report.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'review_complete':True, 'physical_contracts_complete':False,
                      'bridge_matrix_derivation':report['bridge_default_rotation_derivation'],
                      'training_ready':False}, indent=2))


if __name__ == '__main__':
    main()

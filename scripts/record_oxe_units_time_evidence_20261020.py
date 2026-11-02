"""Bind the documentary timing review to the exact local feature schemas.

No inference of physical units from numeric scale or of record spacing from a
policy's loop frequency. No model, source payload, normalization or resampling.
"""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/oxe_units_time_evidence_20261020'


def leaves(node,prefix=''):
    if 'featuresDict' in node:
        result={}
        for name,child in node['featuresDict']['features'].items():
            result.update(leaves(child,prefix+'/'+name))
        return result
    if 'sequence' in node:
        return leaves(node['sequence']['feature'],prefix)
    return {prefix:node}


def main():
    OUT.mkdir(exist_ok=False)
    definitions={
        'Bridge':{'cache':'bridge_v2','nominal_control_hz':5,
                  'nominal_time_evidence':'https://arxiv.org/pdf/2308.12952',
                  'evidence_location':'Section 3.1, system setup; control frequency',
                  'extra_time_evidence':'https://raw.githubusercontent.com/rail-berkeley/bridge_data_v2/main/experiments/eval.py',
                  'extra_evidence_location':'STEP_DURATION = 0.2; deployment script, not per-record timestamps',
                  'position_field':'/steps/observation/state','orientation_field':'/steps/observation/state',
                  'recorded_frame_documentation':'Local state has 7 entries without coordinate/unit descriptions; original controller lineage still needs binding'},
        'Language Table':{'cache':'language_table_real','nominal_control_hz':5,
                          'nominal_time_evidence':'https://arxiv.org/pdf/2210.06407',
                          'evidence_location':'Section V: real collection batched to 5Hz; Appendix C distinguishes 10Hz blocking simulation',
                          'position_field':'/steps/observation/effector_translation','orientation_field':None,
                          'recorded_frame_documentation':'Two-dimensional Cartesian position; no tool orientation or real-robot axis calibration in local schema'},
        'BC-Z':{'cache':'bc_z','nominal_control_hz':10,
                'nominal_time_evidence':'https://proceedings.mlr.press/v164/jang22a/jang22a.pdf',
                'evidence_location':'Section 4 system setup: 10Hz non-real-time teleoperation loop',
                'position_field':'/steps/observation/present/xyz','orientation_field':'/steps/observation/present/axis_angle',
                'recorded_frame_documentation':'Position described as robot frame; axis-angle rotation; position description contains a wording error (axis-angle for xyz)',
                'catalog_evidence':'https://www.tensorflow.org/datasets/catalog/bc_z',
                'catalog_documented_version':'0.1.0','catalog_version_equivalence_to_cache_established':False},
        'Fractal':{'cache':'rt1_fractal','nominal_control_hz':3,
                   'nominal_time_evidence':'https://robotics-transformer1.github.io/',
                   'evidence_location':'Approach: RT-1 closed-loop policy commands at 3Hz; this alone is not a timestamp contract for Fractal export',
                   'position_field':'/steps/observation/base_pose_tool_reached','orientation_field':'/steps/observation/base_pose_tool_reached',
                   'recorded_frame_documentation':'Local metadata identifies base-relative position+quaternion; component ordering and physical units are not explicitly specified',
                   'catalog_evidence':'https://www.tensorflow.org/datasets/catalog/fractal20220817_data'}
    }
    result={}
    for source,definition in definitions.items():
        folder=ROOT/'training_cache/oxe_core'/definition['cache']
        metadata_path,features_path=folder/'dataset_info.json',folder/'features.json'
        metadata=json.loads(metadata_path.read_text());fields=leaves(json.loads(features_path.read_text()))
        time_fields=[key for key in fields if any(token in key.lower() for token in ('timestamp','time_stamp','time_us','time_ms','elapsed_time'))]
        rate=definition['nominal_control_hz']
        position=fields[definition['position_field']]
        orientation=fields[definition['orientation_field']] if definition['orientation_field'] else None
        result[source]={**definition,'cached_dataset_name':metadata['name'],'cached_version':metadata['version'],
                        'metadata_sha256':hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
                        'features_sha256':hashlib.sha256(features_path.read_bytes()).hexdigest(),
                        'local_timestamp_fields':time_fields,
                        'position_field_metadata':position,'orientation_field_metadata':orientation,
                        'position_unit_explicitly_stated_in_local_metadata':False,
                        'position_unit_to_meters_bound_to_cached_export':False,
                        'nominal_16_future_intervals_seconds_illustration_only':16/rate,
                        'nominal_rate_verified_as_actual_record_spacing':False,
                        'all_records_regularly_spaced_proven':False,
                        'source_to_libero_tcp_and_axis_alignment_proven':False}
        assert not time_fields,source
    report={'complete':True,'scope':'Documented nominal loop rates and exact cached metadata; no empirical timestamp or unit calibration',
            'sources':result,'common_16_row_time_horizon_established':False,
            'unit_conversion_factors_changed':False,'normalization_statistics_fitted':False,
            'interpolation_or_resampling_added':False,'source_payloads_read':False,
            'model_or_training_run':False,'training_ready':False,
            'required_next_evidence':['Trace cached-version conversion lineage for position units and orientation conventions',
                                      'Bind nominal controller rates to exported observation spacing or obtain raw timestamps',
                                      'Define a common target time grid only after the source rate binding; no silent interpolation']}
    (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({s:{'version':r['cached_version'],'nominal_hz':r['nominal_control_hz'],
                         'timestamp_fields':r['local_timestamp_fields'],'export_spacing_verified':False}
                      for s,r in result.items()},indent=2))


if __name__=='__main__':main()

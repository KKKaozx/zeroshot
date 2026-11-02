"""Verify engineering evidence or report the fixed historical-control contrast."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

BUNDLE=Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text())


def recompute(prediction, target, rows):
    import numpy as np
    # Execute only these two existing pure metric functions, without importing
    # the training runner, encoders, torch or transformers.
    namespace = dict(np=np, POSITION_CM=10.)
    for filename, name in [('evaluate_bridge_endpoint_metrics.py', 'pose_errors'),
                           ('check_bridge_batch_control.py', 'metrics')]:
        tree=ast.parse((BUNDLE/'frozen/bridge_growth128_v1'/filename).read_text())
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
        exec(compile(ast.Module(body=[node],type_ignores=[]),filename,'exec'),namespace)
    return namespace['metrics'](prediction,target,rows)


def verify(directory, stage):
    import numpy as np
    r=read(directory/'report.json')
    old=read(BUNDLE/'clip207558_report.json')
    assert r['experiment']=='bridge_dino_only_v1' and r['stage']==stage
    assert r['execution_checks_passed'] and not r['local_bounded']
    assert r['contract_sha256']==hashlib.sha256((BUNDLE/'contract.json').read_bytes()).hexdigest()
    assert r['input_identity_sha256']==old['input_identity_sha256']
    assert r['input_row_checks']==old['input_row_checks']
    assert r['episode_payload_checks']==old['episode_payload_checks']
    for key in ['initial_shared_sha256','augmented_initial_sha256','torch_version','gpu',
                'trainable_parameters','initialization_seed','sampling_steps','sampling_seeds']:
        assert r[key]==old[key], key
    assert r['full_input_contract_passed'] and r['other_frozen_unchanged']
    assert r['frozen_visual_and_text_unchanged'] and not r['gripper_trained']
    assert r['dino_initialization_preserves_cpu_cuda_rng'] and r['language_tokens_exact_original207558']
    assert r['peak_reserved_gib']<7.5 and r['estimated_formal_seconds']<9000
    if stage=='preflight':
        assert r['branch_restore_passed'] and r['optimization_operations']==16
        assert r['final_update']==8 and not r['development_scored']
        print('BRIDGE DINO-ONLY PREFLIGHT VERIFIED; no performance verdict or automatic formal submission')
        print(json.dumps(dict(peak_reserved_gib=r['peak_reserved_gib'],
            warm_update_seconds=r['warm_update_seconds'], estimated_formal_seconds=r['estimated_formal_seconds'],
            initial_parameters_match207558=True, all1876_numeric_inputs_match207558=True,
            language_tokens_exact=True),indent=2))
    else:
        assert r['optimization_operations']==8192 and r['final_update']==8192 and r['development_scored']
        current=np.load(directory/'reduced_u8192_development.npz')
        reference=np.load(BUNDLE/'clip207558_development.npz')
        assert np.array_equal(current['targets'],reference['targets'])
        assert np.array_equal(current['row_indices'],reference['row_indices'])
        assert current['predictions'].shape==reference['predictions'].shape==(3,40,16,8)
        oldmetrics=old['runs'][0]['final_development']
        newmetrics=r['runs'][0]['final_development']
        keys=['path_position_cm','path_rotation_deg','endpoint_position_cm','endpoint_rotation_deg',
              'episode_mean_path_position_cm','episode_mean_path_rotation_deg']
        rows=read(BUNDLE/'frozen/bridge_coverage_replication_v1/rows.json')['development']
        metric_verification={}
        for label, saved, values in [('clip',reference,oldmetrics),('dino',current,newmetrics)]:
            computed=recompute(saved['predictions'],saved['targets'],rows)
            differences={k:abs(computed[k]-values[k]) for k in keys}
            for key,difference in differences.items():
                assert difference <= (2e-4 if key.endswith('deg') else 2e-5), (label,key,difference)
            metric_verification[label]=differences
        oldepisodes={e['episode']:e for e in oldmetrics['per_episode']}
        newepisodes={e['episode']:e for e in newmetrics['per_episode']}
        assert oldepisodes.keys()==newepisodes.keys()
        contrast=dict(clip207558={k:oldmetrics[k] for k in keys}, dino={k:newmetrics[k] for k in keys},
            dino_minus_clip={k:newmetrics[k]-oldmetrics[k] for k in keys},
            negative_error_difference_favors_dino=True,
            scope='Same40 reused development windows/10 episodes and3 sampling seeds; one training seed, historical same-GPU-model control. Not independent final confirmation or task success.',
            full_metadata_and_actual_inputs_matched=True,
            comparison_to_4_77cm_heldout_admitted=False,
            main_oxe_model_replaced=False, automatic_following_action=False)
        contrast['independent_saved_prediction_metric_verification']=metric_verification
        contrast['per_episode_dino_minus_clip']=[dict(episode=e,
            path_position_cm=newepisodes[e]['path_position_cm']-oldepisodes[e]['path_position_cm'],
            path_rotation_deg=newepisodes[e]['path_rotation_deg']-oldepisodes[e]['path_rotation_deg']) for e in oldepisodes]
        (directory/'backbone_comparison.json').write_text(json.dumps(contrast,indent=2)+'\n')
        print('BRIDGE DINO-ONLY OUTPUT VERIFIED; offline development contrast, not robot success')
        print(json.dumps(contrast,indent=2))
    return r


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('directory',type=Path)
    ap.add_argument('--stage',choices=['preflight','fit'],required=True)
    args=ap.parse_args()
    verify(args.directory,args.stage)

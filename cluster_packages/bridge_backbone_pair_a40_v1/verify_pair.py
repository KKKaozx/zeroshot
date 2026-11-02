"""Independent saved-prediction metrics and same-allocation pair checks."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from admission import admit,read,sha

BUNDLE=Path(__file__).resolve().parent
KEYS=['path_position_cm','path_rotation_deg','endpoint_position_cm','endpoint_rotation_deg',
      'episode_mean_path_position_cm','episode_mean_path_rotation_deg']


def metrics(prediction,target,rows):
    import numpy as np
    namespace=dict(np=np,POSITION_CM=10.)
    for filename,name in [('evaluate_bridge_endpoint_metrics.py','pose_errors'),
                          ('check_bridge_batch_control.py','metrics')]:
        tree=ast.parse((BUNDLE/'frozen/bridge_growth128_v1'/filename).read_text())
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
        exec(compile(ast.Module(body=[node],type_ignores=[]),filename,'exec'),namespace)
    return namespace['metrics'](prediction,target,rows)


def verify_arm(directory,arm):
    import numpy as np
    r=read(directory/'report.json');old=read(BUNDLE/'clip207558_report.json')
    assert r['experiment']=='bridge_backbone_pair_a40_v1' and r['arm']==arm
    assert r['stage']=='fit' and r['execution_checks_passed'] and not r['local_bounded']
    assert r['pair_contract_sha256']==sha(BUNDLE/'contract.json')
    for key in ['initial_shared_sha256','augmented_initial_sha256','input_identity_sha256','input_row_checks',
                'episode_payload_checks','retained_state_features_sha256','trainable_parameters',
                'initialization_seed','torch_version','sampling_steps','sampling_seeds']:
        assert r[key]==old[key],key
    assert r['gpu']=='NVIDIA A40' and r['full_input_contract_passed']
    assert r['other_frozen_unchanged'] and r['frozen_visual_and_text_unchanged']
    assert not r['gripper_trained'] and r['visual_replacement_preserves_cpu_cuda_rng']
    assert r['language_tokens_exact_original207558'] and r['full_token_cache_reused']
    assert r['new_frames_encoded']==0 and r['optimization_operations']==8192 and r['final_update']==8192
    assert r['development_scored'] and r['actual_arm']==arm
    saved=np.load(directory/'reduced_u8192_development.npz')
    reference=np.load(BUNDLE/'clip207558_development.npz')
    assert saved['predictions'].shape==(3,40,16,8)
    assert np.array_equal(saved['targets'],reference['targets'])
    assert np.array_equal(saved['row_indices'],reference['row_indices'])
    rows=read(BUNDLE/'frozen/bridge_coverage_replication_v1/rows.json')['development']
    computed=metrics(saved['predictions'],saved['targets'],rows)
    for key in KEYS:
        assert abs(computed[key]-r['runs'][0]['final_development'][key]) <= (2e-4 if key.endswith('deg') else 2e-5),key
    return r,computed


def verify_pair(directory):
    clip,cm=verify_arm(directory/'clip','clip');dino,dm=verify_arm(directory/'dino','dino')
    for key in ['gpu','torch_version','same_allocation','visible_device_environment','pair_contract_sha256',
                'initial_shared_sha256','augmented_initial_sha256','input_identity_sha256','sampling_seeds']:
        assert clip[key]==dino[key],key
    assert clip['same_allocation'] is not None, 'A common allocated job identity is required'
    c={e['episode']:e for e in cm['per_episode']};d={e['episode']:e for e in dm['per_episode']}
    assert c.keys()==d.keys()
    result=dict(pair_verified=True,scores={'clip':{k:cm[k] for k in KEYS},'dino':{k:dm[k] for k in KEYS}},
        dino_minus_clip={k:dm[k]-cm[k] for k in KEYS},negative_error_difference_favors_dino=True,
        static={k:cm[k] for k in ['static_path_position_cm','static_path_rotation_deg',
            'static_endpoint_position_cm','static_endpoint_rotation_deg']},
        per_episode=[dict(episode=e,windows=c[e]['windows'],
            position_cm_dino_minus_clip=d[e]['path_position_cm']-c[e]['path_position_cm'],
            rotation_deg_dino_minus_clip=d[e]['path_rotation_deg']-c[e]['path_rotation_deg']) for e in c],
        same_allocation=clip['same_allocation'],gpu='NVIDIA A40',training_seed=42,
        updates_per_arm=8192,development_windows=40,development_episodes=10,
        source_feature_cache_gpu='NVIDIA RTX A6000 for both encoders',
        scope='One training seed on one allocated GPU; reused dev40; offline pose, frozen gripper. Not final LIBERO success or three training seeds.',
        automatic_following_action=False,main_oxe_model_replaced=False)
    (directory/'pair_comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    print('BRIDGE A40 BACKBONE PAIR VERIFIED; offline development pose only')
    print(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('directory',type=Path)
    verify_pair(ap.parse_args().directory)

"""Recompute saved metrics and compare with reused positive x0, never retrain."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from checks import HERE,read,sha


def verify(directory,root):
    base=root/'bridge_backbone_pair_a40_v1';sys.path.insert(0,str(base))
    from verify_pair import metrics,KEYS
    reference=root/'runs/bridge-backbone-pair-a40-229116/dino'
    old=read(reference/'report.json');r=read(directory/'report.json')
    assert r['experiment']=='bridge_positive_dino_epsilon_control_v1' and r['prediction_type']=='epsilon'
    assert r['execution_checks_passed'] and r['final_update']==r['optimization_operations']==8192
    assert r['contract_sha256']==sha(HERE/'protocol.json') and not r['gripper_trained']
    assert r['original_baseline_replay_passed'] and r['first_x0_reference_passed']
    assert r['positive_saved_replay']['prediction_max_abs']==r['positive_saved_replay']['context_max_abs']==0
    assert r['other_frozen_unchanged'] and r['frozen_visual_and_text_unchanged']
    for key in ['initial_shared_sha256','augmented_initial_sha256','input_identity_sha256','input_row_checks',
                'episode_payload_checks','retained_state_features_sha256','trainable_parameters','initialization_seed',
                'torch_version','gpu','sampling_steps','sampling_seeds','token_cache_key_sha256']:
        assert r[key]==old[key],key
    assert r['new_frames_encoded']==0 and r['full_token_cache_reused'] and r['visual_replacement_preserves_cpu_cuda_rng']
    history=[]
    for name in ['common_updates.jsonl','reduced_updates.jsonl']:
        history += [json.loads(line) for line in (directory/name).read_text().splitlines()]
    assert len(history)==8192 and [row['update'] for row in history]==list(range(1,8193))
    assert all(row['learning_rate']==(1e-4 if row['update']<=4096 else 1e-5) for row in history)
    assert all(np.isfinite(row['loss']) and np.isfinite(row['global_unclipped_gradient_norm']) for row in history)
    ck=r['checkpoints'][0];assert sha(directory/ck['path'])==ck['sha256'] and ck['roundtrip_exact']
    plan=read(base/'frozen/bridge_coverage_replication_v1/rows.json');scores={}
    for split in ['train','development']:
        filename=f'reduced_u8192_{split}.npz'
        with np.load(directory/filename) as current,np.load(reference/filename) as prior:
            assert np.array_equal(current['targets'],prior['targets']) and np.array_equal(current['row_indices'],prior['row_indices'])
            assert current['predictions'].shape==prior['predictions'].shape
            scored=metrics(current['predictions'],current['targets'],plan[split])
            previous=metrics(prior['predictions'],prior['targets'],plan[split])
        recorded=r['runs'][0]['final_train' if split=='train' else 'final_development']
        for key in KEYS:assert abs(scored[key]-recorded[key])<=(2e-4 if key.endswith('deg') else 2e-5),key
        scores[split]={'epsilon':scored,'x0':previous}
    dev=scores['development'];a={x['episode']:x for x in dev['epsilon']['per_episode']};b={x['episode']:x for x in dev['x0']['per_episode']}
    assert a.keys()==b.keys() and len(a)==10
    differences=np.asarray([[a[e][k]-b[e][k] for k in KEYS[:4]] for e in sorted(a)])
    rng=np.random.default_rng(20261028);resamples=rng.integers(0,10,size=(10000,10))
    intervals=np.quantile(differences[resamples].mean(axis=1),[.025,.975],axis=0)
    result={'execution_and_output_verified':True,'baseline_job':229116,'candidate_job':r['gpu_allocation'],
        'scores':{name:{k:v[k] for k in KEYS} for name,v in dev.items()},
        'epsilon_minus_x0':{k:dev['epsilon'][k]-dev['x0'][k] for k in KEYS},
        'static':{k:dev['epsilon'][k] for k in ['static_path_position_cm','static_path_rotation_deg','static_endpoint_position_cm','static_endpoint_rotation_deg']},
        'episode_equal_difference_bootstrap95':{key:intervals[:,i].tolist() for i,key in enumerate(KEYS[:4])},
        'episode_equal_mean_difference':{key:float(differences[:,i].mean()) for i,key in enumerate(KEYS[:4])},
        'scope':'Conditional target effect under fixed positive DINO+9Dpose recipe; reuseddev40/10episodes, one trainingseed; scene bootstrap not seed robustness',
        'same_feature_cache_and_initial_parameters':True,'historical_baseline_replay_exact':True,
        'new_x0_training':False,'training_updates':8192,'automatic_following_action':False,'main_oxe_model_replaced':False}
    (directory/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    print('DINO TARGET CONTROL OUTPUT VERIFIED; conditional development comparison only')
    print(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('directory',type=Path);ap.add_argument('--root',type=Path,default=Path('/projects/Zeroshot'))
    a=ap.parse_args();verify(a.directory,a.root)

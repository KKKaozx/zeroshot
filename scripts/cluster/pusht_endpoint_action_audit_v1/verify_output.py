"""Recompute a completed read-only audit from its small saved arrays."""
import argparse
import json
from pathlib import Path
import numpy as np
from resource_probe import sha
from audit_math import reference_indices, select_windows, summarize, summarize_epsilon


def equal(a,b):
    if isinstance(a,dict): return set(a)==set(b) and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,list): return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    if isinstance(a,float): return bool(np.isclose(a,b,rtol=1e-12,atol=1e-12))
    return a==b


def validate(out,bundle):
    protocol = json.loads((bundle/'protocol.json').read_text())
    for name,digest in protocol['bundle_sources_sha256'].items():
        if sha(bundle/name)!=digest: raise ValueError('Bundle changed: '+name)
    r = json.loads((out/'report.json').read_text())
    if not r['complete'] or r['errors'] or r['training_updates'] or r['environment_steps'] or r['policy_rollouts'] or r['final50_accessed'] or r['automatic_submission']:
        raise ValueError('Incomplete or out-of-scope output')
    if r['protocol_sha256']!=sha(bundle/'protocol.json'): raise ValueError('Protocol differs')
    if sha(out/'inputs.npz')!=r['input_sha256'] or sha(out/'window_manifest.json')!=r['manifest_sha256']:
        raise ValueError('Input artifacts changed')
    with np.load(out/'inputs.npz',allow_pickle=False) as f: x={k:f[k] for k in f.files}
    manifest = json.loads((out/'window_manifest.json').read_text())
    rows = manifest['windows']
    if len(rows)!=64 or manifest['execution_slice']!=[1,9]: raise ValueError('Wrong row/slice count')
    expected_info,expected_rows=select_windows(x['episode_ends'],protocol['selected_episode_ids'],reference_indices)
    if expected_rows!=rows or any(not np.array_equal(x[k],v) for k,v in expected_info.items()):
        raise ValueError('Actual frame/window selection differs from fixed author rule')
    if (x['frames'].shape!=(64,16) or x['raw_images'].shape!=(64,2,96,96,3)
            or x['agent_pos'].shape!=(64,2,2) or x['targets'].shape!=(64,16,2)):
        raise ValueError('Wrong input shapes')
    import hashlib
    if hashlib.sha256(np.ascontiguousarray(x['raw_images']).tobytes()).hexdigest()!=manifest['selected_raw_pixels_sha256']:
        raise ValueError('Pixel identity changed')
    for i,row in enumerate(rows):
        role,ep=row['role'],row['episode_id']
        if ep not in protocol['selected_episode_ids'][role] or x['roles'][i]!=role or x['episode_ids'][i]!=ep:
            raise ValueError('Episode role differs')
        if row['frame_ids']!=x['frames'][i].tolist() or row['observation_frame_ids']!=x['frames'][i,:2].tolist() or row['valid_mask']!=x['valid'][i].tolist():
            raise ValueError('Frame metadata differs')
        a,b,c,d=row['author_row']
        expected=np.pad(np.arange(a,b),(c,16-d),mode='edge')
        if not np.array_equal(expected,x['frames'][i]) or not np.array_equal((np.arange(16)>=c)&(np.arange(16)<d),x['valid'][i]):
            raise ValueError('Padding differs')
        n=row['rows_in_episode']
        if n<3 or row['row_rank'] not in (n//3,2*n//3): raise ValueError('Selection rule differs')
    summaries, common={},None
    if set(r['groups'])!={'native212158','project215999'}: raise ValueError('Wrong models')
    for name,row in r['groups'].items():
        if not row['state_unchanged'] or not row['all_frozen'] or row['checkpoint_sha256']!=protocol['checkpoints'][name]['sha256']:
            raise ValueError('Endpoint/freeze audit failed')
        path=out/(name+'_predictions.npz')
        if sha(path)!=row['predictions_sha256']: raise ValueError('Predictions changed')
        with np.load(path,allow_pickle=False) as f:
            if f['predictions'].shape!=(2,64,16,2): raise ValueError('Wrong seed count')
            scores=summarize(f['predictions'],x['targets'],x['valid'],x['episode_ids'],x['roles'])
            epsilon=summarize_epsilon(f['epsilon_mse'],x['episode_ids'],x['roles'])
        if not equal(scores,row['action_metrics']) or not equal(epsilon,row['epsilon_mse_all16']): raise ValueError('Metric mismatch')
        traces=row['common_noise_traces']
        if len(traces)!=32 or [(v['sampling_seed'],v['batch_index']) for v in traces] != [(s,b) for s in protocol['sampling_seeds'] for b in range(16)]:
            raise ValueError('Missing sampling traces')
        if common is None: common=traces
        elif traces!=common: raise ValueError('Noise streams differ')
        summaries[name]={role:{sl:scores[role][sl]['episode_equal_mean_pixels'] for sl in ('full16','executed8')} for role in scores}
    static=np.broadcast_to(x['agent_pos'][:,-1:,:],(64,16,2)).copy()[None]
    mean=np.broadcast_to(x['train_action_mean'],(64,16,2)).copy()[None]
    expected={name:summarize(p,x['targets'],x['valid'],x['episode_ids'],x['roles']) for name,p in (('no_motion',static),('train_mean',mean))}
    if not equal(expected,r['baselines']): raise ValueError('Baseline mismatch')
    clip=r['clip']
    if not clip['worker']['all_frozen'] or not clip['worker']['weights_unchanged'] or clip['selected_reference_max_abs']>1e-4:
        raise ValueError('CLIP qualification failed')
    for name,v in expected.items():
        summaries[name]={role:{sl:v[role][sl]['episode_equal_mean_pixels'] for sl in ('full16','executed8')} for role in v}
    return dict(output_verified=True,action_errors_pixels=summaries,scope=r['scope'],
        training_updates=0,policy_improvement_verdict=None,automatic_following_action=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path);a=p.parse_args()
    print('FROZEN ENDPOINT ACTION OUTPUT VERIFIED; diagnostic only')
    print(json.dumps(validate(a.output,Path(__file__).resolve().parent),indent=2))

"""Endpoint/Adam/budget readback and saved-prediction metric recomputation."""
import argparse
from pathlib import Path
import json


def verify(out):
    import numpy as np
    import torch
    from train_pair import read,state_sha,sha,HERE
    from pair_metrics import summarize,decision
    from pose_core import CachedPoseCore,freeze_adapter_history
    from episode_relative_pose_state import ABSOLUTE_FORMAT,RELATIVE_FORMAT
    out=Path(out);r=read(out/'report.json');rows=read(HERE/'rows.json');rows=rows['train']+rows['development']
    if not r['execution_complete'] or r['total_new_pose_updates']!=4096:raise ValueError('Pair incomplete')
    source=torch.load(read(out/'admission.json')['source'],map_location='cpu',weights_only=False)['trainable_state_dict']
    arms={}; initial=None; parameters=None
    with np.load(out/'predictions.npz',allow_pickle=False) as z:
        for mode,fmt in [('absolute',ABSOLUTE_FORMAT),('relative',RELATIVE_FORMAT)]:
            checkpoint=torch.load(out/mode/'policy.pt',map_location='cpu',weights_only=False)
            latest=torch.load(out/mode/'latest.pt',map_location='cpu',weights_only=False)
            if parameters is None:
                core=CachedPoseCore(checkpoint['config']).restore(checkpoint['state']);parameters=len(freeze_adapter_history(core));del core
            if (checkpoint['format']!='episode-relative-pose-policy-v2' or checkpoint['mode']!=mode
                    or checkpoint['state_feature_format']!=fmt or checkpoint['additional_updates']!=2048
                    or latest['format']!='episode-relative-pair-training-state-v2'
                    or latest['update']!=2048 or len(latest['history'])!=2048
                    or state_sha(checkpoint['state'])!=state_sha(latest['state'])
                    or len(latest['optimizer']['state'])!=parameters
                    or any(int(v['step'])!=2048 for v in latest['optimizer']['state'].values())
                    or any(not torch.isfinite(v).all() for v in checkpoint['state'].values())
                    or any(not torch.isfinite(v[k]).all() for v in latest['optimizer']['state'].values()
                           for k in ('exp_avg','exp_avg_sq'))):
                raise ValueError('State/Adam/budget readback differs')
            if any(not torch.equal(v,source[n]) for n,v in checkpoint['state'].items()
                   if n.startswith(('adapter.','history_projection.'))):
                raise ValueError('Frozen Adapter/history differs from source')
            if initial is not None and initial!=checkpoint['initial_parameters_sha256']:raise ValueError('Pair start differs')
            initial=checkpoint['initial_parameters_sha256'];roles={}
            for role in ('train','development'):
                ids=z[role+'_indices'];roles[role]=summarize(z[mode+'_'+role],z[role+'_targets'],[rows[i] for i in ids])
            if roles!=r['arms'][mode]:raise ValueError('Saved metric differs')
            arms[mode]=roles
            if sha(out/mode/'policy.pt')!=read(out/mode/'report.json')['endpoint_sha256']:
                raise ValueError('Endpoint byte identity differs')
            del checkpoint,latest
    if any(r[k]!=v for k,v in decision(arms,r['baseline']).items()):raise ValueError('Decision differs')
    return {'verified':True,'both_complete_pose_and_adam_states_verified':True,
        'frozen_adapter_history_verified_against_source':True,
        'saved_prediction_metrics_recomputed':True,'new_sampling_training_or_rollouts':0,
        'original_development_reused':True,'formal_admission':False}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);a=p.parse_args();print(json.dumps(verify(a.output),indent=2))

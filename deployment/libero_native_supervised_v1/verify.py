"""Verify the fixed endpoint/Adam and saved offline predictions; no new sampling."""
import argparse
import json
from pathlib import Path


def verify(out):
    import numpy as np
    import torch
    from train import read,state_sha,sha,summarize,HERE
    from native_policy import NativeActionPolicy
    out=Path(out);r=read(out/'report.json')
    endpoint=torch.load(out/'policy.pt',map_location='cpu',weights_only=False)
    latest=torch.load(out/'latest.pt',map_location='cpu',weights_only=False)
    if (endpoint['format']!='libero90-native-supervised-policy-v1' or endpoint['update']!=4096
            or not endpoint['target_supervised'] or endpoint['zero_shot'] or endpoint['formal_matrix_admitted']
            or latest['update']!=4096 or len(latest['history'])!=4096
            or r['optimizer_updates']!=4096 or not r['execution_complete']
            or state_sha(endpoint['state'])!=state_sha(latest['state'])
            or sha(out/'policy.pt')!=r['policy_sha256']):raise ValueError('Budget/endpoint identity differs')
    model=NativeActionPolicy(endpoint['model_config']);model.load_state_dict(endpoint['state'],strict=True)
    if (len(latest['optimizer']['state'])!=len(list(model.parameters()))
            or any(int(v['step'])!=4096 for v in latest['optimizer']['state'].values())
            or any(not torch.isfinite(v).all() for v in endpoint['state'].values())
            or any(not torch.isfinite(v[k]).all() for v in latest['optimizer']['state'].values() for k in ('exp_avg','exp_avg_sq'))):
        raise ValueError('Adam completeness/finiteness differs')
    with np.load(out/'predictions.npz',allow_pickle=False) as z:
        for role in ('train','development'):
            value=summarize(z[role+'_predictions'],z[role+'_targets'],z[role+'_mask'],z[role+'_previous'],endpoint['normalizer'])
            if value!=r['metrics'][role]:raise ValueError('Saved offline metric differs')
    return {'verified':True,'full_policy_and_adam_readback':True,'saved_metrics_recomputed':True,
        'optimizer_updates':4096,'new_sampling':0,'target_supervised':True,'zero_shot':False,
        'learned_task_success_proven':False,'automatic_following_action':False}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);a=p.parse_args();print(json.dumps(verify(a.output),indent=2))

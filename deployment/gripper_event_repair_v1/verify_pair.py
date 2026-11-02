"""Readback of both fixed head endpoints, with no source pose model calls."""
import argparse
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'dino_gripper_sidecar_v1'
sys.path[:0]=[str(HERE/'code'),str(BASE/'code'),str(BASE)]


def verify(out):
    import numpy as np
    import torch
    from train_event_pair import read,summarize,decide
    from dino_gripper_sidecar import GripperSidecar
    out=Path(out); arms={}
    with np.load(BASE/'inputs.npz',allow_pickle=False) as z: arrays={k:z[k] for k in z.files}
    for mode in ('state_control','event_aware'):
        folder=out/mode; r=read(folder/'report.json')
        head=torch.load(folder/'gripper.pt',map_location='cpu',weights_only=False)
        saved=torch.load(folder/'latest.pt',map_location='cpu',weights_only=False)
        checker=GripperSidecar(); checker.load_state_dict(head['state'],strict=True)
        if (saved['additional_updates']!=2048 or len(saved['history'])!=2048 or head['total_head_updates']!=4096
                or head['mode']!=mode or len(saved['optimizer']['state'])!=len(list(checker.parameters()))
                or any(int(s['step'])!=2048 for s in saved['optimizer']['state'].values())
                or any(not torch.isfinite(v).all() for v in head['state'].values())
                or any(not torch.equal(v,saved['head'][k]) for k,v in head['state'].items())):
            raise ValueError('Budget/Adam/head mismatch')
        with np.load(folder/'predictions.npz',allow_pickle=False) as z:
            for role,count in [('train',1836),('development',40)]:
                if z[role+'_logits'].shape!=(3,count,16) or not np.isfinite(z[role+'_logits']).all():
                    raise ValueError('Invalid saved predictions')
            calculated=summarize({role:z[role+'_logits'] for role in ('train','development')},arrays)
        if any(r[k]!=v for k,v in calculated.items()): raise ValueError('Saved command/event metrics differ')
        arms[mode]=r
    calculated=decide(arms,read(HERE/'source_report.json')); pair=read(out/'pair_report.json')
    if any(pair[k]!=v for k,v in calculated.items()): raise ValueError('Protected-state/selection rule differs')
    return {'verified':True,'both_head_and_adam_budgets_checked':True,'saved_predictions_and_guard_rules_recomputed':True,
            'source_pose_model_loaded_or_resampled':False,'automatic_following_action':False}


if __name__=='__main__':
    import json
    p=argparse.ArgumentParser(); p.add_argument('output',type=Path); a=p.parse_args(); print(json.dumps(verify(a.output),indent=2))

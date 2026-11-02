"""Verify saved head, metrics and2048-update budget; no source model sampling."""
import argparse
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE / 'code'))


def verify(out):
    import numpy as np
    import torch
    from train_head import read,sha,summarize
    from dino_gripper_sidecar import GripperSidecar,SOURCE_SHA
    out=Path(out); report=read(out / 'report.json')
    payload=torch.load(out / 'gripper.pt',map_location='cpu',weights_only=False)
    latest=torch.load(out / 'latest.pt',map_location='cpu',weights_only=False)
    head=GripperSidecar(); head.load_state_dict(payload['state'],strict=True)
    if (not report['execution_complete'] or report['updates']!=2048 or latest['update']!=2048
            or payload['source_checkpoint_sha256']!=SOURCE_SHA or payload['updates']!=2048
            or len(latest['history'])!=2048 or any(row['update']!=i+1 for i,row in enumerate(latest['history']))
            or len(latest['optimizer']['state'])!=len(list(head.parameters()))
            or any(not torch.equal(v,latest['head'][k]) for k,v in payload['state'].items())
            or any(int(s['step'])!=2048 for s in latest['optimizer']['state'].values())
            or sha(out / 'gripper.pt')!=report['gripper_file_sha256']):
        raise ValueError('Saved head/Adam/identity/budget mismatch')
    with np.load(HERE / 'inputs.npz',allow_pickle=False) as z: arrays={k:z[k] for k in z.files}
    with np.load(out / 'predictions.npz',allow_pickle=False) as z:
        metrics=summarize({role:z[role+'_logits'] for role in ('train','development')},arrays)
    if any(report[k]!=v for k,v in metrics.items()): raise ValueError('Saved command/event metrics differ')
    return {'verified':True,'head_and_adam_and_budget_checked':True,'command_event_metrics_recomputed':True,
            'pose_network_loaded_or_resampled':False,'automatic_following_action':False}


if __name__=='__main__':
    import json
    p=argparse.ArgumentParser(); p.add_argument('output',type=Path); a=p.parse_args()
    print(json.dumps(verify(a.output),indent=2))

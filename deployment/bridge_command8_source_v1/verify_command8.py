"""Read saved command arrays/state/Adam; no model construction or sampling."""
import argparse
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE/'code'),str(HERE/'vendor/code'),str(HERE/'vendor')]


def verify(out,*,full=True):
    import numpy as np
    import torch
    from command8_runtime import read,sha,load_numeric
    from command8_metrics import summarize
    from completed_policy_export import model_sha
    from step_training_state import _digest
    out=Path(out);r=read(out/'report.json');p=read(HERE/'protocol.json')
    if (not r['execution_complete'] or r['completed_updates']!=8192
            or r['effective_action_representation']!=p['action_representation'] or r['target_supervised']):raise ValueError('Incomplete or wrong-label endpoint')
    a,rows=load_numeric();lookup={(v['role'],v['window_index']):i for i,v in enumerate(rows)}
    role_gates=[]
    with np.load(out/'predictions.npz',allow_pickle=False) as z:
        for role in ('train','validation'):
            ids=z[role+'_window_indices'];slots=np.array([lookup[(role,int(i))] for i in ids])
            expected_ids=[r['window_index'] for r in read(HERE/'vendor/selection.json')['rows'] if r['role']==role]
            if (ids.tolist()!=expected_ids or len(ids)!=(128 if role=='train' else 64) or not np.array_equal(z[role+'_targets'],a['targets'][slots])
                    or not np.array_equal(z[role+'_masks'],a['masks'][slots])):raise ValueError('Evaluation role/command labels differ')
            calculated=summarize(z[role+'_predictions'],z[role+'_targets'],z[role+'_masks'],a['train_mean_command'])
            from bcap_verify import same
            same(calculated,r['metrics'][role])
            role_gates.append(calculated['source_engineering_review_gate_passed'])
    if (r['source_engineering_review_gate_passed']!=all(role_gates)
            or r['formal_matrix_admitted'] or r['zero_shot_transfer_proven']):
        raise ValueError('Report gate or scientific scope differs')
    history=read(out/'history.json');plan=np.load(HERE/'batches.npy',allow_pickle=False)
    if len(history)!=8192 or not read(out/'zero_update_guard.json')['passed']:raise ValueError('Missing complete update history/guard')
    for i,row in enumerate(history):
        if row['update']!=i+1 or row['indices']!=plan[i].tolist() or row['lr']!=[1e-5]*3:raise ValueError('Saved plan/budget/LR differs')
        if not np.isfinite([row[k] for k in ('loss','pose_loss','gripper_loss','gradient_norm')]).all():raise ValueError('Nonfinite training scalar')
    if full:
        policy=torch.load(out/'policy.pt',map_location='cpu',weights_only=False,mmap=True)
        latest=torch.load(out/'latest_state.pt',map_location='cpu',weights_only=False,mmap=True)
        if (policy['format']!='bridge-command8-source-policy-v1' or policy['completed_updates']!=8192
                or policy['effective_action_representation']!=p['action_representation']
                or policy['contract']!=latest['contract'] or latest['contract_sha256']!=_digest(latest['contract'])
                or policy['contract']['protocol_sha256']!=sha(HERE/'protocol.json')
                or policy['contract']['inputs_receipt_sha256']!=sha(HERE/'inputs_receipt.json')
                or policy['source_engineering_review_gate_passed']!=all(role_gates)
                or policy['target_supervised'] or policy['formal_matrix_admitted'] or policy['zero_shot_transfer_proven']
                or latest['plan']['contract']!={'identity':sha(HERE/'batches.npy'),'shape':[8192,8]}
                or latest['plan']['completed_updates']!=8192 or latest['extra']['history']!=history
                or {k:model_sha(v) for k,v in policy['state'].items()}!=r['final_identity']
                or {k:model_sha(v) for k,v in latest['model'].items()}!=r['final_identity']
                or sha(out/'policy.pt')!=r['policy_sha256']):raise ValueError('Full endpoint/atomic state identity differs')
        from step_training_state import _validate_rng,UpdateSchedule
        _validate_rng(latest['rng'])
        if (len(latest['rng']['torch_cuda'])!=1
                or latest['schedule']!=UpdateSchedule(total_updates=8192,base_lrs=[1e-5]*3,kind='constant',warmup_updates=0).contract):
            raise ValueError('Full source GPU RNG or fixed schedule missing')
        optimizer=latest['optimizer']
        if len(optimizer['param_groups'])!=3 or len(optimizer['state'])!=sum(len(g['params']) for g in optimizer['param_groups']):raise ValueError('Incomplete AdamW groups')
        for state in optimizer['state'].values():
            if int(state['step'])!=8192 or any(not torch.isfinite(v).all() for v in state.values() if isinstance(v,torch.Tensor)):raise ValueError('Nonfinite/incomplete AdamW state')
    return {'verified':True,'saved_command_metrics_plan_and_labels_checked':True,
        'full_policy_and_adam_readback':full,'updates':8192,'new_sampling':0,
        'target_supervised':False,'zero_shot_transfer_proven':False,'automatic_following_action':False}


if __name__=='__main__':
    import json
    ap=argparse.ArgumentParser();ap.add_argument('output',type=Path);ap.add_argument('--small-only',action='store_true');args=ap.parse_args()
    print(json.dumps(verify(args.output,full=not args.small_only),indent=2))

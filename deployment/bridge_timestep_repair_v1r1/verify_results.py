"""Saved arrays/history/state checks; no model construction or sampling."""
import argparse
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE / 'code'), str(HERE / 'vendor/code'), str(HERE / 'vendor')]


def verify(out, *, full=True):
    import numpy as np
    import torch
    from bcap_runtime import read
    from bcap_core import summarize
    from bcap_verify import same
    from bridge_timestep_repair import timestep_weights, check_matched_random_stream
    from completed_policy_export import model_sha
    from step_training_state import _digest
    out = Path(out)
    source = read(HERE / 'source_report.json')
    alpha_bars = torch.tensor(read(HERE / 'vendor/profile_sample.json')['schedule_values']['alpha_bars'],dtype=torch.float32)
    histories = []
    batches = np.load(HERE / 'repair_batches.npy',allow_pickle=False)
    for mode in ('uniform','min_snr5_grid_mean_one'):
        folder = out / mode
        report = read(folder / 'report.json'); history = read(folder / 'history.json')
        if (not report['execution_complete'] or report['additional_updates'] != 8192
                or report['mode'] != mode or report['initial_identity'] != source['final_identity']
                or len(history) != 8192 or not read(folder / 'zero_update_guard.json')['passed']):
            raise ValueError('Missing/mismatched completed arm: ' + mode)
        for i,row in enumerate(history):
            weights = timestep_weights(alpha_bars,torch.tensor(row['timesteps']),mode)
            if row['update'] != i+1 or row['indices'] != batches[i].tolist() or row['lr'] != [1e-5]*3:
                raise ValueError('Saved budget/plan/LR differs')
            for key, value in [('timestep_weight_mean',weights.mean()),('timestep_weight_min',weights.min()),('timestep_weight_max',weights.max())]:
                if not np.isclose(row[key],float(value),atol=1e-6,rtol=1e-6):
                    raise ValueError('Saved loss weighting differs from fixed schedule')
            if not np.isfinite([row[k] for k in ('loss','pose_loss','gripper_loss','gradient_norm')]).all():
                raise ValueError('Nonfinite saved training scalar')
        with np.load(folder / 'predictions_and_targets.npz',allow_pickle=False) as z, np.load(HERE / 'source_arrays.npz',allow_pickle=False) as old:
            if (z['after'].shape != (3,192,16,8) or not np.isfinite(z['after']).all()
                    or not np.array_equal(z['targets'],old['targets']) or not np.array_equal(z['masks'],old['masks'])):
                raise ValueError('Saved evaluation population/arrays differ')
            same(summarize({'before':old['after'],'after':z['after'],'static':old['static']},z['targets'],z['masks'],read(HERE / 'vendor/selection.json')),report['metrics'])
        if full:
            latest = torch.load(folder / 'latest_state.pt',map_location='cpu',weights_only=False,mmap=True)
            policy = torch.load(folder / 'final_policy.pt',map_location='cpu',weights_only=False,mmap=True)
            if (latest['contract_sha256'] != _digest(latest['contract']) or latest['contract'] != policy['contract']
                    or latest['plan']['completed_updates'] != 8192 or policy['additional_updates'] != 8192
                    or policy['source_completed_updates'] != 65536 or len(latest['optimizer']['param_groups']) != 3
                    or latest['extra']['history'] != history
                    or {k:model_sha(v) for k,v in latest['model'].items()} != report['final_identity']
                    or {k:model_sha(v) for k,v in policy['state'].items()} != report['final_identity']):
                raise ValueError('Full saved optimizer/policy/budget differs')
            for state in latest['optimizer']['state'].values():
                if int(state['step']) != 8192 or any(not torch.isfinite(v).all() for v in state.values() if isinstance(v,torch.Tensor)):
                    raise ValueError('Incomplete/nonfinite Adam endpoint')
        histories.append(history)
    stream = check_matched_random_stream(*histories)
    return {'verified':True,'arrays_metrics_and_timestep_weights_recomputed':True,
            'full_policy_adam_checked':full,'random_stream_check':stream,'automatic_following_action':False}


if __name__ == '__main__':
    import json
    p = argparse.ArgumentParser(); p.add_argument('output',type=Path); p.add_argument('--small-only',action='store_true')
    args = p.parse_args(); print(json.dumps(verify(args.output,full=not args.small_only),indent=2))

"""Verify resource/interface output only; never score or launch a policy."""
import argparse
import json
from pathlib import Path
import numpy as np
from native_data import sha


def validate(out,bundle):
    p=json.loads((bundle/'protocol.json').read_text());r=json.loads((out/'report.json').read_text())
    for name,digest in p['bundle_sources_sha256'].items():
        if sha(bundle/name)!=digest: raise ValueError('Changed package '+name)
    if not r['preflight_passed'] or r['errors'] or r['protocol_sha256']!=sha(bundle/'protocol.json'):
        raise ValueError('Incomplete or changed preflight')
    if any(r[k] for k in ('formal_training_submitted','development_scored','final50_accessed','trained_weights_loaded','clip_training_executed')):
        raise ValueError('Out-of-scope preflight')
    if r['unique_preflight_updates']!=8 or r['recomputed_replay_updates']!=1 or r['isolated_partial_probe_updates']!=1:
        raise ValueError('Incorrect preflight budget')
    base=json.loads((bundle/'base_protocol.json').read_text());old=json.loads((bundle/'project_contract.json').read_text())
    reference=json.loads((bundle/'eight_block_reference.json').read_text())
    if (sha(bundle/'base_protocol.json')!=p['base_protocol_sha256']
            or sha(bundle/'project_contract.json')!=p['original_project_contract_sha256']
            or sha(bundle/'eight_block_reference.json')!=p['eight_block_report_sha256']
            or r['data']!=reference['data']):
        raise ValueError('Reference8 data/source differs')
    if (r['initial_backend_sha256']!=base['initial_backend_sha256'] or r['normalizer_sha256']!=base['normalizer_sha256']
            or r['sample_order_sha256']!=old['sample_order_sha256'] or r['sample_plan_sha256']!=old['sample_plan_sha256']
            or r['cache_report_sha256']!=old['cache_report_sha256'] or r['eight_block_report_sha256']!=p['eight_block_report_sha256']):
        raise ValueError('Common reference contract differs')
    initial=r['initialization']
    if not (initial['common_parameter_objects_preserved'] and initial['rng_unchanged'] and initial['active_cross_attention_blocks']==0
            and initial['remaining_frontend_parameters']==67648 and not initial['active_text_conditioning']
            and initial['original_frontend_parameters']-initial['removed_block_parameters']==67648):
        raise ValueError('Shared initialization/ablation differs')
    for k in ('fresh_initialization_only','zero_block_formula_exact','shared_text_unchanged','ema_frozen',
              'projection_updated','output_norm_updated','ema_projection_updated'):
        if not r[k]: raise ValueError('Failed audit '+k)
    audit=r['loss_and_gradient_audit']
    required=('original_compute_loss_used','common_noise_exact','common_timesteps_exact','epsilon_objective_loss_exact',
              'representative_backend_gradient_exact','frontend_and_backend_gradients_valid','adapter_gradients_valid','projection_gradients_valid')
    if not all(audit.get(k) is True for k in required): raise ValueError('Loss/gradient failed')
    d=r['disk_resume']
    if (not d['exact'] or d['loss_before']!=d['loss_after'] or d['state_before_sha256']!=d['state_after_sha256']
            or d['replay_completed_updates']!=2 or sha(out/'replay.pt')!=d['replay_checkpoint_sha256']):
        raise ValueError('Full restore audit differs')
    if len(r['losses'])!=8 or len(r['warm_durations'])!=5 or not all(np.isfinite(x) and x>=0 for x in r['losses']):
        raise ValueError('Missing warm updates')
    if (not np.isclose(r['warm_update_seconds'],np.mean(r['warm_durations']),rtol=1e-12,atol=1e-12)
            or r['warm_update_seconds']<=0 or not np.isfinite(r['peak_reserved_gib']) or r['peak_reserved_gib']<=0):
        raise ValueError('Invalid time estimate')
    if not np.isclose(r['estimated_formal_training_hours_excluding_io_and_rollout'],r['warm_update_seconds']*84168/3600):
        raise ValueError('Wrong formal estimate')
    if (r['partial38_probe']['microbatch_sizes']!=[8,8,8,8,6] or not r['partial38_probe']['finite_loss']
            or r['ema_optimization_step_after_disposable_probe']!=9 or r['active_cross_attention_blocks']!=0):
        raise ValueError('Partial-batch/counter failure')
    if sha(out/'smoke_actions.npz')!=r['smoke_actions_sha256']: raise ValueError('Changed smoke actions')
    with np.load(out/'smoke_actions.npz',allow_pickle=False) as a:
        if a['action'].shape!=(2,8,2) or a['action_pred'].shape!=(2,16,2) or not np.isfinite(a['action_pred']).all() or not np.array_equal(a['action'],a['action_pred'][:,1:9]):
            raise ValueError('Bad sampling shape/slice')
    return dict(preflight_verified=True,warm_update_seconds=r['warm_update_seconds'],peak_reserved_gib=r['peak_reserved_gib'],
        estimated_formal_training_hours_excluding_io_and_rollout=r['estimated_formal_training_hours_excluding_io_and_rollout'],
        formal_training_submitted=False,task_performance_verdict=None)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path);a=p.parse_args()
    print('ZERO BLOCK PREFLIGHT VERIFIED; no performance verdict or automatic formal submission')
    print(json.dumps(validate(a.output,Path(__file__).resolve().parent),indent=2))

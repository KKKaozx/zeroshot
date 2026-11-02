"""Verify a completed DINO segment and its fixed endpoint comparison, no model load."""
import argparse
import json
import math
from pathlib import Path

from cache_tools import sha
from formal_binding import check_sources,read
from compare_backbones import compare


def equal(a,b):
    if isinstance(a,dict):return set(a)==set(b) and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,list):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    if isinstance(a,float):return isinstance(b,(float,int)) and math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12)
    return a==b


def validate(out,bundle):
    c = check_sources(bundle)
    base = read(bundle/'base_protocol.json')
    r = read(out/'report.json')
    segment = r['segment']
    if segment not in (1,2):raise ValueError('Wrong DINO segment')
    target = c['segment_endpoints'][segment-1]
    if (r['errors'] or not r['segment_completed'] or r['completed_updates'] != target
            or r['target_updates'] != target or r['total_project_budget'] != 84168
            or r['project_contract_sha256'] != sha(bundle/'project_contract.json')
            or r['protocol_sha256'] != sha(bundle/'base_protocol.json')
            or any(r[k] for k in ('final_set_accessed','clip_training_executed',
                                  'dino_training_executed','trained_native_weights_loaded','automatic_submission'))):
        raise ValueError('Incomplete or wrong-scope DINO segment')
    for key in ('sample_plan_sha256','sample_order_sha256','cache_report_sha256','dino_backbone_binding_sha256'):
        if r[key] != c[key]:raise ValueError('DINO identity differs: '+key)
    if (r['initial_backend_sha256'] != base['initial_backend_sha256']
            or r['normalizer_sha256'] != base['normalizer_sha256']
            or r['data'] != read(bundle/'cache_report_reference.json')['training_data']
            or not r['frozen_dino_cache_verified'] or not r['frozen_text_unchanged']
            or r['active_cross_attention_blocks'] != 8 or r['lr_horizon_updates'] != 512400):
        raise ValueError('Data, backbone/frontend structure or shared training settings differ')
    if (sha(out/'backbone_binding.json') != c['dino_backbone_binding_sha256']
            or read(out/'backbone_binding.json') != read(bundle/'cache_binding_reference.json')):
        raise ValueError('Saved DINO/cache identity binding changed')
    audit = r['startup_checks']
    required_audit = {'original_compute_loss_used','common_noise_exact','common_timesteps_exact',
        'epsilon_objective_loss_exact','representative_backend_gradient_exact',
        'frontend_and_backend_gradients_valid','adapter_gradients_valid','projection_gradients_valid'}
    if (not audit['disk_resume_replay_exact'] or set(audit['loss_audit']) != required_audit
            or not all(audit['loss_audit'].values())
            or audit['loss_before_restore'] != audit['loss_after_restore']
            or audit['state_digest_before_restore'] != audit['state_digest_after_restore']):
        raise ValueError('Complete first disk restore failed')
    side = read(out/'latest.json')
    if (sha(out/'latest.pt') != side['sha256'] or side['sha256'] != r['checkpoint_sha256']
            or Path(r['checkpoint']).name != 'latest.pt'
            or Path(r['checkpoint']).parent.resolve() != out.resolve()
            or side['startup_checks'] != audit or side['segment'] != segment):
        raise ValueError('DINO complete checkpoint identity differs')
    for key in ('completed_updates','project_contract_sha256','protocol_sha256',
                'sample_plan_sha256','dino_backbone_binding_sha256'):
        if side[key] != r[key]:raise ValueError('DINO sidecar changed: '+key)
    if (r['initial_completed_updates'] > 0 or segment == 2) and not r.get('complete_resume_state_exact'):
        raise ValueError('Resumed segment lacks full exact state restore')
    lower = 0 if segment == 1 else 42084
    if not lower <= r['initial_completed_updates'] < target:
        raise ValueError('Segment begins outside its fixed budget')
    count = 0
    with (out/'updates.jsonl').open() as stream:
        for count,line in enumerate(stream,1):
            row = json.loads(line)
            expected = r['initial_completed_updates']+count
            if row['completed_updates'] != expected or not math.isfinite(row['loss']) or row['loss']<0:
                raise ValueError('DINO update counter/loss changed')
            lr = 1e-4*(expected/500 if expected<500 else .5*(1+math.cos(math.pi*(expected-500)/(512400-500))))
            if not math.isclose(row['lr_for_next_update'],lr,rel_tol=1e-10,abs_tol=1e-12):
                raise ValueError('Original learning-rate schedule changed')
    if count != target-r['initial_completed_updates']:
        raise ValueError('Missing DINO updates')
    if segment == 1:
        if r['training_completed'] or r['development_complete'] or (out/'development.json').exists():
            raise ValueError('Segment1 must not score/select the development set')
        return dict(segment_verified=True,segment=1,completed_updates=42084,
            checkpoint=r['checkpoint'],performance_verdict=None,automatic_following_action=False)
    if not (r['training_completed'] and r['development_complete'] and r['ema_unchanged_during_evaluation']):
        raise ValueError('Endpoint evaluation incomplete')
    v = read(bundle/'visual_protocol.json')
    worker = r['evaluation_worker']
    if (not worker['all_frozen'] or not worker['weights_unchanged']
            or any(worker[k] != v[k] for k in ('model_state_sha256','backbone_weights_sha256','text_sha256','preprocessing'))):
        raise ValueError('Endpoint uses different/nonfrozen visual representation')
    dev = read(out/'development.json')
    if dev != r['development']:raise ValueError('DINO development output differs')
    for row in dev['episodes'][:2]:
        if not (out/row['video']).is_file() or (out/row['video']).stat().st_size==0:
            raise ValueError('Missing preselected video')
    outcome = compare(dev,read(bundle/'clip_development_reference.json'),
                      read(bundle/'native_development_reference.json'),base)
    if not equal(outcome,r['backbone_comparison']) or not equal(outcome,read(out/'comparison.json')):
        raise ValueError('Backbone paired comparison differs')
    return dict(segment_verified=True,segment=2,completed_updates=84168,
                comparison=outcome,automatic_following_action=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('output',type=Path)
    args = p.parse_args()
    result = validate(args.output,Path(__file__).resolve().parent)
    print('DINO SEGMENT OUTPUT VERIFIED; fixed development reward only, no robot grasp verdict')
    print(json.dumps(result,indent=2))

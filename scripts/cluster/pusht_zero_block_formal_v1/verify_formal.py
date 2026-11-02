"""Verify complete zero-control output and recompute its development comparison."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
from native_data import sha
from base_verifier import validate_metrics
from pair_metrics import compare


def equal(a,b):
    if isinstance(a,dict):return set(a)==set(b) and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,list):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    if isinstance(a,float):return math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12)
    return a==b


def validate(out,bundle):
    c=json.loads((bundle/'project_contract.json').read_text());base=json.loads((bundle/'base_protocol.json').read_text())
    r=json.loads((out/'report.json').read_text())
    for name,digest in c['bundle_sources_sha256'].items():
        if sha(bundle/name)!=digest:raise ValueError('Changed formal bundle '+name)
    if (r['errors'] or not all(r[k] for k in ('segment_completed','training_completed','development_complete','formal_complete'))
            or r['completed_updates']!=84168 or r['target_updates']!=84168 or r['segment']!=1
            or any(r[k] for k in ('final_set_accessed','clip_training_executed','trained_native_weights_loaded','automatic_submission'))
            or r['project_contract_sha256']!=sha(bundle/'project_contract.json') or r['protocol_sha256']!=sha(bundle/'base_protocol.json')):
        raise ValueError('Incomplete/changed formal endpoint')
    for key in ('sample_plan_sha256','sample_order_sha256','cache_report_sha256'):
        if r[key]!=c[key]:raise ValueError('Wrong '+key)
    if r['initial_backend_sha256']!=base['initial_backend_sha256'] or r['normalizer_sha256']!=base['normalizer_sha256']:
        raise ValueError('Shared initialization/normalizer differs')
    eight=json.loads((bundle/'eight_block_reference.json').read_text())
    if r['data']!=eight['data'] or not r['frozen_clip_cache_verified'] or not r['frozen_text_unchanged']:
        raise ValueError('Data/frozen input changed')
    b=json.loads((out/'preflight_binding.json').read_text())
    if (sha(out/'preflight_binding.json')!=r['preflight_binding_sha256'] or b['preflight_job']!=219260
            or b['formal_contract_sha256']!=sha(bundle/'project_contract.json')
            or b['preflight_protocol_sha256']!=c['preflight_protocol_sha256']
            or r['zero_block_initialization']!=b['initialization'] or b['preflight_weights_used']):
        raise ValueError('Preflight identity changed')
    if (r['active_cross_attention_blocks']!=0 or r['zero_block_initialization']['remaining_frontend_parameters']!=67648
            or r['zero_block_initialization']['active_text_conditioning']):raise ValueError('Incorrect zero control')
    audit=r['startup_checks']
    if not audit['disk_resume_replay_exact'] or not all(audit['loss_audit'].values()):raise ValueError('Startup audit failed')
    side=json.loads((out/'latest.json').read_text())
    if (sha(out/'latest.pt')!=side['sha256'] or side['sha256']!=r['checkpoint_sha256']
            or side['completed_updates']!=84168 or side['project_contract_sha256']!=r['project_contract_sha256']
            or side['preflight_binding_sha256']!=r['preflight_binding_sha256'] or side['startup_checks']!=audit):
        raise ValueError('Checkpoint identity differs')
    logs=[json.loads(line) for line in (out/'updates.jsonl').read_text().splitlines()]
    if r['initial_completed_updates']>0 and not r.get('complete_resume_state_exact'):
        raise ValueError('Resumed run lacks exact full-state restore')
    if len(logs)!=84168-r['initial_completed_updates']:raise ValueError('Missing updates')
    for expected,row in enumerate(logs,r['initial_completed_updates']+1):
        if row['completed_updates']!=expected or not math.isfinite(row['loss']) or row['loss']<0:raise ValueError('Bad update trace')
        lr=1e-4*(expected/500 if expected<500 else .5*(1+math.cos(math.pi*(expected-500)/(512400-500))))
        if not math.isclose(row['lr_for_next_update'],lr,rel_tol=1e-10,abs_tol=1e-12):raise ValueError('Learning-rate trace changed')
    dev=json.loads((out/'development.json').read_text())
    if dev!=r['development']:raise ValueError('Development differs')
    validate_metrics(dev,base)
    if not (r['evaluation_worker']['all_frozen'] and r['evaluation_worker']['weights_unchanged'] and r['ema_unchanged_during_evaluation']):
        raise ValueError('Evaluation freeze failed')
    for ep in dev['episodes'][:2]:
        if not (out/ep['video']).is_file() or (out/ep['video']).stat().st_size==0:raise ValueError('Missing fixed video')
    outcome=compare(dev,eight['development'],base,c)
    if not equal(outcome,r['zero_vs_eight']) or not equal(outcome,json.loads((out/'comparison.json').read_text())):
        raise ValueError('Comparison changed')
    return dict(output_verified=True,comparison=outcome,training_updates=84168,automatic_following_action=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path);a=p.parse_args()
    print('ZERO BLOCK FORMAL OUTPUT VERIFIED; development mechanism ablation only')
    print(json.dumps(validate(a.output,Path(__file__).resolve().parent),indent=2))

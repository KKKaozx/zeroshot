"""Verify full cluster output and recompute ALL development admission on CPU."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from bridge_spatial_pair_metrics_v1 import summarize_pair,admission

CODE=Path(__file__).resolve().parent


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda:stream.read(8*1024**2),b''):h.update(part)
    return h.hexdigest()


def compare(a,b):
    if isinstance(a,dict):
        assert set(a)==set(b)
        for k in a:compare(a[k],b[k])
    elif isinstance(a,list):
        assert len(a)==len(b)
        for x,y in zip(a,b):compare(x,y)
    elif isinstance(a,float):assert abs(a-b)<=1e-9,(a,b)
    else:assert a==b,(a,b)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('directory',type=Path)
    args=ap.parse_args();d=args.directory
    r=json.loads((d/'report.json').read_text());contract=json.loads((CODE/'formal_contract.json').read_text())
    assert r['stage']=='spatial_formal_pair' and r['execution_checks_passed'] and r['generalization_verdict_available']
    assert r['preflight_job']==210291 and r['protocol_sha256']==contract['protocol_sha256']
    assert r['formal_source_sums_sha256']==sha(CODE/'SHA256SUMS')
    assert r['formal_contract_sha256']==sha(CODE/'formal_contract.json')
    assert r['initial_checkpoint_sha256']==contract['initial_checkpoint_sha256']
    for k in ('final25_payloads_read','gripper_trained','old40_or_consumed13_rescored','downloads',
              'weights_selected_by_development','CLIP_training_allowed'):assert not r[k],k
    for k in ('both_weights_frozen_before_development_scoring','clip_unchanged','gripper_unchanged',
              'training_token_cache_reused'):assert r[k],k
    assert r['optimization_operations']==8192 and r['new_training_tokens_encoded']==0
    assert r['development_episodes']==10 and r['development_windows']==50 and r['peak_reserved_gib']<=7.5
    assert r['numeric_inputs']['episodes']==405 and r['numeric_inputs']['windows']==1836
    assert [v['arm'] for v in r['training_runs']]==['control','candidate']
    a,b=r['training_runs'];assert a['initial_base_sha256']==b['initial_base_sha256']
    assert a['first8_losses'][0]==b['first8_losses'][0]
    assert b['trainable_parameters']-a['trainable_parameters']==1026
    for run in r['training_runs']:
        assert run['logical_optimizer_updates']==4096 and run['initial_optimizer_empty']
        assert run['base_parameters_changed'] and run['checkpoint_roundtrip_exact']
        assert run['new_score_parameters_changed']==(run['arm']=='candidate')
        assert sha(d/run['checkpoint'])==run['checkpoint_sha256']
        with np.load(d/f"{run['arm']}_losses.npz",allow_pickle=False) as z:
            losses=z['losses'].copy();assert losses.shape==(4096,) and np.isfinite(losses).all()
            assert losses[:8].tolist()==run['first8_losses']
        lines=[json.loads(line) for line in (d/f"{run['arm']}_updates.jsonl").read_text().splitlines()]
        assert [v['update'] for v in lines]==list(range(1,4097))
        assert np.array_equal(losses,[v['loss'] for v in lines])
        assert all(v['learning_rate']==1e-5 for v in lines)
    rows=json.loads((d/'validation_rows.json').read_text())
    with np.load(d/'validation_inputs.npz',allow_pickle=False) as z:
        targets=z['targets'].copy();assert z['state_features'].shape==(50,9) and z['current'].shape==(50,1)
    assert targets.shape==(50,16,8)
    results={}
    for name in ('unchanged207558','control','candidate'):
        with np.load(d/f'{name}_full16.npz',allow_pickle=False) as z:
            assert np.array_equal(z['targets'],targets) and np.array_equal(z['sampling_seeds'],[1101,1102,1103])
            assert z['predictions'].shape==(3,50,16,7)
            results[name]=summarize_pair(z['predictions'],targets,rows)
    identity=np.zeros((1,50,16,7));identity[...,6]=1.
    results['static']=summarize_pair(identity,targets,rows)
    compare(results,r['results']);decision=admission(results);compare(decision,r['adoption'])
    print('SPATIAL FORMAL OUTPUT VERIFIED; offline same-task pose only')
    for name in ('unchanged207558','control','candidate','static'):
        print(name,json.dumps({part:results[name][part]['episode_equal'] for part in ('first','full16')}))
    print('ADMISSION',json.dumps(decision))


if __name__=='__main__':main()

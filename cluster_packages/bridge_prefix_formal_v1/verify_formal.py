"""Validate completed formal output and recompute NEW-development results onCPU."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from bridge_prefix_pair_metrics_v1 import summarize,admission


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
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
    args=ap.parse_args();d=args.directory;r=json.loads((d/'report.json').read_text())
    contract=json.loads((Path(__file__).parent/'formal_contract.json').read_text())
    assert r['stage']=='formal_pair' and r['execution_checks_passed'] and r['generalization_verdict_available']
    assert r['protocol_sha256']==contract['protocol_sha256']
    assert r['initial_checkpoint_sha256']==contract['initial_checkpoint_sha256']
    assert r['formal_source_sums_sha256']==sha(Path(__file__).parent/'SHA256SUMS')
    assert r['preflight_job']==209420 and r['optimization_operations']==8192
    assert r['development_episodes']==10 and r['development_windows']==50
    for k in ('final25_payloads_read','old40_or_consumed13_rescored','gripper_trained',
              'downloads','weights_selected_by_development'):assert r[k] is False,k
    for k in ('clip_unchanged','gripper_unchanged','both_weights_frozen_before_development_scoring',
              'old_visual_cache_reused','delta_visual_cache_reused'):assert r[k],k
    assert [v['arm'] for v in r['training_runs']]==['control','candidate']
    assert r['training_runs'][0]['initial_trainable_sha256']==r['training_runs'][1]['initial_trainable_sha256']
    assert r['training_runs'][0]['first8_losses'][0]==r['training_runs'][1]['first8_losses'][0]
    for run in r['training_runs']:
        assert run['logical_optimizer_updates']==4096 and run['K1_updates']==run['K16_updates']==2048
        assert run['checkpoint_roundtrip_exact'] and run['optimizer_initial_state_empty']
        assert sha(d/run['checkpoint'])==run['checkpoint_sha256']
        with np.load(d/f"{run['arm']}_losses.npz",allow_pickle=False) as z:
            assert z['losses'].shape==(4096,) and np.isfinite(z['losses']).all()
            assert np.array_equal(z['requested_horizons'],np.tile([16,1],2048))
    rows=json.loads((d/'validation_rows.json').read_text());results={}
    with np.load(d/'validation_inputs.npz',allow_pickle=False) as z:targets=z['targets'].copy()
    assert targets.shape==(50,16,8)
    for method in ('unchanged207558','control','candidate'):
        with np.load(d/f'{method}_full16.npz',allow_pickle=False) as z:
            assert np.array_equal(z['targets'],targets)
            assert np.array_equal(z['sampling_seeds'],[1101,1102,1103])
            assert z['predictions'].shape==(3,50,16,7)
            results[method]=summarize(z['predictions'],targets,rows)
        if method!='unchanged207558':
            with np.load(d/f'{method}_K1_auxiliary.npz',allow_pickle=False) as z:
                assert z['predictions'].shape==(3,50,1,7) and np.array_equal(z['targets'],targets[:,:1])
                assert np.array_equal(z['sampling_seeds'],[1101,1102,1103])
                compare(summarize(z['predictions'],targets[:,:1],rows),r['auxiliary_K1'][method])
    identity=np.zeros((1,50,16,7));identity[...,6]=1.
    results['static']=summarize(identity,targets,rows)
    compare(results,r['results_full16']);compare(admission(results),r['adoption'])
    print('PREFIX FORMAL OUTPUT VERIFIED; offline same-task pose results only')
    for name,method in results.items():print(name,json.dumps(method['episode_equal']))
    print('ADMISSION',json.dumps(r['adoption']))


if __name__=='__main__':main()
